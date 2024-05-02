# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TODO."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from povm_toolbox.quantum_info.base_dual import BaseDUAL
from povm_toolbox.quantum_info.multi_qubit_dual import MultiQubitDUAL
from povm_toolbox.quantum_info.multi_qubit_povm import MultiQubitPOVM
from povm_toolbox.quantum_info.product_dual import ProductDUAL
from povm_toolbox.quantum_info.product_povm import ProductPOVM
from povm_toolbox.sampler import POVMPubResult


class POVMPostProcessor:
    """Class to represent a POVM post-processor.."""

    def __init__(
        self,
        povm_sample: POVMPubResult,
        DUAL_CLASS: type[BaseDUAL] | None = None,
    ) -> None:
        """Initialize the POVM post-processor.

        Args:
            povm_sample: a result from a POVM sampler run.
            alphas: parameters of the frame super-operator of the POVM.
        """
        self.povm = povm_sample.metadata.povm_implementation.definition()
        self.counts: np.ndarray = povm_sample.get_counts()  # type: ignore
        # TODO: find a way to avoid the type ignore

        if DUAL_CLASS is None:
            if isinstance(self.povm, MultiQubitPOVM):
                DUAL_CLASS = MultiQubitDUAL
            elif isinstance(self.povm, ProductPOVM):
                DUAL_CLASS = ProductDUAL
            else:
                raise TypeError
        elif not issubclass(DUAL_CLASS, BaseDUAL):
            raise TypeError

        self.dual = DUAL_CLASS.build_dual_from_frame(self.povm)

    def optimize(self, **options) -> None:
        """Optimize the dual inplace."""
        self.dual.optimize(self.povm, **options)

    def get_expectation_value(
        self, observable: SparsePauliOp, loc: int | tuple[int, ...] | None = None
    ) -> np.ndarray | float:
        """Return the expectation value of a given observable."""
        if loc is not None:
            return self._single_exp_val(observable, loc)
        if self.counts.shape == (1,):
            return self._single_exp_val(observable, 0)

        exp_val = np.zeros(shape=self.counts.shape, dtype=float)
        for idx in np.ndindex(self.counts.shape):
            exp_val[idx] = self._single_exp_val(observable, idx)
        return exp_val

    def _single_exp_val(self, observable: SparsePauliOp, loc: int | tuple[int, ...]) -> float:
        """Return the expectation value of an observable for a given circuit."""
        exp_value, _ = self.get_single_exp_value_and_std(observable, loc)
        return exp_value

    def get_single_exp_value_and_std(
        self,
        observable: SparsePauliOp,
        loc: int | tuple[int, ...] | None = None,
    ) -> tuple[float, float]:
        """Return the expectation value of a given observable."""
        # loc is allowed to be None only if there's only one counter in the counter array
        if loc is None:
            if self.counts.shape == (1,):
                loc = (0,)
            else:
                raise ValueError
        exp_val = 0.0
        std = 0.0
        count = self.counts[loc]
        # TODO: performance gains to be made when computing the omegas here ?
        # like storing the dict of computed omegas and updating the dict with the
        # missing values that were still never computed.
        omegas = dict(self.dual.get_omegas(observable, set(count.keys())))  # type: ignore
        for outcome in count:
            exp_val += count[outcome] * omegas[outcome]
            std += count[outcome] * omegas[outcome] ** 2
        shots = sum(count.values())
        exp_val /= shots
        std /= shots
        std = np.sqrt((std - exp_val**2) / (shots - 1))
        return exp_val, std
