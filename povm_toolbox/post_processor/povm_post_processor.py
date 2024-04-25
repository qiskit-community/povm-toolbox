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

from povm_toolbox.sampler.result import POVMPubResult


class POVMPostprocessor:
    """Class to represent a POVM post-processor.."""

    def __init__(
        self,
        povm_sample: POVMPubResult,
        alphas: np.ndarray | None = None,
    ) -> None:
        """Initialize the POVM post-processor.

        Args:
            povm_sample: a result from a POVM sampler run.
            alphas: parameters of the frame superoperator of the POVM.
        """
        self.povm = povm_sample.get_povm()
        self.counts = povm_sample.get_counts()
        if alphas is not None:
            self.povm.alphas = alphas

    def get_expectation_value(
        self, observable: SparsePauliOp, loc: int | tuple[int, ...] | None = None
    ) -> np.ndarray | float:
        """Return the expectation value of a given observable."""
        if loc is not None:
            return self._single_exp_val(observable, loc)

        exp_val = np.zeros(shape=self.counts.shape, dtype=float)
        for idx in np.ndindex(self.counts.shape):
            exp_val[idx] = self._single_exp_val(observable, idx)
        return exp_val

    def _single_exp_val(self, observable: SparsePauliOp, loc: int | tuple[int, ...]) -> float:
        exp_val = 0.0
        count = self.counts[loc]
        # TODO: performance gains to be made in getting the omegas here ?
        omegas = dict(self.povm.get_omegas(observable, set(count.keys())))  # type: ignore
        for outcome in count:
            exp_val += count[outcome] * omegas[outcome]
        exp_val /= sum(count.values())
        return exp_val
