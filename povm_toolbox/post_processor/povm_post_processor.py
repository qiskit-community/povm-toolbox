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

from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from povm_toolbox.quantum_info import BaseDUAL, BasePOVM
from povm_toolbox.sampler import POVMPubResult


class POVMPostProcessor:
    """A common POVM result post-processor."""

    def __init__(
        self,
        povm_sample: POVMPubResult,
        dual: BaseDUAL | None = None,
    ) -> None:
        """Initialize the POVM post-processor.

        Args:
            povm_sample: a result from a POVM sampler run.
            dual_class: the subclass of :class:`.BaseDUAL` that will be used to
                build the dual frame to the POVM of ``povm_sample``. The dual
                frame is then used to compute the decomposition weights of any
                observable.

        Raises:
            ValueError: If the provided `dual` is not a dual frame to the POVM
                used to produce `povm_sample`.
        """
        self._povm = povm_sample.metadata.povm_implementation.definition()
        self.counts: np.ndarray = povm_sample.get_counts()  # type: ignore
        # TODO: find a way to avoid the type ignore

        if (dual is not None) and (not dual.is_dual_to(self._povm)):
            raise ValueError(
                "The `dual` argument is not valid. It is not a dual"
                " frame to the POVM stored in `povm_sample`."
            )

        self._dual = dual

    @property
    def povm(self) -> BasePOVM:
        """Return the POVM that was used to sample outcomes."""
        return self._povm

    @property
    def dual(self) -> BaseDUAL:
        """Return the dual that is used.

        .. warning::
            If the dual frame is not already built, this could be computationally demanding.
        """
        if self._dual is None:
            dual_class = self.povm.default_dual_class
            self._dual = dual_class.build_dual_from_frame(self.povm)
        return self._dual

    def get_decomposition_weights(
        self, observable: SparsePauliOp, outcome_idx: set[Any]
    ) -> dict[Any, float]:
        """Get the decomposition weights of ``observable`` into the elements of ``self.povm``.

        Args:
            observable: the observable to be decomposed into the POVM effects.
            outcome_idx: set of labels indicating which decomposition weights are queried.

        Returns:
            An dictionary of decomposition weights.
        """
        return dict(self.dual.get_omegas(observable, outcome_idx))  # type: ignore

    def get_expectation_value(
        self, observable: SparsePauliOp, loc: int | tuple[int, ...] | None = None
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """Return the expectation value of a given observable and standard deviation of the estimator.

        Args:
            observable: the observable whose expectation value is queried.
            loc: this argument is relevant if multiple sets of parameter values
                were supplied to the sampler in the same :class:`.POVMSamplerPub`.
                The index ``loc`` then corresponds to the set of parameter values
                that was supplied to the sampler through the PUB. If None, the
                expectation value (and standard deviation) for each set of circuit
                parameters is returned.

        Returns:
            A tuple of (estimated) expectation value and standard deviation of the
            estimator if a single value is queried. If all values are queried a
            tuple of two :class:`numpy.ndarray` is returned, the first containing
            the expectation values and the second the standard deviations.
        """
        if loc is not None:
            return self._single_exp_value_and_std(observable, loc)
        if self.counts.shape == (1,):
            return self._single_exp_value_and_std(observable, 0)

        exp_val = np.zeros(shape=self.counts.shape, dtype=float)
        std = np.zeros(shape=self.counts.shape, dtype=float)
        for idx in np.ndindex(self.counts.shape):
            exp_val[idx], std[idx] = self._single_exp_value_and_std(observable, idx)
        return exp_val, std

    def _single_exp_value_and_std(
        self,
        observable: SparsePauliOp,
        loc: int | tuple[int, ...],
    ) -> tuple[float, float]:
        """Return the expectation value of a given observable and standard deviation of the estimator.

        Args:
            observable: the observable whose expectation value is queried.
            loc: index of the results to use. The index corresponds to the set
                of parameter values that was supplied to the sampler through a
                :class:`.POVMSamplerPub`. If the circuit was not parametrized,
                the index ``loc`` should be 0.

        Returns:
            A tuple of (estimated) expectation value and standard deviation.
        """
        count = self.counts[loc]
        shots = sum(count.values())
        # TODO: performance gains to be made when computing the omegas here ?
        # like storing the dict of computed omegas and updating the dict with the
        # missing values that were still never computed.
        omegas = self.get_decomposition_weights(observable, set(count.keys()))

        exp_val = 0.0
        pw2 = 0.0

        for outcome in count:
            exp_val += count[outcome] * omegas[outcome]
            pw2 += count[outcome] * omegas[outcome] ** 2

        # Normalize
        exp_val /= shots
        pw2 /= shots

        std = np.sqrt((pw2 - exp_val**2) / (shots - 1))

        return exp_val, std
