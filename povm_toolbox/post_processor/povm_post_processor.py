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
from numpy.random import Generator, default_rng
from qiskit.quantum_info import SparsePauliOp

from povm_toolbox.quantum_info.base import BaseDual, BasePOVM
from povm_toolbox.sampler import POVMPubResult


class POVMPostProcessor:
    """A common POVM result post-processor."""

    def __init__(
        self,
        povm_sample: POVMPubResult,
        dual: BaseDual | None = None,
    ) -> None:
        """Initialize the POVM post-processor.

        Args:
            povm_sample: a result from a POVM sampler run.
            dual: the subclass of :class:`.BaseDual` that will be used to
                build the dual frame to the POVM of ``povm_sample``. The dual
                frame is then used to compute the decomposition weights of any
                observable.

        Raises:
            ValueError: If the provided ``dual`` is not a dual frame to the POVM
                used to produce ``povm_sample``.
        """
        self._povm = povm_sample.metadata.povm_implementation.definition()
        self.counts: np.ndarray = povm_sample.get_counts()  # type: ignore
        # TODO: find a way to avoid the type ignore

        if (dual is not None) and (not dual.is_dual_to(self._povm)):
            raise ValueError(
                "The ``dual`` argument is not valid. It is not a dual"
                " frame to the POVM stored in ``povm_sample``."
            )

        self._dual = dual

    @property
    def povm(self) -> BasePOVM:
        """Return the POVM that was used to sample outcomes."""
        return self._povm

    @property
    def dual(self) -> BaseDual:
        """Return the dual that is used.

        .. warning::
            If the dual frame is not already built, this could be computationally demanding.
        """
        if self._dual is None:
            dual_class = self.povm.default_dual_class
            self._dual = dual_class.build_dual_from_frame(self.povm)
        return self._dual

    @dual.setter
    def dual(self, new_dual: BaseDual):
        if not new_dual.is_dual_to(self.povm):
            raise ValueError(
                "The provided ``dual`` instance is not valid. It is not a dual"
                " frame to the POVM used to obtained the post-processing results."
            )
        self._dual = new_dual

    def get_decomposition_weights(
        self, observable: SparsePauliOp, outcome_set: set[Any]
    ) -> dict[Any, float]:
        r"""Get the decomposition weights of ``observable`` into the elements of ``self.povm``.

        Given an observable :math:`O` which is in the span of a given POVM, one
        can write the observable :math:`O` as the weighted sum of the POVM effects,
        :math:`O = \sum_k w_k M_k` for real weights :math:`w_k` and where :math:`k`
        labels the outcomes.

        Args:
            observable: the observable to be decomposed into the POVM effects.
            outcome_set: set of outcome labels indicating which decomposition
                weights are queried. An outcome of a :class:`.ProductPOVM` is
                labeled by a tuple of integers for instance. For a :class:`.MultiQubitPOVM`,
                an outcome is simply labeled by an integer.

        Returns:
            A dictionary mapping outcome labels to decomposition weights.
        """
        return dict(self.dual.get_omegas(observable, outcome_set))  # type: ignore

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
        std = 0.0

        for outcome in count:
            exp_val += count[outcome] * omegas[outcome]
            std += count[outcome] * omegas[outcome] ** 2

        # Normalize
        exp_val /= shots
        std /= shots

        std = np.sqrt((std - exp_val**2) / (shots - 1))

        return exp_val, std

    def median_of_means(
        self,
        observable: SparsePauliOp,
        num_batches: int,
        loc: int | tuple[int, ...],
        rng: int | Generator | None = None,
    ) -> float:
        """Return the expectation value of a given observable using a 'median of means' estimator.

        Args:
            observable: the observable whose expectation value is queried.
            num_batches: TODO.
            loc: index of the results to use. The index corresponds to the set
                of parameter values that was supplied to the sampler through a
                :class:`.POVMSamplerPub`. If the circuit was not parametrized,
                the index ``loc`` should be 0.
            rng: TODO.

        Raises:
            TypeError: TODO.

        Returns:
            A tuple of (estimated) expectation value and standard deviation.
        """
        if rng is None:
            rng = default_rng()
        elif isinstance(rng, int):
            rng = default_rng(rng)
        elif not isinstance(rng, Generator):
            raise TypeError(f"The type of `rng` ({type(rng)}) is not valid.")

        count = self.counts[loc]
        shots = sum(count.values())
        omegas = self.get_decomposition_weights(observable, set(count.keys()))

        batch_size = shots//num_batches

        sampled_weights = np.zeros(shots)
        idx = 0
        for outcome in count:
            sampled_weights[idx : idx + count[outcome]] = omegas[outcome]
            idx += count[outcome]

        sampled_weights = rng.permutation(sampled_weights)

        batches = sampled_weights[:num_batches*batch_size].reshape((batch_size, num_batches))
        if shots%num_batches != 0:
            last_samples = np.full((1, num_batches), np.nan)
            last_samples[:,:shots%num_batches] = sampled_weights[num_batches*batch_size:]
            batches = np.concatenate((batches,last_samples))
        median_of_means: float = np.median(np.nanmean(batches, axis=0))

        return median_of_means
