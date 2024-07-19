# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""POVMPostProcessor."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from povm_toolbox.quantum_info.base import BaseDual, BasePOVM
from povm_toolbox.sampler import POVMPubResult


class POVMPostProcessor:
    """The canonical POVM result post-processor.

    This post-processor implementation provides a straight-forward interface for computing the
    expectation values (and standard deviations) of any Pauli-based observable. It is initialized
    with a :class:`.POVMPubResult` as shown below:

    >>> from povm_toolbox.library import ClassicalShadows
    >>> from povm_toolbox.sampler import POVMSampler
    >>> from povm_toolbox.post_processor import POVMPostProcessor
    >>> from qiskit.circuit import QuantumCircuit
    >>> from qiskit.primitives import StatevectorSampler
    >>> from qiskit.quantum_info import SparsePauliOp
    >>> circ = QuantumCircuit(2)
    >>> _ = circ.h(0)
    >>> _ = circ.cx(0, 1)
    >>> povm = ClassicalShadows(2, seed=42)
    >>> sampler = StatevectorSampler(seed=42)
    >>> povm_sampler = POVMSampler(sampler)
    >>> job = povm_sampler.run([circ], povm=povm, shots=16)
    >>> result = job.result()
    >>> post_processor = POVMPostProcessor(result[0])
    >>> post_processor.get_expectation_value(SparsePauliOp("ZI"))  # doctest: +FLOAT_CMP
    (-0.75, 0.33541019662496846)

    Additionally, this post-processor also supports the customization of the Dual frame in which the
    decomposition weights of the provided observable are obtained. Check out
    `this how-to guide <../how_tos/dual_optimizer.ipynb>`_ for more details on how to do this.
    """

    def __init__(
        self,
        povm_sample: POVMPubResult,
        dual: BaseDual | None = None,
    ) -> None:
        """Initialize the POVM post-processor.

        Args:
            povm_sample: a result from a POVM sampler run.
            dual: the Dual frame that will be used to obtain the decomposition weights of an
                observable when computing its expectation value. For more details, refer to
                :meth:`get_decomposition_weights`. When this is ``None``, the canonical Dual frame
                will be constructed from the POVM stored in the ``povm_sample``'s
                :attr:`.POVMPubResult.metadata`.

        Raises:
            ValueError: If the provided ``dual`` is not a dual frame to the POVM stored in the
                ``povm_samples``'s :attr:`.POVMPubResult.metadata`.
        """
        self._povm = povm_sample.metadata.povm_implementation.definition()

        self._counts = cast(np.ndarray, povm_sample.get_counts())

        if (dual is not None) and (not dual.is_dual_to(self._povm)):
            raise ValueError(
                "The ``dual`` argument is not valid. It is not a dual"
                " frame to the POVM stored in ``povm_sample``."
            )

        self._dual = dual

    @property
    def povm(self) -> BasePOVM:
        """Return the POVM definition that was used to sample outcomes."""
        return self._povm

    @property
    def counts(self) -> np.ndarray:
        """Return the histogram of the POVM outcomes via :meth:`.POVMPubResult.get_counts`."""
        return self._counts

    @property
    def dual(self) -> BaseDual:
        """Return the Dual frame that is used.

        .. warning::
            If the dual frame is not already built, accessing this property could be computationally
            demanding.
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
        self,
        observable: SparsePauliOp,
        outcome_set: set[Any],
    ) -> dict[Any, float]:
        r"""Get the decomposition weights of ``observable`` into the elements of :attr:`povm`.

        Given an observable :math:`O` which is in the span of a POVM (here, :attr:`povm`), one can
        write :math:`O` as the weighted sum of the POVM effects, :math:`O = \sum_k w_k M_k` for real
        weights :math:`w_k` and where :math:`k` labels the outcomes.

        See also :meth:`.BaseDual.get_omegas`.

        Args:
            observable: the observable to be decomposed into the POVM effects.
            outcome_set: set of outcome labels indicating which decomposition weights are queried.
                An outcome of a :class:`.ProductPOVM` is labeled by a tuple of integers for
                instance. For a :class:`.MultiQubitPOVM`, an outcome is simply labeled by an
                integer.

        Returns:
            A dictionary mapping outcome labels to decomposition weights.
        """
        return dict(self.dual.get_omegas(observable, outcome_set))  # type: ignore

    def get_expectation_value(
        self,
        observable: SparsePauliOp,
        *,
        loc: int | tuple[int, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """Return the expectation value and standard deviation of the given ``observable``.

        Args:
            observable: the observable whose expectation value is queried.
            loc: this argument is relevant if multiple sets of parameter values were supplied to the
                sampler in the same :class:`.POVMSamplerPub`. The index ``loc`` then corresponds to
                the set of parameter values that was supplied to the sampler through the Pub. If
                ``None``, the expectation value (and standard deviation) for each set of circuit
                parameters is returned.

        Returns:
            A tuple of (estimated) expectation value(s) and standard deviation(s). If a single value
            was queried (via ``loc``), both of these will be a ``float``. Otherwise, they will be
            instances of :class:`numpy.ndarray`.
        """
        if loc is not None:
            return self._single_exp_value_and_std(observable, loc=loc)
        if self.counts.shape == (1,):
            return self._single_exp_value_and_std(observable, loc=0)

        exp_val = np.zeros(shape=self.counts.shape, dtype=float)
        std = np.zeros(shape=self.counts.shape, dtype=float)
        for idx in np.ndindex(self.counts.shape):
            exp_val[idx], std[idx] = self._single_exp_value_and_std(observable, loc=idx)
        return exp_val, std

    def _single_exp_value_and_std(
        self,
        observable: SparsePauliOp,
        *,
        loc: int | tuple[int, ...],
    ) -> tuple[float, float]:
        """Return the expectation value and standard deviation of the given ``observable``.

        Args:
            observable: the observable whose expectation value is queried.
            loc: index of the results to use. The index corresponds to the set of parameter values
                that was supplied to the sampler through a :class:`.POVMSamplerPub`. If the circuit
                was not parametrized, the index ``loc`` should be 0.

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

        std = float(np.sqrt((std - exp_val**2) / (shots - 1)))

        return exp_val, std
