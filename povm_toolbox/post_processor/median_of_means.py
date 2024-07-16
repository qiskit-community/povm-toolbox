# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""MedianOfMeans."""

from __future__ import annotations

import sys

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


import numpy as np
from numpy.random import Generator, default_rng
from qiskit.quantum_info import SparsePauliOp

from povm_toolbox.post_processor.povm_post_processor import POVMPostProcessor
from povm_toolbox.quantum_info.base import BaseDual
from povm_toolbox.sampler import POVMPubResult


class MedianOfMeans(POVMPostProcessor):
    """A POVM result post-processor which uses a 'median of means' estimator.

    This post-processor implementation provides a straight-forward interface for computing the
    expectation values (and standard deviations) of any Pauli-based observable. It is initialized
    with a :class:`.POVMPubResult` as shown below:

    >>> from povm_toolbox.library import ClassicalShadows
    >>> from povm_toolbox.sampler import POVMSampler
    >>> from povm_toolbox.post_processor import MedianOfMeans
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
    >>> post_processor = MedianOfMeans(result[0], num_batches=4, seed=42)
    >>> post_processor.get_expectation_value(SparsePauliOp("ZI"))  # doctest: +FLOAT_CMP
    (-0.75, 0.0)

    Additionally, this post-processor also supports the customization of the Dual frame in which the
    decomposition weights of the provided observable are obtained. Check out
    `this how-to guide <../how_tos/dual_optimizer.ipynb>`_ for more details on how to do this.
    """

    def __init__(
        self,
        povm_sample: POVMPubResult,
        dual: BaseDual | None = None,
        num_batches: int = 1,
        seed: int | Generator | None = None,
    ) -> None:
        """Initialize the median-of-means post-processor.

        Args:
            povm_sample: a result from a POVM sampler run.
            dual: the Dual frame that will be used to obtain the decomposition weights of an
                observable when computing its expectation value. For more details, refer to
                :meth:`get_decomposition_weights`. When this is ``None``, the canonical Dual frame
                will be constructed from the POVM stored in the ``povm_sample``'s
                :attr:`.POVMPubResult.metadata`.
            num_batches: TODO.
            seed: TODO.

        Raises:
            ValueError: If the provided ``dual`` is not a dual frame to the POVM stored in the
                ``povm_samples``'s :attr:`.POVMPubResult.metadata`.
            TypeError: TODO.
        """
        super().__init__(povm_sample=povm_sample, dual=dual)

        self.num_batches = num_batches

        if seed is None:
            seed = default_rng()
        elif isinstance(seed, int):
            seed = default_rng(seed)
        elif not isinstance(seed, Generator):
            raise TypeError(f"The type of `seed` ({type(seed)}) is not valid.")

        self.seed = seed

    @override
    def _single_exp_value_and_std(
        self, observable: SparsePauliOp, loc: int | tuple[int, ...]
    ) -> tuple[float, float]:
        count = self.counts[loc]
        shots = sum(count.values())
        omegas = self.get_decomposition_weights(observable, set(count.keys()))

        batch_size = shots // self.num_batches

        sampled_weights = np.zeros(shots)
        idx = 0
        for outcome in count:
            sampled_weights[idx : idx + count[outcome]] = omegas[outcome]
            idx += count[outcome]

        sampled_weights = self.seed.permutation(sampled_weights)

        batches = sampled_weights[: self.num_batches * batch_size].reshape(
            (batch_size, self.num_batches)
        )
        if shots % self.num_batches != 0:
            last_samples = np.full((1, self.num_batches), np.nan)
            last_samples[:, : shots % self.num_batches] = sampled_weights[
                self.num_batches * batch_size :
            ]
            batches = np.concatenate((batches, last_samples))
        median_of_means = float(np.median(np.nanmean(batches, axis=0)))

        epsilon = 0.0  # TODO

        return median_of_means, epsilon
