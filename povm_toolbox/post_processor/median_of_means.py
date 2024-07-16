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
    """A POVM result post-processor which uses a 'median of means' estimator."""

    def __init__(
        self,
        povm_sample: POVMPubResult,
        dual: BaseDual | None = None,
        num_batches: int = 1,
        seed: int | Generator | None = None,
    ) -> None:
        """Initialize the POVM post-processor.

        Args:
            povm_sample: a result from a POVM sampler run.
            dual: the subclass of :class:`.BaseDual` that will be used to
                build the dual frame to the POVM of ``povm_sample``. The dual
                frame is then used to compute the decomposition weights of any
                observable.
            num_batches: TODO.
            seed: TODO.

        Raises:
            ValueError: If the provided ``dual`` is not a dual frame to the POVM
                used to produce ``povm_sample``.
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
        median_of_means: float = np.median(np.nanmean(batches, axis=0))

        epsilon = 0.0  # TODO

        return (median_of_means, epsilon)
