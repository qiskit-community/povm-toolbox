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
    from typing import override  # pragma: no cover


import numpy as np
from numpy.random import Generator, default_rng
from qiskit.quantum_info import SparsePauliOp

from povm_toolbox.post_processor.povm_post_processor import POVMPostProcessor
from povm_toolbox.quantum_info.base import BaseDual
from povm_toolbox.sampler import POVMPubResult


class MedianOfMeans(POVMPostProcessor):
    r"""A POVM result post-processor which uses a 'median of means' estimator.

    Given ``num_shots=num_batches*batch_size`` samples, we partition the
    samples into ``num_batches`` batches. We compute the mean of each batch,
    :math:`\hat{o}_j`, and then output the median of the means, :math:`\hat{o}
    =\mathrm{median}\{\hat{o}_1, ..., \hat{o}_{\mathrm{num\_batches}}\}`.
    It can be shown that

    .. math::
        \lvert \mathrm{Tr}[\mathcal{O} \rho] - \hat{o} \rvert \leq \epsilon
        \quad \textrm{with probability at least } 1-\delta \, ,

    where :math:`\delta = 2 \exp{(-\mathrm{num\_batches}/2)}` and :math:`\epsilon
    = \sqrt{\frac{34}{\mathrm{batch\_size}} } \lVert \mathcal{O} - \frac{\mathrm{Tr}
    [\mathcal{O}]}{2^N} \mathbb{I} \rVert_\textrm{shadow}`. For more details, see
    the work of H.-Y. Huang, R. Kueng, and J. Preskill, "*Predicting Many Properties
    of a Quantum System from Very Few Measurements*", Nature Physics 16, 1050 (2020).

    The interface of this post-processor is essentially identical to the one of its
    baseclass (see :class:`.POVMPostProcessor` for more details). For completeness,
    here is an example how to use it:

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
    (-0.75, 2.9154759474226504)
    """

    def __init__(
        self,
        povm_sample: POVMPubResult,
        dual: BaseDual | None = None,
        *,
        num_batches: int | None = None,
        upper_delta_confidence: float | None = None,
        seed: int | Generator | None = None,
    ) -> None:
        r"""Initialize the median-of-means post-processor.

        Args:
            povm_sample: a result from a POVM sampler run.
            dual: the Dual frame that will be used to obtain the decomposition weights of an
                observable when computing its expectation value. For more details, refer to
                :meth:`get_decomposition_weights`. When this is ``None``, the canonical Dual frame
                will be constructed from the POVM stored in the ``povm_sample``'s
                :attr:`.POVMPubResult.metadata`.
            num_batches: number of batches, i.e. number of samples means, used in the median-of-means
                estimator. This value will be overridden if a ``delta_confidence`` argument is supplied.
            upper_delta_confidence: an upper bound for the confidence parameter :math:`\delta` used to
                determine the necessary number of batches as :math:`\mathrm{num\_batches} = \lceil 2
                \log{(2/\delta)} \rceil`. It will override any ``num_batches`` supplied argument. If
                both ``num_batches`` and ``delta_confidence`` are ``None``, ``delta_confidence`` is
                set to 0.05. Note that this argument is actually an upper bound for the true
                :math:`\delta`-parameter which is given by :math:`\delta=2 \exp(-\mathrm{num\_batches}/2)`.
            seed: optional seed to fix the :class:`numpy.random.Generator` used to generate the
                batches. The user can also directly provide a random generator. If ``None``, a
                random seed will be used.

        Raises:
            ValueError: If the provided ``dual`` is not a dual frame to the POVM stored in the
                ``povm_samples``'s :attr:`.POVMPubResult.metadata`.
            TypeError: If the type of ``seed`` is not valid.
        """
        super().__init__(povm_sample=povm_sample, dual=dual)

        self.num_batches: int
        """Number of batches, i.e. number of samples means, used in the median-of-means estimator."""

        if isinstance(upper_delta_confidence, float) or num_batches is None:
            self.delta_confidence = upper_delta_confidence or 0.05
        else:
            self.num_batches = num_batches

        if seed is None:
            seed = default_rng()
        elif isinstance(seed, int):
            seed = default_rng(seed)
        elif not isinstance(seed, Generator):
            raise TypeError(f"The type of `seed` ({type(seed)}) is not valid.")

        self.seed = seed

    @property
    def delta_confidence(self) -> float:
        r"""The confidence parameter :math:`\delta=2 \exp(-` ``num_batches`` :math:`/2)`."""
        return float(2 * np.exp(-0.5 * self.num_batches))

    @delta_confidence.setter
    def delta_confidence(self, delta: float) -> None:
        r"""Set the upper bound for the confidence parameter :math:`\delta`.

        It is used to determine the necessary number of batches as :math:`\mathrm{
        num\_batches} = \lceil 2 \log{(2/\delta)} \rceil`. Then, the actual value
        of the :math:`\delta`-parameter is given by :math:`\delta=2 \exp(-\mathrm{
        num\_batches}/2)`.
        """
        self.num_batches = int(np.ceil(2 * np.log(2 / delta)))

    @override
    def _single_exp_value_and_std(
        self, observable: SparsePauliOp, *, loc: int | tuple[int, ...]
    ) -> tuple[float, float]:
        count = self.counts[loc]
        shots = sum(count.values())
        omegas = self.get_decomposition_weights(observable, set(count.keys()))

        # get the samples and randomize their order
        sampled_weights = np.zeros(shots)
        idx = 0
        for outcome in count:
            sampled_weights[idx : idx + count[outcome]] = omegas[outcome]
            idx += count[outcome]
        sampled_weights = self.seed.permutation(sampled_weights)

        # compute the batch size
        batch_size = shots // self.num_batches
        batches = sampled_weights[: self.num_batches * batch_size].reshape(
            (batch_size, self.num_batches)
        )

        # if the numbers of shots is not a multiple of number of batches, some batches
        # will have ``batch_size`` samples and some will have ``batch_size+1`` samples
        # such that all samples are used.
        if shots % self.num_batches != 0:
            last_samples = np.full((1, self.num_batches), np.nan)
            last_samples[:, : shots % self.num_batches] = sampled_weights[
                self.num_batches * batch_size :
            ]
            batches = np.concatenate((batches, last_samples))

        # compute the median of means estimate
        median_of_means = float(np.median(np.nanmean(batches, axis=0)))

        # compute the coefficient in front of the shadow norm in the formula to compute
        # the confidence parameter ``delta``.
        coefficient_epsilon = float(np.sqrt(34 / batch_size))

        return median_of_means, coefficient_epsilon
