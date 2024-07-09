# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""dual_from_empirical_frequencies."""

from __future__ import annotations

from typing import cast

import numpy as np
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector

from povm_toolbox.post_processor.povm_post_processor import POVMPostProcessor
from povm_toolbox.quantum_info import ProductDual, ProductPOVM
from povm_toolbox.quantum_info.base import BaseDual


def dual_from_empirical_frequencies(
    povm_post_processor: POVMPostProcessor,
    *,
    loc: int | tuple[int, ...] | None = None,
    bias: list[float] | float | None = None,
    ansatz: list[SparsePauliOp | DensityMatrix | Statevector]
    | SparsePauliOp
    | DensityMatrix
    | Statevector
    | None = None,
) -> BaseDual:
    """Return the Dual frame of ``povm`` based on the frequencies of the sampled outcomes.

    Given outcomes sampled from a :class:`.ProductPOVM`, each local Dual frame is parametrized with
    the alpha-parameters set as the marginal outcome frequencies. For stability, the (local)
    empirical frequencies can be biased towards the (marginal) outcome probabilities of an
    ``ansatz`` state.

    Args:
        povm_post_processor: the :class:`.POVMPostProcessor` object from which to extract the
            :attr:`.POVMPostProcessor.povm` and the empirical frequencies to build the Dual frame.
        loc: index of the results to use. This is relevant if multiple sets of parameter values were
            supplied to the sampler in the same Pub. If ``None``, it is assumed that the supplied
            circuit was not parametrized or that a unique set of parameter values was supplied. In
            this case, ``loc`` is trivially set to 0.
        bias: the strength of the bias towards the outcome distribution of the ``ansatz`` state. If
            it is a ``float``, the same bias is applied to each (local) sub-system. If it is a list
            of ``float``, a specific bias is applied to each sub-system. If ``None``, the bias for
            each sub-system is set to be the number of outcomes of the POVM acting on this
            sub-system.
        ansatz: list of quantum states for each local sub-system. If a single (local) quantum state
            is supplied, it is used for all sub-systems. From these states, the local outcome
            probability distributions are computed for each sub-system. The empirical marginal
            frequencies are biased towards these distributions. If None, the fully mixed state is
            used for each sub-system.

    Raises:
        NotImplementedError: if :attr:`.POVMPostProcessor.povm` is not a :class:`.ProductPOVM`
            instance.
        ValueError: if ``loc`` is ``None`` and :attr:`.POVMPostProcessor.counts` stores more than a
            single counter (i.e., multiple sets of parameter values were supplied to the sampler in
            a single Pub).
        ValueError: if ``bias`` is a list but its length does not match the number of local POVMs
            forming the product POVM.
        ValueError: if ``ansatz`` is a list but its length does not match the number of local POVMs
            forming the product POVM.

    Returns:
        The Dual frame based on the empirical outcome frequencies from the post-processed result.
    """
    povm = povm_post_processor.povm
    if not isinstance(povm, ProductPOVM):
        raise NotImplementedError("This method is only implemented for `ProductPOVM` objects.")

    if loc is None:
        if povm_post_processor.counts.shape == (1,):
            loc = (0,)
        else:
            raise ValueError(
                "`loc` has to be specified if the POVM post-processor stores"
                " more than one counter (i.e., if multiple sets of parameter"
                " values were supplied to the sampler in a single pub). The"
                f" array of counters is of shape {povm_post_processor.counts.shape}."
            )
    counts = povm_post_processor.counts[loc]

    if isinstance(bias, list) and len(bias) != len(povm.sub_systems):
        raise ValueError(
            f"A list of biases was submitted but its length ({len(bias)})"
            f" does not match the number of local POVMs ({len(povm.sub_systems)})."
        )

    if isinstance(ansatz, list) and len(ansatz) != len(povm.sub_systems):
        raise ValueError(
            "A list of ansatz local states was submitted but its length"
            f" ({len(ansatz)}) does not match the number of local POVMs"
            f" ({len(povm.sub_systems)})."
        )

    marginals = [np.zeros(subsystem_shape) for subsystem_shape in povm.shape]

    # Computing marginals
    shots = sum(counts.values())
    for outcome, count in counts.items():
        for i, k_i in enumerate(outcome):
            marginals[i][k_i] += count / shots

    alphas = []
    # Computing alphas for each subsystem
    for i, sub_system in enumerate(povm.sub_systems):
        sub_povm = povm[sub_system]
        dim = sub_povm.dimension
        ansatz_state = (
            DensityMatrix(np.eye(dim) / dim)
            if ansatz is None
            else (ansatz[i] if isinstance(ansatz, list) else ansatz)
        )
        sub_bias = (
            sub_povm.num_outcomes if bias is None else (bias[i] if isinstance(bias, list) else bias)
        )
        sub_alphas = shots * marginals[i] + sub_bias * cast(
            np.ndarray, sub_povm.get_prob(ansatz_state)
        )
        alphas.append(tuple(sub_alphas / (shots + sub_bias)))

    # Building ProductDual from frequencies
    return ProductDual.build_dual_from_frame(povm, alphas=tuple(alphas))
