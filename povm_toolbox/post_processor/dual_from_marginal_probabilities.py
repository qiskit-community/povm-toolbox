# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""dual_from_marginal_probabilities."""

from __future__ import annotations

from typing import cast

import numpy as np
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector

from povm_toolbox.quantum_info import ProductDual, ProductPOVM
from povm_toolbox.quantum_info.base import BaseDual, BasePOVM


def dual_from_marginal_probabilities(
    povm: BasePOVM,
    state: SparsePauliOp | DensityMatrix | Statevector,
    *,
    threshold: float = 1e-5,
) -> BaseDual:
    """Return the Dual frame of ``povm`` based on the marginal distribution of a supplied ``state``.

    This method constructs a product Dual frame where each local Dual frame is parametrized with the
    alpha-parameters set as the marginal outcome probabilities of the supplied ``state``.

    You can use this function like any of the Dual frame constructors to set the
    :attr:`.POVMPostProcessor.dual` attribute as shown in
    `this how-to guide <../how_tos/dual_optimizer.ipynb>`_.

    .. warning::
       Computing this Dual frame obviously requires knowledge of an exact reference state and, thus,
       is limited in its applicability to development and testing cases in which an exact state is
       available.

    Args:
        povm: the POVM for which we want to build a Dual frame.
        state: the state from which to compute the marginal outcome probabilities.
        threshold: if an outcome probability is below this value, an offset equal to the threshold
            magnitude will be added to all probabilities in the same marginal distribution. This is
            designed to avoid having parameters of the frame super-operator set to zero, which would
            result in a :exc:`.ZeroDivisionError`.

    Raises:
        NotImplementedError: if ``povm`` is not a :class:`.ProductPOVM` instance. If you have
            :class:`.MultiQubitPOVM`, have a look at :func:`.dual_from_state`.

    Returns:
        The Dual frame based on the marginal outcome probabilities of the given ``state``.
    """
    if not isinstance(povm, ProductPOVM):
        raise NotImplementedError("This method is only implemented for `ProductPOVM` objects.")

    axes = np.arange(len(povm.sub_systems), dtype=int)
    joint_prob = cast(np.ndarray, povm.get_prob(state))
    alphas = []
    for qubit_idx in povm.sub_systems:
        marg_prob = joint_prob.sum(axis=tuple(np.delete(axes, [qubit_idx])))
        if np.any(np.absolute(marg_prob) < threshold):
            marg_prob += threshold
        alphas.append(tuple(marg_prob))

    return ProductDual.build_dual_from_frame(povm, alphas=tuple(alphas))
