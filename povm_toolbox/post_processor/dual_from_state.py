# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""dual_from_state."""

from __future__ import annotations

from typing import cast

import numpy as np
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector

from povm_toolbox.quantum_info import MultiQubitDual, MultiQubitPOVM
from povm_toolbox.quantum_info.base import BaseDual, BasePOVM


def dual_from_state(
    povm: BasePOVM,
    state: SparsePauliOp | DensityMatrix | Statevector,
) -> BaseDual:
    """Return the Dual frame of ``povm`` based on the outcome distribution of a supplied ``state``.

    This method constructs a joint Dual frame where the alpha-parameters are set as the outcome
    probabilities of the supplied ``state``. It can be shown that this is the Dual frame minimizing
    the variance of the estimator (irrespective of the observable to estimate) if outcomes are
    sampled from this state.

    You can use this function like any of the Dual frame constructors to set the
    :attr:`.POVMPostProcessor.dual` attribute as shown in
    `this how-to guide <../how_tos/dual_optimizer.ipynb>`_.

    .. warning::
       Computing this Dual frame obviously requires knowledge of an exact reference state and, thus,
       is limited in its applicability to development and testing cases in which an exact state is
       available.

    Args:
        povm: the POVM for which we want to build a Dual frame.
        state: the state from which to compute the outcome probabilities.

    Raises:
        NotImplementedError: if ``povm`` is not a :class:`.MultiQubitPOVM` instance. If you have a
            :class:`.ProductPOVM`, have a look at :func:`.dual_from_marginal_probabilities`.

    Returns:
        The Dual frame with minimal variance of the estimator for any arbitrary observable.
    """
    if not isinstance(povm, MultiQubitPOVM):
        # TODO : implement for ProductPOVM
        raise NotImplementedError("This method is only implemented for `MultiQubitPOVM` objects.")

    alphas = tuple(cast(np.ndarray, povm.get_prob(state)))
    return MultiQubitDual.build_dual_from_frame(povm, alphas=alphas)
