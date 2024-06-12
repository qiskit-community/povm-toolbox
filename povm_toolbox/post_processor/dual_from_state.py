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

from typing import cast

import numpy as np
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector

from povm_toolbox.quantum_info import BaseDUAL, BasePOVM, MultiQubitPOVM
from povm_toolbox.quantum_info.multi_qubit_dual import MultiQubitDUAL


def dual_from_state(
    povm: BasePOVM,
    state: SparsePauliOp | DensityMatrix | Statevector,
) -> BaseDUAL:
    """Return the dual frame of ``povm`` based on the outcome distribution of a supplied state.

    This methods constructs a joint dual frame where the alpha-parameters are
    set as the outcome probabilities of the supplied state. It can be shown
    that this is the dual frame minimizing the variance of the estimator (irrespective
    of the observable to estimate) if outcomes are sampled from this state.

    Args:
        povm: the POVM for which we want to build a dual frame.
        state: state from which to compute the outcome probabilities.

    Raises:
        NotImplementedError: if ``povm`` is not a :class:`povm_toolbox.quantum_info.product_povm.MultiQubitPOVM`
            instance.
    """
    if not isinstance(povm, MultiQubitPOVM):
        # TODO : implement for ProductPOVM
        raise NotImplementedError(
            "This method is only implemented for ``povm_toolbox.quantum_info.product_povm.MultiQubitPOVM``."
        )
    alphas = tuple(cast(np.ndarray, povm.get_prob(state)))
    return MultiQubitDUAL.build_dual_from_frame(povm, alphas=alphas)
