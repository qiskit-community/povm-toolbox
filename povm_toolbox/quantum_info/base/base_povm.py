# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""BasePOVM."""

from __future__ import annotations

from abc import ABC

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp, Statevector

from .base_dual import BaseDual
from .base_frame import BaseFrame, LabelT


class BasePOVM(BaseFrame[LabelT], ABC):
    """Abstract base class that contains all methods that any specific POVM should implement."""

    default_dual_class: type[BaseDual]
    """The default :class:`.BaseDual` associated with this POVM."""

    @property
    def num_outcomes(self) -> int:
        """The number of outcomes of the POVM."""
        return self.num_operators

    def get_prob(
        self,
        rho: SparsePauliOp | DensityMatrix | Statevector,
        outcome_idx: LabelT | set[LabelT] | None = None,
    ) -> float | dict[LabelT, float] | np.ndarray:
        r"""Return the outcome probabilities given a state, :math:`\rho`.

        Each outcome :math:`k` is associated with an effect :math:`M_k` of the POVM. The probability
        of obtaining the outcome :math:`k` when measuring a state ``rho`` is given by
        :math:`p_k = \text{Tr}\left[M_k \rho\right]`.

        .. note::
            In the frame theory formalism, the mapping
            :math:`A: \rho \mapsto \{\text{Tr}\left[M_k \rho\right]\}_k`
            is referred to as the *analysis operator*, which is implemented by
            the :meth:`.analysis` method.

        Args:
            rho: the state for which to compute the outcome probabilities.
            outcome_idx: label or set of labels indicating which outcome probabilities
                are queried. If ``None``, all outcome probabilities are queried.

        Returns:
            Probabilities of obtaining the outcome(s) specified by ``outcome_idx`` over the state
            ``rho``. If a specific outcome was queried, a ``float`` is returned. If a specific set
            of outcomes was queried, a dictionary mapping outcomes to probabilities is returned. If
            all outcomes were queried, an array with all probabilities is returned.
        """
        if isinstance(rho, (DensityMatrix, Statevector)):
            rho = Operator(rho)
        return self.analysis(rho, outcome_idx)
