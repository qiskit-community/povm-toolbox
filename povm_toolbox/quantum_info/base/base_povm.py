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

from abc import ABC, abstractmethod

import numpy as np
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector

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

    @abstractmethod
    def get_prob(
        self,
        rho: SparsePauliOp | DensityMatrix | Statevector,
        outcome_idx: LabelT | set[LabelT] | None = None,
    ) -> float | dict[LabelT, float] | np.ndarray:
        r"""Return the outcome probabilities given a state, :math:`\rho`.

        Each outcome :math:`k` is associated with an effect :math:`M_k` of the POVM. The probability
        of obtaining the outcome :math:`k` when measuring a state ``rho`` is given by
        :math:`p_k = Tr[M_k \rho]`.

        .. note::
           TODO: explain how this relates to the :meth:`.BaseFrame.analysis` method.

        Args:
            rho: the state for which to compute the outcome probabilities.
            outcome_idx: label(s) indicating which outcome probabilities are queried.

        Returns:
            TODO explain the different output types and how these represent the outcome
            probabilities of the provided state.
        """
        # TODO: why is this method labeled abstract but still has an implementation? One of these
        # should be removed.
        return self.analysis(rho, outcome_idx)
