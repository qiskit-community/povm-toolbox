# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""BaseDual."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

from .base_frame import BaseFrame, LabelT


class BaseDual(BaseFrame[LabelT], ABC):
    """Abstract base class that contains all methods that any specific Dual should implement."""

    @property
    def num_outcomes(self) -> int:
        """The number of outcomes of the Dual."""
        return self.num_operators

    def get_omegas(
        self,
        observable: SparsePauliOp | Operator,
        outcome_idx: LabelT | set[LabelT] | None = None,
    ) -> float | dict[LabelT, float] | np.ndarray:
        r"""Return the decomposition weights of the provided observable.

        Computes the :math:`\omega_k` in

        .. math::
           \mathcal{O} = \sum_{k=1}^n \omega_k M_k

        where :math:`\mathcal{O}` is the ``observable`` and :math:`M_k` are the effects of the POVM
        of which ``self`` is the dual. The closed form for computing :math:`\omega_k` is

        .. math::
           \omega_k = \text{Tr}\left[\mathcal{O} D_k\right]

        where :math:`D_k` make of this dual frame (i.e. ``self``).

        .. note::
            In the frame theory formalism, the mapping
            :math:`A: \mathcal{O} \mapsto \{\text{Tr}\left[\mathcal{O} D_k\right]\}_k`
            is referred to as the *analysis operator*, which is implemented by
            the :meth:`.analysis` method.

        Args:
            observable: the observable for which to compute the decomposition weights.
            outcome_idx: label or set of labels indicating which decomposition weights
                are queried. If ``None``, all weights are queried.

        Returns:
            Decomposition weight(s) associated to the effect(s) specified by ``outcome_idx``. If a
            specific outcome was queried, a ``float`` is returned. If a specific set of outcomes was
            queried, a dictionary mapping outcome labels to weights is returned. If all outcomes
            were queried, an array with all weights is returned.
        """
        return self.analysis(observable, outcome_idx)

    @abstractmethod
    def is_dual_to(self, frame: BaseFrame) -> bool:
        """Check if ``self`` is a dual to another frame.

        Args:
            frame: the other frame to check duality against.

        Returns:
            Whether ``self`` is dual to ``frame``.
        """

    @classmethod
    @abstractmethod
    def build_dual_from_frame(cls, frame: BaseFrame, alphas: tuple[Any] | None = None) -> BaseDual:
        """Construct a dual frame to another (primal) frame.

        Args:
            frame: The primal frame from which we will build the dual frame.
            alphas: parameters of the frame super-operator used to build the dual frame.
                If ``None``, the parameters are set as the traces of each operator in the primal
                frame.

        Returns:
            A dual frame to the supplied ``frame``.
        """
