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

from abc import ABC, abstractmethod

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

from .base_frame import BaseFrame, LabelT


class BaseDUAL(BaseFrame[LabelT], ABC):
    """Abstract base class that contains all methods that any specific DUAL subclass should implement."""

    @property
    def n_outcomes(self) -> int:
        """Give the number of outcomes of the DUAL."""
        return self.n_operators

    @abstractmethod
    def get_omegas(
        self,
        obs: SparsePauliOp | Operator,
        outcome_idx: LabelT | set[LabelT] | None = None,
    ) -> float | dict[LabelT, float] | np.ndarray:
        """Return the decomposition weights of the provided observable into the POVM effects to which ``self`` is a dual."""
        return self.analysis(obs, outcome_idx)

    @abstractmethod
    def is_dual_to(self, frame: BaseFrame) -> bool:
        """Check if `self` is a dual to another frame."""

    @abstractmethod
    def optimize(self, frame: BaseFrame, **options) -> None:
        """Optimize the dual to `frame` inplace.

        Args:
            frame: The primal frame to which ``self`` is a dual.
            options: keyword arguments specifying how to optimize ``self``.
        """

    @classmethod
    @abstractmethod
    def build_dual_from_frame(cls, frame: BaseFrame) -> BaseDUAL:
        """Construct a dual frame to another frame.

        Args:
            frame: The primal frame from which we will build the dual frame.

        Returns:
            A dual frame to the supplied ``frame``.
        """