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
from typing import TypeVar

import numpy as np
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector

from .base_dual import BaseDUAL
from .base_frame import BaseFrame

# type of the labels used to specify povm effects
LabelT = TypeVar("LabelT")


class BasePOVM(BaseFrame[LabelT], ABC):
    """Abstract base class that contains all methods that any specific POVM subclass should implement."""

    default_dual_class: type[BaseDUAL]

    @property
    def n_outcomes(self) -> int:
        """Give the number of outcomes of the POVM."""
        return self.n_operators

    @abstractmethod
    def get_prob(
        self,
        rho: SparsePauliOp | DensityMatrix | Statevector,
        outcome_idx: LabelT | set[LabelT] | None = None,
    ) -> float | dict[LabelT, float] | np.ndarray:
        """Return the outcome probabilities given a state rho."""
        return self.analysis(rho, outcome_idx)
