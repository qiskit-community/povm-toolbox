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

import math
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

# Each operator in the frame is identified by a label. ``LabelT`` is the type of
# these labels. For instance, labels could be strings, integers, or it could be
# tuples of integers among other possibilities.
LabelT = TypeVar("LabelT")


class BaseFrame(ABC, Generic[LabelT]):
    """Abstract base class that contains all methods that any specific frame should implement.

    A frame is a generalization of the notion of the basis of a vector space to sets that may
    not necessarily be linearly independent. Consider a Hilbert space of finite dimension :math:`d`,
    then the set of Hermitian operators is an operator-valued vector space on the Hilbert space.
    Therefore, a set of Hermitian operators that spans the entire Hilbert space is said to be a
    frame.

    If a set of operators does not span the entire Hilbert space, it can still be considered as a
    frame on the subspace it spans.
    """

    @property
    @abstractmethod
    def informationally_complete(self) -> bool:
        """Return if the frame spans the entire Hilbert space."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Give the dimension of the Hilbert space on which the effects act."""

    @property
    def num_subsystems(self) -> int:
        r"""Return the number of subsystems which the effects act on.

        For qubits, this is always :math:`\log_2(`:attr:`.dimension`:math:`)`.
        """
        return int(math.log2(self.dimension))

    @property
    @abstractmethod
    def num_operators(self) -> int:
        """Give the number of effects of the frame."""

    @abstractmethod
    def _check_validity(self) -> None:
        """Check if frame axioms are fulfilled."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of effects of the POVM."""

    @abstractmethod
    def analysis(
        self,
        hermitian_op: SparsePauliOp | Operator,
        frame_op_idx: LabelT | set[LabelT] | None = None,
    ) -> float | dict[LabelT, float] | np.ndarray:
        """Return the frame coefficients of ``hermitian_op``."""
