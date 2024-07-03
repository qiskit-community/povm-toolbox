# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""BaseFrame."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

LabelT = TypeVar("LabelT")
"""Each operator in the frame is identified by a label.

This is the type of these labels. For instance, labels could be strings, integers, or it could be
tuples of integers among other possibilities.
"""


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
        """If the frame spans the entire Hilbert space."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimension of the Hilbert space on which the frame operators act."""

    @property
    def num_subsystems(self) -> int:
        r"""The number of subsystems which the frame operators act on.

        For qubits, this is always :math:`\log_2(`:attr:`.dimension`:math:`)`.
        """
        return int(math.log2(self.dimension))

    @property
    @abstractmethod
    def num_operators(self) -> int:
        """The number of frame operators of the frame."""

    @abstractmethod
    def _check_validity(self) -> None:
        """Check if frame axioms are fulfilled."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of frame operators of the POVM."""

    @abstractmethod
    def analysis(
        self,
        hermitian_op: SparsePauliOp | Operator,
        frame_op_idx: LabelT | set[LabelT] | None = None,
    ) -> float | dict[LabelT, float] | np.ndarray:
        r"""Return the frame coefficients of ``hermitian_op``.

        This method implements the *analysis operator* :math:`A` of the frame :math:`\{F_k\}_k`:

        .. math::
            A: \mathcal{O} \mapsto \{ \mathrm{Tr}\left[F_k \mathcal{O} \right] \}_k,

        where :math:`c_k =  \mathrm{Tr}\left[F_k \mathcal{O} \right]` are called the *frame
        coefficients* of the Hermitian operator :math:`\mathcal{O}`.

        Args:
            hermitian_op: a hermitian operator whose frame coefficients to compute.
            frame_op_idx: label or set of labels indicating which coefficients are
                queried. If ``None``, all coefficients are queried.

        Returns:
            Frame coefficients, specified by ``frame_op_idx``, of the Hermitian operator
            ``hermitian_op``. If a specific coefficient was queried, a ``float`` is returned. If a
            specific set of coefficients was queried, a dictionary mapping labels to coefficients is
            returned. If all coefficients were queried, an array with all coefficients is returned.

        Raises:
            TypeError: when the provided single or sequence of labels ``frame_op_idx`` does not have
                a valid type.
            ValueError: when the dimension of the provided ``hermitian_op`` does not match the
                dimension of the frame operators.
        """
