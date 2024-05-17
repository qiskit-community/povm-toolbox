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

import sys

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator, SparsePauliOp

from povm_toolbox.utilities import matrix_to_double_ket

from .base_frame import BaseFrame


class MultiQubitFrame(BaseFrame[int]):
    """Class that collects all information that any MultiQubit frame should specify.

    This is a representation of an operator-valued vector space frame. The effects are
    specified as a list of :class:`~qiskit.quantum_info.Operator`.
    """

    def __init__(self, list_operators: list[Operator]) -> None:
        r"""Initialize from explicit operators.

        Args:
            list_operators: list that contains the explicit frame operators. The
                length of the list is the number of operators of the frame.


        Raises:
            ValueError: if the frame operators do not have a correct shape. They should all
                be square and of the same dimension.
        """
        self._n_operators: int = len(list_operators)
        self._dimension: int = list_operators[0].dim[0]
        for frame_op in list_operators:
            if not (self._dimension == frame_op.dim[0] and self._dimension == frame_op.dim[1]):
                raise ValueError(
                    f"Frame operators need to be square ({frame_op.dim[0]},{frame_op.dim[1]}) and all of the same dimension."
                )
        self._operators: list[Operator] = list_operators

        self._pauli_operators: list[dict[str, complex]] | None = None

        self._array: np.ndarray = np.ndarray((self.dimension**2, self.n_operators), dtype=complex)
        for k, frame_op in enumerate(list_operators):
            self._array[:, k] = matrix_to_double_ket(frame_op.data)

        self._informationally_complete: bool = bool(
            np.linalg.matrix_rank(self._array) == self.dimension**2
        )

        self._check_validity()

    def __repr__(self) -> str:
        f_subsystems = f'(num_qubits={self.n_subsystems})' if self.n_subsystems>1 else ''
        return f"{self.__class__.__name__}{f_subsystems}<{self.n_operators}> at {hex(id(self))}"

    @property
    def informationally_complete(self) -> bool:
        """Return if the frame spans the entire Hilbert space."""
        return self._informationally_complete

    @property
    def dimension(self) -> int:
        """Give the dimension of the Hilbert space on which the frame operators act."""
        return self._dimension

    @property
    def n_operators(self) -> int:
        """Give the number of outcomes of the frame."""
        return self._n_operators

    @property
    def operators(self) -> list[Operator]:
        """Return the list of frame operators."""
        return self._operators

    @operators.setter
    def operators(self, new_operators: list[Operator]):
        """Set the frame operators."""
        self._n_operators = len(new_operators)
        self._dimension = new_operators[0].dim[0]
        for frame_op in new_operators:
            if not (self._dimension == frame_op.dim[0] and self._dimension == frame_op.dim[1]):
                raise ValueError(
                    f"Frame operators need to be square ({frame_op.dim[0]},{frame_op.dim[1]}) and all of the same dimension."
                )

        self._operators = new_operators

        self._pauli_operators = None

        self._array = np.ndarray((self.dimension**2, self.n_operators), dtype=complex)
        for k, frame_op in enumerate(new_operators):
            self._array[:, k] = matrix_to_double_ket(frame_op.data)

        self._informationally_complete = bool(
            np.linalg.matrix_rank(self._array) == self.dimension**2
        )

        self._check_validity()

    @property
    def pauli_operators(self) -> list[dict[str, complex]]:
        """Convert the internal frame operators to Pauli form.

        Raises:
            ValueError: when the frame operators are not N-qubit operators.
        """
        if self._pauli_operators is None:
            try:
                self._pauli_operators = [
                    dict(SparsePauliOp.from_operator(op).label_iter()) for op in self.operators
                ]
            except QiskitError as exc:
                raise ValueError("Failed to convert frame operators to Pauli form.") from exc
        return self._pauli_operators

    def _check_validity(self) -> None:
        r"""Check if frame axioms are fulfilled.

        Raises:
            ValueError: if any of the frame operators is not hermitian.
        """
        for k, op in enumerate(self.operators):
            if not np.allclose(op, op.adjoint(), atol=1e-5):
                raise ValueError(f"Frame operator {k} is not hermitian.")

    def __getitem__(self, index: slice) -> Operator | list[Operator]:
        r"""Return a frame operator or a list of frame operators.

        Args:
            index: indicate the operator(s) to be returned.

        Returns:
            The operator or list of operators corresponding to the index.
        """
        return self.operators[index]

    def __len__(self) -> int:
        """Return the number of operators of the frame."""
        return self.n_operators

    def __array__(self) -> np.ndarray:
        """Return the array representation of the frame, with shape.

        The array has a shape :math:`(``self.dimension``**2, ``self.n_operators``)`.
        """
        return self._array

    def analysis(
        self, hermitian_op: SparsePauliOp | Operator, frame_op_idx: int | set[int] | None = None
    ) -> float | dict[int, float] | np.ndarray:
        r"""Return the frame coefficients given an operator ``hermitian_op``.

        Given a Hermitian operator :math:`\mathcal{O}`, one can compute its frame
        coefficients :math:`c_k = Tr[M_k \mathcal{O}]`, where :math:`M_k` is the
        frame operator labelled by :math:`k`. In frame theory, this operation is
        called the 'analysis operation'.

        Args:
            hermitian_op: the Hermitian operator whose frame coefficient(s) are queried.
            frame_op_idx: label(s) indicating which frame coefficients are queried.

        Returns:
            An array of coefficients. TODO: update.

        Raises:
            TypeError: if the label(s) ``frame_op_idx`` do not have a valid type.
        """
        if isinstance(hermitian_op, SparsePauliOp):
            hermitian_op = hermitian_op.to_operator()
        op_vectorized = np.conj(matrix_to_double_ket(hermitian_op.data))

        if isinstance(frame_op_idx, int):
            return float(np.dot(op_vectorized, self._array[:, frame_op_idx]).real)
        if isinstance(frame_op_idx, set):
            return {
                idx: float(np.dot(op_vectorized, self._array[:, idx]).real) for idx in frame_op_idx
            }
        if frame_op_idx is None:
            return np.array(np.dot(op_vectorized, self._array).real)
        raise TypeError(
            f"The optional ``frame_op_idx`` can either be a single or sequence of integers, not a {type(frame_op_idx)}."
        )

    @classmethod
    def from_vectors(cls, frame_vectors: np.ndarray) -> Self:
        r"""Initialize a frame from non-normalized bloch vectors :math:`|\psi \rangle`.

        Args:
            frame_vectors: list of vectors :math:`|\psi \rangle`. The length of the list corresponds to
                the number of operators of the frame. Each vector is of shape :math:`(\mathrm{dim},)` where :math:`\mathrm{dim}`
                is the :attr:`.dimension` of the Hilbert space on which the frame acts. The resulting frame
                operators :math:`\Pi = |\psi \rangle \langle \psi|` are of shape :math:`(\mathrm{dim}, \mathrm{dim})` as expected.

        Returns:
            The frame corresponding to the vectors.
        """
        return cls([Operator(np.outer(vec, vec.conj())) for vec in frame_vectors])
