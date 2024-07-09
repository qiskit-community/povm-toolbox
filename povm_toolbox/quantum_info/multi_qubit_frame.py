# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""MultiQubitFrame."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self  # pragma: no cover

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override  # pragma: no cover

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator, SparsePauliOp

from povm_toolbox.utilities import matrix_to_double_ket

from .base import BaseFrame


class MultiQubitFrame(BaseFrame[int]):
    """Class that collects all information that any frame of multiple qubits should specify.

    This is a representation of an operator-valued vector space frame. The effects are specified as
    a list of :class:`~qiskit.quantum_info.Operator`.

    .. note::
       This is a base class which collects functionality common to various subclasses. As an
       end-user you would not use this class directly. Check out :mod:`povm_toolbox.quantum_info`
       for more general information.
    """

    def __init__(self, list_operators: list[Operator]) -> None:
        """Initialize from explicit operators.

        Args:
            list_operators: list that contains the explicit frame operators. The length of the list
                is the number of operators of the frame.


        Raises:
            ValueError: if the frame operators do not have a correct shape. They should all be
                hermitian and of the same dimension.
        """
        self._num_operators: int
        self._dimension: int
        self._operators: list[Operator]
        self._pauli_operators: list[dict[str, complex]] | None
        self._array: np.ndarray
        self._informationally_complete: bool

        self.operators = list_operators

    def __repr__(self) -> str:
        """Return the string representation of a :class:`.MultiQubitFrame` instance."""
        f_subsystems = f"(num_qubits={self.num_subsystems})" if self.num_subsystems > 1 else ""
        return f"{self.__class__.__name__}{f_subsystems}<{self.num_operators}> at {hex(id(self))}"

    @property
    def informationally_complete(self) -> bool:
        """If the frame spans the entire Hilbert space."""
        return self._informationally_complete

    @property
    def dimension(self) -> int:
        """The dimension of the Hilbert space on which the effects act."""
        return self._dimension

    @property
    def num_operators(self) -> int:
        """The number of effects of the frame."""
        return self._num_operators

    @property
    def operators(self) -> list[Operator]:
        """Return the list of frame operators."""
        return self._operators

    @operators.setter
    def operators(self, new_operators: list[Operator]):
        """Set the frame operators."""
        self._num_operators = len(new_operators)
        self._dimension = new_operators[0].dim[0]
        for frame_op in new_operators:
            if not (self._dimension == frame_op.dim[0] and self._dimension == frame_op.dim[1]):
                raise ValueError(
                    f"Frame operators need to be square ({frame_op.dim[0]},{frame_op.dim[1]}) and "
                    "all of the same dimension."
                )

        self._operators = new_operators

        self._pauli_operators = None

        self._array = np.ndarray((self.dimension**2, self.num_operators), dtype=complex)
        for k, frame_op in enumerate(new_operators):
            self._array[:, k] = matrix_to_double_ket(frame_op.data)

        self._informationally_complete = bool(
            np.linalg.matrix_rank(self._array) == self.dimension**2
        )

        self._check_validity()

    @property
    def pauli_operators(self) -> list[dict[str, complex]]:
        """Convert the internal frame operators to Pauli form.

        .. warning::
           The conversion to Pauli form can be computationally intensive.

        Returns:
            The frame operators in Pauli form. Each frame operator is returned as a dictionary
            mapping Pauli labels to coefficients.

        Raises:
            QiskitError: when the frame operators could not be converted to Pauli form (e.g. when
                they are not N-qubit operators).
        """
        if self._pauli_operators is None:
            try:
                self._pauli_operators = [
                    dict(SparsePauliOp.from_operator(op).label_iter()) for op in self.operators
                ]
            except QiskitError as exc:
                raise QiskitError(
                    f"Failed to convert frame operators to Pauli form: {exc.message}"
                ) from exc
        return self._pauli_operators

    def _check_validity(self) -> None:
        """Check if frame axioms are fulfilled.

        Raises:
            ValueError: if any one of the frame operators is not hermitian.
        """
        for k, op in enumerate(self.operators):
            if not np.allclose(op, op.adjoint(), atol=1e-5):
                raise ValueError(f"The {k}-the frame operator is not hermitian.")

    def __getitem__(self, index: slice) -> Operator | list[Operator]:
        """Return a frame operator or a list of frame operators.

        Args:
            index: indicate the operator(s) to be returned.

        Returns:
            The operator or list of operators corresponding to the index.
        """
        return self.operators[index]

    def __len__(self) -> int:
        """Return the number of operators of the frame."""
        return self.num_operators

    def __array__(self) -> np.ndarray:
        """Return the array representation of the frame.

        The array has a shape :math:`(``self.dimension``**2, ``self.num_operators``)`.
        """
        return self._array

    @override
    def analysis(
        self,
        hermitian_op: SparsePauliOp | Operator,
        frame_op_idx: int | set[int] | None = None,
    ) -> float | dict[int, float] | np.ndarray:
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
            "The optional `frame_op_idx` can either be a single or set of integers, not a "
            f"{type(frame_op_idx)}."
        )

    @classmethod
    def from_vectors(cls, frame_vectors: np.ndarray) -> Self:
        r"""Initialize a frame from non-normalized bloch vectors.

        The non-normalized Bloch vectors are given by :math:`|\tilde{\psi}_k \rangle =
        \sqrt{\gamma_k} |\psi_k \rangle`. The resulting frame operators are :math:`F_k = \gamma_k
        |\psi_k \rangle \langle \psi_k |` where :math:`\gamma_k` is the trace of the :math:`k`'th
        frame operator.

        Args:
            frame_vectors: list of vectors :math:`|\tilde{\psi_k} \rangle`. The length of the list
                corresponds to the number of operators of the frame. Each vector is of shape
                :math:`(\mathrm{dim},)` where :math:`\mathrm{dim}` is the :attr:`.dimension` of the
                Hilbert space on which the frame acts.

        Returns:
            The frame corresponding to the vectors.
        """
        return cls([Operator(np.outer(vec, vec.conj())) for vec in frame_vectors])
