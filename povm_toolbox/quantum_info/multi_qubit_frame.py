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
from typing import cast

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self  # pragma: no cover

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override  # pragma: no cover

from math import prod
from typing import TypeVar

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator, SparsePauliOp

from povm_toolbox.utilities import matrix_to_double_ket

from .base import BaseFrame

LabelMultiQubitT = TypeVar("LabelMultiQubitT", int, tuple[int, ...])
"""Each operator in the frame spanning multiple qubits is identified by a label.

This is the type of these labels. They are either integers or tuples of integers.
"""


class MultiQubitFrame(BaseFrame[LabelMultiQubitT]):
    """Class that collects all information that any frame of multiple qubits should specify.

    This is a representation of an operator-valued vector space frame. The effects are specified as
    a list of :class:`~qiskit.quantum_info.Operator`.

    .. note::
       This is a base class which collects functionality common to various subclasses. As an
       end-user you would not use this class directly. Check out :mod:`povm_toolbox.quantum_info`
       for more general information.
    """

    def __init__(
        self, list_operators: list[Operator], *, shape: tuple[int, ...] | None = None
    ) -> None:
        """Initialize from explicit operators.

        Args:
            list_operators: list that contains the explicit frame operators. The length of the list
                is the number of operators of the frame.
            shape: the shape defining the indexing of operators in ``list_operators``. If ``None``,
                the default shape is ``(self.num_operators,)``.


        Raises:
            ValueError: if the length of ``list_operators`` is not compatible with ``shape``.
            ValueError: if the frame operators do not have a correct shape. They should all be
                hermitian and of the same dimension.
        """
        self._num_operators: int
        self._dimension: int
        self._operators: list[Operator]
        self._pauli_operators: list[dict[str, complex]] | None
        self._array: np.ndarray
        self._informationally_complete: bool

        self._num_operators = len(list_operators)
        self.shape = shape or (self._num_operators,)
        self._dimension = list_operators[0].dim[0]
        for frame_op in list_operators:
            if not (self._dimension == frame_op.dim[0] and self._dimension == frame_op.dim[1]):
                raise ValueError(
                    f"Frame operators need to be square ({frame_op.dim[0]},{frame_op.dim[1]}) and "
                    "all of the same dimension."
                )

        self._operators = list_operators

        self._pauli_operators = None

        self._array = np.ndarray((self.dimension**2, self.num_operators), dtype=complex)
        for k, frame_op in enumerate(list_operators):
            self._array[:, k] = matrix_to_double_ket(frame_op.data)

        self._informationally_complete = bool(
            np.linalg.matrix_rank(self._array) == self.dimension**2
        )

        self._check_validity()

    def __repr__(self) -> str:
        """Return the string representation of a :class:`.MultiQubitFrame` instance."""
        f_subsystems = f"(num_qubits={self.num_subsystems})" if self.num_subsystems > 1 else ""
        repr_str = (
            f"{self.__class__.__name__}{f_subsystems}<{','.join(map(str, self.shape))}> "
            f"at {hex(id(self))}"
        )
        return repr_str

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
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the frame."""
        return self._shape

    @shape.setter
    def shape(self, new_shape: tuple[int, ...]) -> None:
        """Set the shape of the frame."""
        if prod(new_shape) != self.num_operators:
            raise ValueError(
                f"The shape {new_shape} is not compatible with the number of operators in the frame"
                f" ({self.num_operators})."
            )
        self._shape = new_shape

    @property
    def operators(self) -> list[Operator]:
        """Return the list of frame operators."""
        return self._operators

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

    def _ravel_index(self, index: LabelMultiQubitT) -> int:
        """Ravel a multi-index into a flat index when applicable..

        Args:
            index: an integer index or multi-index matching the shape of the frame.

        Returns:
            A flattened integer index.

        Raises:
            ValueError: if an integer index is supplied for a frame that has a multi-dimensional
                shape.
        """
        if isinstance(index, tuple):
            return int(np.ravel_multi_index(multi_index=index, dims=self.shape))
        if len(self.shape) > 1:
            raise ValueError(
                f"The integer index `{index}` is invalid because the frame has a {len(self.shape)}-"
                "dimensional shape."
            )
        return index

    def __getitem__(self, index: LabelMultiQubitT) -> Operator | list[Operator]:
        """Return a frame operator or a list of frame operators.

        Args:
            index: indicate the operator(s) to be returned.

        Returns:
            The operator or list of operators corresponding to the index.
        """
        return self.operators[self._ravel_index(index)]

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
        frame_op_idx: LabelMultiQubitT | set[LabelMultiQubitT] | None = None,
    ) -> float | dict[LabelMultiQubitT, float] | np.ndarray:
        if isinstance(hermitian_op, SparsePauliOp):
            hermitian_op = hermitian_op.to_operator()
        op_vectorized = np.conj(matrix_to_double_ket(hermitian_op.data))

        if isinstance(frame_op_idx, (int, tuple)):
            return float(
                np.dot(op_vectorized, self._array[:, self._ravel_index(frame_op_idx)]).real
            )
        if isinstance(frame_op_idx, set):
            return {
                idx: float(np.dot(op_vectorized, self._array[:, self._ravel_index(idx)]).real)
                for idx in frame_op_idx
            }
        if frame_op_idx is None:
            return cast(np.ndarray, np.asarray(np.dot(op_vectorized, self._array).real))
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
