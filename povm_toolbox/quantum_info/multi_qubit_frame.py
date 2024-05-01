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

from povm_toolbox.utilities import double_ket_to_matrix, matrix_to_double_ket

from .base_frame import BaseFrame


class MultiQubitFrame(BaseFrame):
    """Class that collects all information that any MultiQubit frame should specify.

    This is a representation of an operator-valued vector space frame. The effects are
    specified as a list of :class:`~qiskit.quantum_info.Operator`.
    """

    def __init__(self, list_operators: list[Operator]) -> None:
        r"""Initialize from explicit operators.

        Args:
            list_operators: list that contains the explicit POVM operators. Each Operator
                in the list corresponds to a POVM effect. The length of the list is
                the number of outcomes of the POVM.


        Raises:
            ValueError: if the POVM operators do not have a correct shape. They should all
                be square and of the same dimension.
        """
        self._n_operators: int = len(list_operators)
        self._dimension: int = list_operators[0].dim[0]
        for frame_op in list_operators:
            if not (self._dimension == frame_op.dim[0] and self._dimension == frame_op.dim[1]):
                raise ValueError(
                    f"POVM operators need to be square ({frame_op.dim[0]},{frame_op.dim[1]}) and all of the same dimension."
                )
        self.operators: list[Operator] = list_operators

        self._array: np.ndarray = np.ndarray((self.dimension**2, self.n_operators), dtype=complex)
        for k, frame_op in enumerate(list_operators):
            self._array[:, k] = matrix_to_double_ket(frame_op.data)

        # TODO.
        self._informationally_complete: bool

        self._check_validity()

    @property
    def informationally_complete(self) -> bool:
        """Return if the frame spans the entire Hilbert space."""
        return self._informationally_complete

    @property
    def dimension(self) -> int:
        """Give the dimension of the Hilbert space on which the effects act."""
        return self._dimension

    @property
    def n_operators(self) -> int:
        """Give the number of outcomes of the POVM."""
        return self._n_operators

    @property
    def pauli_operators(self) -> list[dict[str, complex]]:
        """Convert the internal POVM operators to Pauli form.

        Args:
            dual: False if the pauli decomposition of the effects should be returned.
                True if the pauli decomposition of the dual operators should be returned.

        Raises:
            ValueError: when the POVM operators are not N-qubit operators.
        """
        try:
            return [dict(SparsePauliOp.from_operator(op).label_iter()) for op in self.operators]
        except QiskitError as exc:
            raise ValueError("Failed to convert POVM operators to Pauli form.") from exc

    def _check_validity(self) -> None:
        r"""Check if POVM axioms are fulfilled.

        Raises:
            ValueError: if any of the POVM operators is not hermitian.
            ValueError: if any of the POVM operators has a negative eigenvalue.
            ValueError: if all POVM operators do not sum to the identity.
        """
        for k, op in enumerate(self.operators):
            if not np.allclose(op, op.adjoint(), atol=1e-5):
                raise ValueError(f"Frame operator {k} is not hermitian.")

    def __getitem__(self, index: slice) -> Operator | list[Operator]:
        r"""Return a povm operator or a list of povm operators.

        Args:
            index: indicate the operator(s) to be returned.

        Returns:
            The operator or list of operators corresponding to the index.
        """
        return self.operators[index]

    def __len__(self) -> int:
        """Return the number of outcomes of the POVM."""
        return self.n_operators

    def __array__(self) -> np.ndarray:
        """Return the array representation of the frame, with shape.

        The array has a shape :math:``(self.dimension**2, self.n_operators)``.
        """
        return self._array

    def analysis(
        self, op: Operator, outcome_idx: int | set[int] | None = None
    ) -> float | dict[int, float] | np.ndarray:
        r"""Return the outcome probabilities given a state ``rho``.

        Each outcome :math:`k` is associated with an effect :math:`M_k` of the POVM. The probability of obtaining
        the outcome :math:`k` when measuring a state ``rho`` is given by :math:`p_k = Tr[M_k \rho]`.

        Args:
            rho: the input state over which to compute the outcome probabilities.
            outcome_idx: TODO.

        Returns:
            An array of probabilities. The length of the array is given by the number of outcomes of the POVM.

        Raises:
            TypeError: TODO.
        """
        op_vectorized = np.conj(matrix_to_double_ket(op.data))
        if isinstance(outcome_idx, int):
            return float(np.dot(op_vectorized, self._array[:, outcome_idx]).real)
        if isinstance(outcome_idx, set):
            return {
                idx: float(np.dot(op_vectorized, self._array[:, idx]).real) for idx in outcome_idx
            }
        if outcome_idx is None:
            return np.array(np.dot(op_vectorized, self._array).real)
        raise TypeError(
            f"The optional ``outcome_idx`` can either be a single or sequence of integers, not a {type(outcome_idx)}."
        )

    def synthesis(
        self,
        frame_coef: np.ndarray,
        bias: np.ndarray | None = None,
    ) -> Operator:
        """Adjoint of the analysis operator."""
        if bias is not None:
            np.multiply(bias, frame_coef, out=frame_coef)
        if frame_coef.shape != (self.n_operators,):
            raise ValueError
        op = np.dot(self._array, frame_coef)
        return Operator(double_ket_to_matrix(op))

    @classmethod
    def from_vectors(cls, povm_vectors: np.ndarray) -> Self:
        r"""Initialize a POVM from non-normalized bloch vectors :math:``|psi>``.

        Args:
            povm_vectors: list of vectors :math:``|psi>``. The length of the list corresponds to
                the number of outcomes of the POVM. Each vector is of shape ``(dim,)`` where ``dim``
                is the dimension of the Hilbert space on which the POVM acts. The resulting POVM
                effects :math:``Pi = |psi><psi|`` are of shape ``(dim, dim)`` as expected.

        Returns:
            The POVM corresponding to the vectors.
        """
        return cls([Operator(np.outer(vec, vec.conj())) for vec in povm_vectors])
