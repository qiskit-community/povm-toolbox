"""TODO."""

from __future__ import annotations

from functools import lru_cache
from typing import TypeVar

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp, SuperOp

from povms.utilities import double_ket_to_matrix, matrix_to_double_ket

from .base_povm import BasePOVM

# Create a generic variable that can be 'MultiQubitPOVM', or any subclass.
T = TypeVar("T", bound="MultiQubitPOVM")


class MultiQubitPOVM(BasePOVM):
    """Class that collects all information that any MultiQubit POVM should specifiy.

    This is a representation of a positive operator-valued measure (POVM). The effects are
    sepcified as a list of :class:`~qiskit.quantum_info.Operator`.
    """

    def __init__(self, povm_ops: list[Operator]) -> None:
        r"""Initialize from explicit POVM operators.

        Args:
            povm_operators: list that contains the explicit POVM operators. Each Operator
                in the list corresponds to a POVM effect. The length of the list is
                the number of outcomes of the POVM.


        Raises:
            ValueError: if the POVM operators do not have a correct shape. They should all
                be square and of the same dimension.
        """
        self._n_outcomes: int = len(povm_ops)
        self._dimension: int = povm_ops[0].dim[0]
        for op in povm_ops:
            if not (self._dimension == op.dim[0] and self._dimension == op.dim[1]):
                raise ValueError(
                    f"POVM operators need to be square ({op.dim[0]},{op.dim[1]}) and all of the same dimension."
                )
        self.povm_operators: list[Operator] = povm_ops
        self.array_ops = None

        self._alphas = np.array([np.trace(povm_op.data) for povm_op in self.povm_operators])

        self._informationlly_complete: bool
        self._frame_superop: SuperOp | None = None
        self._dual_operators: list[Operator] | None = None
        # TODO: should all of the public attributes above become part of the BasePOVM interface?

        self._check_validity()

    @property
    def dimension(self) -> int:
        """Give the dimension of the Hilbert space on which the effects act."""
        return self._dimension

    @property
    def n_outcomes(self) -> int:
        """Give the number of outcomes of the POVM."""
        return self._n_outcomes

    @property
    def alphas(self) -> np.ndarray:
        """Paremeters of the dual frame."""
        return self._alphas

    @alphas.setter
    def alphas(self, var: np.ndarray) -> None:
        self._alphas = var
        self._frame_superop = None
        self._dual_operators = None

    @property
    def frame_superop(self) -> SuperOp:
        """Give the frame superoperator of the POVM.

        CAREFUL: if the frame superoperator is not already computed, this will be computationally heavy.
        TODO: use admonition.
        """
        if self._frame_superop is None:
            mat: np.ndarray = np.zeros((self.dimension**2, self.dimension**2), dtype=complex)
            for k, povm_op in enumerate(self.povm_operators):
                double_ket = matrix_to_double_ket(povm_op.data)
                mat += np.outer(double_ket, double_ket.conjugate()) / self.alphas[k]
            self._frame_superop = SuperOp(mat)
        return self._frame_superop

    @property
    def dual_operators(self) -> list[Operator]:
        """Give the dual operators of the POVM.

        CAREFUL: if the duals are not already computed, this will be computationally heavy.
        """
        if self._dual_operators is None:
            self._dual_operators = [
                Operator(
                    double_ket_to_matrix(
                        np.linalg.solve(
                            self.frame_superop.data,
                            matrix_to_double_ket(povm_op.data) / self.alphas[k],
                        )
                    )
                )
                for k, povm_op in enumerate(self.povm_operators)
            ]
        return self._dual_operators

    # TODO: cleaner implementation that does not trigger B019
    @lru_cache(maxsize=64)  # noqa: B019
    def pauli_operators(self, dual: bool = False) -> list[dict[str, complex]]:
        """Convert the internal POVM operators to Pauli form.

        This method will cache its returned data to avoid re-computation.

        Args:
            dual: False if the pauli decomposistion of the effects should be returned.
                True if the pauli decomposistion of the dual operators should be returned.

        Raises:
            ValueError: when the POVM operators are not N-qubit operators.
        """
        if not dual:
            try:
                return [
                    dict(SparsePauliOp.from_operator(op).label_iter()) for op in self.povm_operators
                ]
            except QiskitError as exc:
                raise ValueError("Failed to convert POVM operators to Pauli form.") from exc
        try:
            return [
                dict(SparsePauliOp.from_operator(op).label_iter()) for op in self.dual_operators
            ]
        except QiskitError as exc:
            raise ValueError("Failed to convert dual operators to Pauli form.") from exc

    def _check_validity(self) -> None:
        r"""Check if POVM axioms are fulfilled.

        Raises:
            ValueError: if any of the POVM operators is not hermitian.
            ValueError: if any of the POVM operators has a negative eigenvalue.
            ValueError: if all POVM operators do not sum to the identity.
        """
        summed_op: np.ndarray = np.zeros((self.dimension, self.dimension), dtype=complex)

        for k, op in enumerate(self.povm_operators):
            if not np.allclose(op, op.adjoint(), atol=1e-5):
                raise ValueError(f"POVM operator {k} is not hermitian.")

            for eigval in np.linalg.eigvalsh(op.data):
                if eigval.real < -1e-6 or np.abs(eigval.imag) > 1e-5:
                    raise ValueError(f"Negative eigenvalue {eigval} in POVM operator {k}.")

            summed_op += op.data

        if not np.allclose(summed_op, np.identity(self.dimension, dtype=complex), atol=1e-5):
            raise ValueError(f"POVM operators not summing up to the identity : \n{summed_op}")

    def __getitem__(self, index: slice) -> Operator | list[Operator]:
        r"""Return a povm operator or a list of povm operators.

        Args:
            index: indicate the operator(s) to be returned.

        Returns:
            The operator or list of operators corresponding to the index.
        """
        return self.povm_operators[index]

    def __len__(self) -> int:
        """Return the number of outcomes of the POVM."""
        return self.n_outcomes

    def get_prob(
        self, rho: DensityMatrix, outcome_idx: int | set[int] | None = None
    ) -> float | dict[int, float] | np.ndarray:
        r"""Return the outcome probablities given a state ``rho``.

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
        if isinstance(outcome_idx, int):
            return float(np.real(np.trace(rho.data @ self.povm_operators[outcome_idx].data)))
        if isinstance(outcome_idx, set):
            return {
                idx: float(np.real(np.trace(rho.data @ self.povm_operators[idx].data)))
                for idx in outcome_idx
            }
        if outcome_idx is None:
            return np.array(
                [np.real(np.trace(rho.data @ povm_op.data)) for povm_op in self.povm_operators]
            )
        raise TypeError(
            f"The optional ``outcome_idx`` can either be a single or sequence of integers, not a {type(outcome_idx)}."
        )

    def get_omegas(
        self, obs: Operator, outcome_idx: int | set[int] | None = None
    ) -> float | dict[int, float] | np.ndarray:
        r"""Return the decomposition weights of observable ``obs`` into the POVM effects.

        Given an obseravble :math:`O` which is in the span of the POVM, one can write the
        observable :math:`O` as the weighted sum of the POVM effects, :math:`O = \sum_k w_k M_k`
        for real weights :math:`w_k`. There might be infinitely many valid sets of weight.
        This method returns a possible set of weights.

        Args:
            obs: the observable to be decomposed into the POVM effects.

        Returns:
            An array of decomposition weights.
        """
        if isinstance(outcome_idx, int):
            return float(np.real(np.trace(obs.data @ self.dual_operators[outcome_idx].data)))
        if isinstance(outcome_idx, set):
            return {
                idx: float(np.real(np.trace(obs.data @ self.dual_operators[idx].data)))
                for idx in outcome_idx
            }
        if outcome_idx is None:
            return np.array(
                [np.real(np.trace(obs.data @ dual_op.data)) for dual_op in self.dual_operators]
            )
        raise TypeError(
            f"The optional ``outcome_idx`` can either be a single or sequence of integers, not a {type(outcome_idx)}."
        )

    @classmethod
    def from_vectors(cls: type[T], povm_vectors: np.ndarray) -> T:
        r"""Initialize a POVM from non-normalized bloch vectors :math:``|psi>``.

        Args:
            povm_vectors: list of vectors :math:``|psi>``. The length of the list corresponds to
                the number of outcomes of the POVM. Each vector is of shape ``(dim,)`` where ``dim``
                is the dimension of the Hilbert space on which the POVM acts. The resulting POVM
                effects :math:``Pi = |psi><psi|`` are of shape ``(dim, dim)`` as expected.

        Returns:
            The POVM corresponding to the vectors.
        """
        povm_operators = []
        for vec in povm_vectors:
            povm_operators.append(Operator(np.outer(vec, vec.conj())))
        return cls(povm_operators)
