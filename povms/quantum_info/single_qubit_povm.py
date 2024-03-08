"""TODO."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

from .multi_qubit_povm import MultiQubitPOVM


class SingleQubitPOVM(MultiQubitPOVM):
    """Class to represent a set of IC single-qubit POVM operators."""

    def __init__(self, povm_ops: list[Operator]):
        """Initialize from explicit POVM operators."""
        super().__init__(povm_ops)

        # TODO: why does SingleQubitPOVM have these attributes but neither MultiQubitPOVM nor
        # BasePOVM have them?
        self.povm_pauli_ops = [SparsePauliOp.from_operator(op) for op in self.povm_operators]
        self.povm_pauli_decomp = [
            SingleQubitPOVM.pauli_op_to_array(pauli_op) for pauli_op in self.povm_pauli_ops
        ]

    def _check_validity(self) -> bool:
        """TODO.

        Returns:
            TODO.

        Raises:
            ValueError: TODO.
        """
        if not self.dimension == 2:
            raise ValueError(
                f"Dimension of Single Qubit POVM operator space should be 2, not {self.dimension}."
            )
        return super()._check_validity()

    @staticmethod
    def pauli_op_to_array(pauli_op: SparsePauliOp) -> np.ndarray:
        """Convert a single-qubit SparsePauliOp into an array ``[c0, c1, c2, c3]``.

        In the returned array the indices represent paulis as ``{"I": 0, "X": 1, "Y": 2, "Z": 3}``.

        Args:
            pauli_op: TODO.

        Returns:
            TODO.

        Raises:
            ValueError: TODO.
        """
        labels: np.ndarray = np.zeros(4, dtype=complex)

        for pauli_idx, coeff in pauli_op.label_iter():
            if pauli_idx == "I":
                labels[0] = coeff
            elif pauli_idx == "X":
                labels[1] = coeff
            elif pauli_idx == "Y":
                labels[2] = coeff
            elif pauli_idx == "Z":
                labels[3] = coeff
            else:
                raise ValueError(f"Unexpected pauli string {pauli_idx}")

        return np.real_if_close(labels)
