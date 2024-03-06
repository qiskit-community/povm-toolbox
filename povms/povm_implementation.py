"""TODO."""

from __future__ import annotations

from abc import ABC, abstractmethod

from qiskit.circuit import QuantumCircuit
from .base_povm import BasePOVM


class POVMImplementation(ABC):
    """TODO."""

    def __init__(
        self,
        n_qubit: int,
    ) -> None:
        """TODO.

        Args:
            n_qubit: TODO.
        """
        super().__init__()
        self.n_qubit = n_qubit
        self.parametrized_qc = self._build_qc()

    @abstractmethod
    def _build_qc(self) -> QuantumCircuit:
        """TODO."""

    @abstractmethod
    def get_parameter_and_shot(self, shot: int) -> QuantumCircuit:
        """TODO."""

    @abstractmethod
    def to_povm(self) -> BasePOVM:
        """TODO."""
