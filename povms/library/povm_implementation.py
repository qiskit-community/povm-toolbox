"""TODO."""

from __future__ import annotations

from abc import ABC, abstractmethod

from qiskit.circuit import QuantumCircuit

from povms.quantum_info.base_povm import BasePOVM


class POVMImplementation(ABC):
    """Abstract base class that contains all methods that any specific POVMImplementation subclass should implement."""

    def __init__(
        self,
        n_qubit: int,
    ) -> None:
        """Initialize the POVMImplementation.

        Args:
            n_qubit: number of logical qubits in the system.
        """
        super().__init__()
        self.n_qubit = n_qubit
        self.parametrized_qc = self._build_qc()

    @abstractmethod
    def _build_qc(self) -> QuantumCircuit:
        """Return the parametetrized quantum circuit to implement the POVM."""

    @abstractmethod
    def get_parameter_and_shot(self, shot: int) -> QuantumCircuit:
        """TODO: change, relevant for randomized measurements only."""

    @abstractmethod
    def to_povm(self) -> BasePOVM:
        """Return the corresponding POVM."""
