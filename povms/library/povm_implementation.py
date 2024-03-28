"""TODO."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
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

    @abstractmethod
    def _build_qc(self) -> QuantumCircuit:
        """Return the parametetrized quantum circuit to implement the POVM."""

    # specific to randomized measurements
    @abstractmethod
    def distribute_shots(self, shots: int) -> Counter[tuple]:
        """Return a list with POVM label and associated number of shots.

        This method can be used in the case of randomized measurements. Otherwise, it should
        allocate all the shots to the single fixed measurement.

        Args:
            shots: total number of shots to be performed.

        Returns:
            The distribution of the shots among the different POVMs constituting the overall POVM.
        """

    # specific to randomized measurements
    @abstractmethod
    def get_pvm_parameter(self, pvm_idx: tuple[int, ...]) -> np.ndarray:
        """Return the concrete parameter values associated to a POVM label.

        Args:
            pvm_idx: label indicating a specific POVM. Its structure depends on the implementations.

        Returns:
            Parameter values for the specified POVM.
        """

    # specific to randomized measurements
    @abstractmethod
    def get_outcome_label(
        self, pvm_idx: tuple[int, ...], bitstring_outcome: str
    ) -> tuple[int, ...]:
        """Transform a POVM label and a bitstring outcome to a POVM outcome."""

    @abstractmethod
    def to_povm(self) -> BasePOVM:
        """Return the corresponding POVM."""
