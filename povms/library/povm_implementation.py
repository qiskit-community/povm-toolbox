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

from abc import ABC, abstractmethod

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
        self.msmt_qc: QuantumCircuit

    @abstractmethod
    def _build_qc(self) -> QuantumCircuit:
        """Return the parametetrized quantum circuit to implement the POVM."""

    # specific to randomized measurements
    @abstractmethod
    def distribute_shots(self, shots: int) -> list[tuple[int, ...]]:
        """Return a list of sampled PVM labels.

        In the case of PM-simulable POVMs, each time we perfom a measurement we pick a
        random projective measurement among a given set of PVMs. This method return a
        list of labels of length :math:``shots``.

        Args:
            shots: total number of shots to be performed.

        Returns:
            The labels of the :math:``shots`` sampled PVMs.
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

    @abstractmethod
    def get_outcome_label(
        self, pvm_idx: tuple[int, ...], bitstring_outcome: str
    ) -> tuple[int, ...]:
        """Convert a PVM label and a bitsring outcome obtained with it to a POVM outcome."""

    @abstractmethod
    def to_povm(self) -> BasePOVM:
        """Return the corresponding POVM."""
