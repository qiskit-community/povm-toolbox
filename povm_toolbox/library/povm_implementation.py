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
from dataclasses import dataclass
from typing import Generic, TypeVar

from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers import DataBin
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.transpiler import StagedPassManager

from povm_toolbox.quantum_info.base_povm import BasePOVM

MetadataT = TypeVar("MetadataT", bound="POVMMetadata")


class POVMImplementation(ABC, Generic[MetadataT]):
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

    @abstractmethod
    def to_sampler_pub(
        self,
        circuit: QuantumCircuit,
        parameter_values: BindingsArray,
        shots: int,
        pass_manager: StagedPassManager,
    ) -> tuple[SamplerPub, MetadataT]:
        """Append the measurement circuit(s) to the supplied circuit.

        This method takes a supplied circuit and append the measurement circuit(s)
        to it. If the measurement circuit is parametrized, its parameters values
        should be concatenated with the parameter values associated with the supplied
        quantum circuit.

        Args:
            circuit: A quantum circuit.
            parameter_values: A bindings array.
            shots: A specific number of shots to run with.

        Returns:
            A tuple of a sampler pub and a dictionnary of metadata which include
            the ``POVMImplementation`` objetc itself. The metadata should contain
            all the information neceassary to extract the POVM outcomes out of raw
            bitstrings.
        """
        # TODO: figure out if it would be better to pass these arguments as a
        #    ``SamplerPubLike`` object or even as a ``SamplerPub`` object.

        # TODO: is it the right place to coerce the ``SamplerPub`` ? Or should
        # just return a ``SamplerPubLike`` object that the SamplerV2 will coerce?

    @abstractmethod
    def get_counts_from_raw(
        self,
        data: DataBin,
        povm_metadata: MetadataT,
        loc: int | tuple[int, ...] | None = None,
    ):
        """TODO."""

    @abstractmethod
    def to_povm(self) -> BasePOVM:
        """Return the corresponding POVM."""


@dataclass
class POVMMetadata:
    """TODO."""

    povm: POVMImplementation
