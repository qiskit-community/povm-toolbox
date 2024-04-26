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
from collections import Counter
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers import DataBin
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.bit_array import BitArray
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.transpiler import StagedPassManager

from povm_toolbox.quantum_info.base_povm import BasePOVM

if TYPE_CHECKING:
    from .metadata import POVMMetadata

MetadataT = TypeVar("MetadataT", bound="POVMMetadata")


class POVMImplementation(ABC, Generic[MetadataT]):
    """Abstract base class that contains all methods that any specific POVMImplementation subclass should implement."""

    classical_register_name = "povm_measurement_creg"

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

    def __repr__(self) -> str:
        """Return the string representation of a POVMImplementation instance."""
        return f"{self.__class__.__name__}(n_qubits={self.n_qubit})"

    @abstractmethod
    def _build_qc(self) -> QuantumCircuit:
        """Return the parametetrized quantum circuit to implement the POVM."""

    @abstractmethod
    def to_sampler_pub(
        self,
        circuit: QuantumCircuit,
        circuit_binding: BindingsArray,
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
            A tuple of a sampler pub and a dictionary of metadata which include
            the ``POVMImplementation`` object itself. The metadata should contain
            all the information neceassary to extract the POVM outcomes out of raw
            bitstrings.
        """
        # TODO: figure out if it would be better to pass these arguments as a
        #    ``SamplerPubLike`` object or even as a ``SamplerPub`` object.

        # TODO: is it the right place to coerce the ``SamplerPub`` ? Or should
        # just return a ``SamplerPubLike`` object that the SamplerV2 will coerce?

    @abstractmethod
    def reshape_data_bin(self, data: DataBin) -> DataBin:
        """TODO."""

    def _get_bitarray(self, data: DataBin) -> BitArray:
        """TODO."""
        return getattr(data, self.classical_register_name)

    @abstractmethod
    def _counter(
        self,
        bit_array: BitArray,
        povm_metadata: MetadataT,
        loc: int | tuple[int, ...] | None = None,
    ) -> Counter:
        """TODO."""

    def get_counts_from_raw(
        self,
        data: DataBin,
        povm_metadata: MetadataT,
        loc: int | tuple[int, ...] | None = None,
    ) -> np.ndarray | Counter:
        """TODO."""
        bit_array = self._get_bitarray(data)

        if loc is not None:
            return self._counter(bit_array, povm_metadata, loc)

        if bit_array.ndim == 0:
            return np.array([self._counter(bit_array, povm_metadata)])

        shape = bit_array.shape
        counters_array: np.ndarray = np.ndarray(shape=shape, dtype=object)
        for idx in np.ndindex(shape):
            counters_array[idx] = self._counter(bit_array, povm_metadata, idx)
        return counters_array

    @abstractmethod
    def definition(self) -> BasePOVM:
        """Return the corresponding POVM."""
