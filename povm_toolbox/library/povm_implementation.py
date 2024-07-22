# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The interface for "ready-to-go" POVM implementations."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import Counter
from copy import copy
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from qiskit.circuit import AncillaRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.converters import circuit_to_dag
from qiskit.primitives.containers import DataBin
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.bit_array import BitArray
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.transpiler import StagedPassManager

from povm_toolbox.quantum_info.base import BasePOVM

if TYPE_CHECKING:
    from .metadata import POVMMetadata  # pragma: no cover

LOGGER = logging.getLogger(__name__)

MetadataT = TypeVar("MetadataT", bound="POVMMetadata")
"""The metadata type variable bound to :class:`.POVMMetadata`."""


class POVMImplementation(ABC, Generic[MetadataT]):
    """The abstract base interface for all POVM implementations in this library.

    Since this is an `abstract` class, an end-user will not actually create an instance of it at
    runtime. Instead, you should look at the various concrete implementations of this interface
    documented in :mod:`~povm_toolbox.library`.
    """

    classical_register_name: str = "povm_measurement_creg"
    """The name given to the classical bit register in which the POVM outcomes are stored.

    The :class:`~qiskit.primitives.containers.data_bin.DataBin` container result object will have an
    attribute with this name, which will contain the raw measurement data.
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        measurement_layout: list[int] | None = None,  # TODO: add | Layout
        insert_barriers: bool = False,
    ) -> None:
        """Initialize the POVMImplementation.

        Args:
            num_qubits: number of logical qubits in the system.
            measurement_layout: optional list of indices specifying on which qubits the POVM acts.
                See :attr:`.measurement_layout` for more details.
            insert_barriers: whether to insert a barrier between the composed circuits. This is not
                done by default but can prove useful when visualizing the composed circuit.
        """
        self.num_qubits: int = num_qubits
        """The number of logical qubits in the system."""

        self.measurement_layout: list[int] | None = measurement_layout
        """An optional list of indices specifying on which qubits the POVM acts.

        If ``None``, two cases can be distinguished:

        1. if a circuit supplied to the :meth:`.compose_circuits` has been transpiled, its final
           transpile layout will be used as default value,
        2. otherwise, a simple one-to-one layout ``list(range(num_qubits))`` is used.
        """

        self.insert_barriers: bool = insert_barriers
        """Whether to insert a barrier between the original circuit and the measurement circuit
        produced by this POVM implementation.
        """

        self.measurement_circuit: QuantumCircuit
        """The :class:`~qiskit.circuit.quantumcircuit.QuantumCircuit` actually implementing this
        POVM's measurement."""

    def __repr__(self) -> str:
        """Return the string representation of a POVMImplementation instance."""
        # NOTE: this is not covered by tests because it is a fallback intended to give a meaningful
        # representation for actual implementations of this interface which do not overwrite this
        # function. However, all actual implementation in this library do overwrite this and, thus,
        # this fallback is not executed during the test coverage.
        return f"{self.__class__.__name__}(num_qubits={self.num_qubits})"  # pragma: no cover

    @abstractmethod
    def definition(self) -> BasePOVM:
        """Return the corresponding quantum-informational POVM representation."""

    @abstractmethod
    def _build_qc(self) -> QuantumCircuit:
        """Return the parametrized quantum circuit to implement the POVM."""

    @abstractmethod
    def to_sampler_pub(
        self,
        circuit: QuantumCircuit,
        circuit_binding: BindingsArray,
        shots: int,
        *,
        pass_manager: StagedPassManager | None = None,
    ) -> tuple[SamplerPub, MetadataT]:
        """Append the measurement circuit(s) to the supplied circuit.

        This method takes a supplied circuit and appends the measurement circuit(s) to it. If the
        measurement circuit is parametrized, its parameters values should be concatenated with the
        parameter values associated with the supplied quantum circuit.

        Args:
            circuit: A quantum circuit.
            circuit_binding: A bindings array.
            shots: A specific number of shots to run with.
            pass_manager: An optional transpilation pass manager. After the supplied circuit has
                been composed with the measurement circuit, the pass manager will be used to
                transpile the composed circuit.

        Returns:
            A tuple of a sampler pub and a dictionary of metadata which includes the
            :class:`.POVMImplementation` object itself. The metadata should contain all the
            information necessary to extract the POVM outcomes out of raw bitstrings.
        """

    def compose_circuits(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Compose the circuit to sample from, with the measurement circuit.

        If the measurement circuit requires some ancilla qubits, this method will inspect the input
        circuit. If the input circuit has some idling qubits available, they will be used as ancilla
        measurement qubits. If not enough idling qubits are available, this method will add the
        necessary number of qubits to the input circuit before composing it with the measurement
        circuit.

        Args:
            circuit: The quantum circuit to be sampled from.

        Raises:
            ValueError: if the number of qubits specified by `self.measurement_layout` does not
                match the number of qubits on which this POVM implementation acts.
            CircuitError: if an error has occurred when adding the classic register, used to save
                POVM results, to the input circuit.

        Returns:
            The composition of the supplied quantum circuit with the :attr:`.measurement_circuit` of
            this POVM implementation.
        """
        t1 = time.time()
        LOGGER.info("Starting circuit composition")

        # Create a copy of the circuit and remove final measurements:
        dest_circuit = circuit.copy()
        dest_circuit.remove_final_measurements(inplace=True)

        if self.measurement_layout is None:
            if dest_circuit.layout is None:
                # Basic one-to-one layout
                index_layout = list(range(dest_circuit.num_qubits))
            else:
                # Extract the final layout of the transpiled circuit (ancillas are filtered).
                index_layout = dest_circuit.layout.final_index_layout(filter_ancillas=True)
        else:
            index_layout = self.measurement_layout

        # Check that the number of qubits of the circuit (before the transpilation, if
        # applicable) matches the number of qubits of the POVM implementation.
        if self.num_qubits != len(index_layout):
            raise ValueError(
                f"The supplied measurement layout (specifying {len(index_layout)} qubits) does not"
                f" match this POVM implementation which acts on {self.num_qubits} qubits."
            )

        # Check if the measurement circuit requires some ancilla qubits
        if self.measurement_circuit.num_qubits > self.num_qubits:
            index_layout = copy(index_layout)
            # Get idle qubits in supplied circuit (to be used as ancilla for measurement circuit)
            idle_qubits = list(circuit_to_dag(dest_circuit).idle_wires())
            # Get the indices of the idle qubits
            idle_index = [dest_circuit.qubits.index(qubit) for qubit in idle_qubits]
            # Remove the idle qubits that will be measured (as specified by index_layout)
            idle_index = [idx for idx in idle_index if idx not in index_layout]
            idle_index.sort()

            # If exactly enough idle qubits available, we use all of them
            if self.num_qubits + len(idle_index) == self.measurement_circuit.num_qubits:
                index_layout += idle_index
            # If not enough idle qubits are available, we add some ancilla qubits
            elif self.num_qubits + len(idle_index) < self.measurement_circuit.num_qubits:
                index_layout += idle_index
                ancilla_register = AncillaRegister(
                    self.measurement_circuit.num_qubits - len(index_layout),
                    name="measurement_ancilla",
                )
                ancilla_layout = list(
                    range(dest_circuit.num_qubits, dest_circuit.num_qubits + ancilla_register.size)
                )
                index_layout += ancilla_layout
                dest_circuit.add_register(ancilla_register)
            # If more than enough idle qubits are available, we pick only the necessary number
            else:
                index_layout += idle_index[: self.measurement_circuit.num_qubits - self.num_qubits]

        try:
            dest_circuit.add_register(*self.measurement_circuit.cregs)
        except CircuitError as exc:
            raise CircuitError(
                f"{exc}\nNote: the supplied quantum circuit should not have a classical register"
                " which has the same name as the classical register that the POVM"
                " implementation uses to store measurement outcomes (creg name: "
                f"'{self.classical_register_name}').\nTo fix it, either delete this register or "
                " change the name of the register of the supplied circuit or of the"
                " POVM implementation."
            ) from exc

        if self.insert_barriers:
            dest_circuit.barrier()

        # Compose the two circuits with the correct routing.
        ret = dest_circuit.compose(
            self.measurement_circuit, qubits=index_layout, clbits=self.measurement_circuit.clbits
        )

        t2 = time.time()
        LOGGER.info(f"Finished circuit composition. Took {t2 - t1:.6f}s")

        return ret

    @abstractmethod
    def reshape_data_bin(self, data: DataBin) -> DataBin:
        """Reshapes the provided data.

        This method should reshape the provided data to the output dimensions expected by the
        end-user. That is, the dimensions should match those of the
        :class:`qiskit.primitives.SamplerPubLike` object provided by the user when submitting their
        primitive job.

        Args:
            data: The raw primitive result data still shaped according to the internally submitted
                :class:`.POVMSamplerJob`.

        Returns:
            A new data structure of the correct shape.
        """

    def _get_bitarray(self, data: DataBin) -> BitArray:
        """Get the bitstrings of the POVM's classical register name from the data object.

        Args:
            data: The data object from which to extract the bitstrings.

        Returns:
            The bitstrings.
        """
        return getattr(data, self.classical_register_name)

    @abstractmethod
    def _povm_outcomes(
        self,
        bit_array: BitArray,
        povm_metadata: MetadataT,
        *,
        loc: int | tuple[int, ...] | None = None,
    ) -> list[tuple[int, ...]]:
        """Convert the raw bitstrings into POVM outcomes based on the associated metadata.

        Args:
            bit_array: The raw bitstrings.
            povm_metadata: The associated metadata.
            loc: an optional location to slice the bitstrings.

        Returns:
            The converted POVM outcomes.
        """

    def get_povm_counts_from_raw(
        self,
        data: DataBin,
        povm_metadata: MetadataT,
        *,
        loc: int | tuple[int, ...] | None = None,
    ) -> np.ndarray | Counter:
        """Get the POVM counts.

        Args:
            data: The raw sampled data.
            povm_metadata: The associated metadata.
            loc: an optional location to slice the bitstrings.

        Returns:
            The POVM counts.
        """
        bit_array = self._get_bitarray(data)

        if loc is not None:
            return Counter(self._povm_outcomes(bit_array, povm_metadata, loc=loc))

        if bit_array.ndim == 0:
            return np.array([Counter(self._povm_outcomes(bit_array, povm_metadata))], dtype=object)

        shape = bit_array.shape
        outcomes_array: np.ndarray = np.ndarray(shape=shape, dtype=object)
        for idx in np.ndindex(shape):
            outcomes_array[idx] = Counter(self._povm_outcomes(bit_array, povm_metadata, loc=idx))
        return outcomes_array

    def get_povm_outcomes_from_raw(
        self,
        data: DataBin,
        povm_metadata: MetadataT,
        *,
        loc: int | tuple[int, ...] | None = None,
    ) -> np.ndarray | list[tuple[int, ...]]:
        """Get the POVM bitstrings.

        Args:
            data: The raw sampled data.
            povm_metadata: The associated metadata.
            loc: an optional location to slice the bitstrings.

        Returns:
            The POVM bitstrings.
        """
        bit_array = self._get_bitarray(data)

        if loc is not None or bit_array.ndim == 0:
            return self._povm_outcomes(bit_array, povm_metadata, loc=loc)

        shape = bit_array.shape
        outcomes_array: np.ndarray = np.ndarray(shape=shape, dtype=object)
        for idx in np.ndindex(shape):
            outcomes_array[idx] = self._povm_outcomes(bit_array, povm_metadata, loc=idx)
        return outcomes_array
