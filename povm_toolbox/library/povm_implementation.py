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

import logging
import time
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.primitives.containers import DataBin
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.bit_array import BitArray
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.transpiler import StagedPassManager, TranspileLayout

from povm_toolbox.quantum_info.base_povm import BasePOVM

if TYPE_CHECKING:
    from .metadata import POVMMetadata

LOGGER = logging.getLogger(__name__)

MetadataT = TypeVar("MetadataT", bound="POVMMetadata")


class POVMImplementation(ABC, Generic[MetadataT]):
    """The abstract base interface for all POVM implementations in this library."""

    _classical_register_name = "povm_measurement_creg"

    def __init__(
        self,
        n_qubit: int,
        measurement_layout: list[int] | None = None,  # TODO: add | Layout
    ) -> None:
        """Initialize the POVMImplementation.

        Args:
            n_qubit: number of logical qubits in the system.
            measurement_layout: list of indices specifying on which qubits the POVM
                acts. If None, two cases can be distinguished: 1) if a circuit supplied
                to the :meth:`.compose_circuits` has been transpiled, its final
                transpile layout will be used as default value, 2) otherwise, a
                simple one-to-one layout ``list(range(n_qubits))`` is used.
        """
        super().__init__()
        self.n_qubit = n_qubit
        self.measurement_layout = measurement_layout

        self.msmt_qc: QuantumCircuit

    def __repr__(self) -> str:
        """Return the string representation of a POVMImplementation instance."""
        return f"{self.__class__.__name__}(n_qubits={self.n_qubit})"

    @abstractmethod
    def _build_qc(self) -> QuantumCircuit:
        """Return the parametrized quantum circuit to implement the POVM."""

    @abstractmethod
    def to_sampler_pub(
        self,
        circuit: QuantumCircuit,
        circuit_binding: BindingsArray,
        shots: int,
        pass_manager: StagedPassManager | None = None,
    ) -> tuple[SamplerPub, MetadataT]:
        """Append the measurement circuit(s) to the supplied circuit.

        This method takes a supplied circuit and append the measurement circuit(s)
        to it. If the measurement circuit is parametrized, its parameters values
        should be concatenated with the parameter values associated with the supplied
        quantum circuit.

        Args:
            circuit: A quantum circuit.
            circuit_binding: A bindings array.
            shots: A specific number of shots to run with.
            pass_manager: An optional pass manager. After the supplied circuit has
                been composed with the measurement circuit, the pass manager will
                transpile the composed circuit.

        Returns:
            A tuple of a sampler pub and a dictionary of metadata which include
            the ``POVMImplementation`` object itself. The metadata should contain
            all the information necessary to extract the POVM outcomes out of raw
            bitstrings.
        """
        # TODO: figure out if it would be better to pass these arguments as a
        #    ``SamplerPubLike`` object or even as a ``SamplerPub`` object.

        # TODO: is it the right place to coerce the ``SamplerPub`` ? Or should
        # just return a ``SamplerPubLike`` object that the SamplerV2 will coerce?

    def compose_circuits(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Compose the circuit to sample from, with the measurement circuit.

        Args:
            circuit: Quantum circuit to be sampled from.

        Returns:
            The composition of the supplied quantum circuit with the measurement
            circuit of this POVM implementation.
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

            elif isinstance(dest_circuit.layout, TranspileLayout):
                # Extract the final layout of the transpiled circuit (ancillas are filtered).
                index_layout = dest_circuit.layout.final_index_layout(filter_ancillas=True)
            else:
                raise NotImplementedError
        else:
            index_layout = self.measurement_layout

        # Check that the number of qubits of the circuit (before the transpilation, if
        # applicable) matches the number of qubits of the POVM implementation.
        if self.n_qubit != len(index_layout):
            raise ValueError(
                f"The supplied circuit (acting on {len(index_layout)} qubits)"
                " does not match this POVM implementation which acts on"
                f" {self.n_qubit} qubits."
            )

        try:
            dest_circuit.add_register(*self.msmt_qc.cregs)
        except CircuitError as exc:
            raise CircuitError(
                f"{exc}\nNote: the supplied quantum circuit should not have a classical register"
                " which has the same name as the classical register that the POVM"
                " implementation uses to store measurement outcomes (creg name: "
                f"'{self._classical_register_name}').\nTo fix it, either delete this register or "
                " change the name of the register of the supplied circuit or of the"
                " POVM implementation."
            ) from exc

        # Compose the two circuits with the correct routing.
        ret = dest_circuit.compose(self.msmt_qc, qubits=index_layout, clbits=self.msmt_qc.clbits)

        t2 = time.time()
        LOGGER.info(f"Finished circuit composition. Took {t2 - t1:.6f}s")

        return ret

    @abstractmethod
    def reshape_data_bin(self, data: DataBin) -> DataBin:
        """TODO."""

    def _get_bitarray(self, data: DataBin) -> BitArray:
        """TODO."""
        return getattr(data, self._classical_register_name)

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

    @property
    def kwargs(self) -> dict[str, Any]:
        """Return the attributes of ``self`` needed to build a copy of ``self``."""
        kwargs = {"n_qubit": self.n_qubit, "measurement_layout": self.measurement_layout}
        return kwargs
