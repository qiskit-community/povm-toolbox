# (C) Copyright IBM 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the POVMMetadata class."""

import platform

from povm_toolbox.library import ClassicalShadows
from povm_toolbox.library.metadata import POVMMetadata
from qiskit import QuantumCircuit


class TestPOVMMetadata:
    def test_init(self):
        """Test the POVMMetadata class initialization."""
        num_qubits = 2

        qc = QuantumCircuit(num_qubits)
        qc.h(0)

        measurement = ClassicalShadows(num_qubits=num_qubits)
        qc_composed = measurement.compose_circuits(qc)

        povm_metadata = POVMMetadata(measurement, qc_composed)
        assert povm_metadata.povm_implementation is measurement
        assert povm_metadata.composed_circuit is qc_composed

        povm_metadata = POVMMetadata(composed_circuit=qc_composed, povm_implementation=measurement)
        assert povm_metadata.povm_implementation is measurement
        assert povm_metadata.composed_circuit is qc_composed

    def test_repr(self):
        """Test that the ``__repr__`` method works correctly."""
        num_qubits = 2

        qc = QuantumCircuit(num_qubits)
        qc.h(0)

        measurement = ClassicalShadows(num_qubits=num_qubits)
        qc_composed = measurement.compose_circuits(qc)
        qc_id = id(qc_composed)
        encoding = "016X" if platform.system() == "Windows" else "x"
        qc_hex_id = f"0x{qc_id:{encoding}}"

        povm_metadata = POVMMetadata(measurement, qc_composed)
        assert f"{povm_metadata}" == (
            "POVMMetadata(povm_implementation=ClassicalShadows(num_qubits=2), composed_circuit="
            f"<qiskit.circuit.quantumcircuit.QuantumCircuit object at {qc_hex_id}>)"
        )
