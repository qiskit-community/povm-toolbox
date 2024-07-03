# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the POVMMetadata class."""

from unittest import TestCase

from povm_toolbox.library import ClassicalShadows
from povm_toolbox.library.metadata import POVMMetadata
from qiskit import QuantumCircuit


class TestPOVMMetadata(TestCase):
    def test_init(self):
        """Test the POVMMetadata class initialization."""
        num_qubits = 2

        qc = QuantumCircuit(num_qubits)
        qc.h(0)

        measurement = ClassicalShadows(num_qubits=num_qubits)
        qc_composed = measurement.compose_circuits(qc)

        povm_metadata = POVMMetadata(measurement, qc_composed)
        self.assertIs(povm_metadata.povm_implementation, measurement)
        self.assertIs(povm_metadata.composed_circuit, qc_composed)

        povm_metadata = POVMMetadata(composed_circuit=qc_composed, povm_implementation=measurement)
        self.assertIs(povm_metadata.povm_implementation, measurement)
        self.assertIs(povm_metadata.composed_circuit, qc_composed)

    def test_repr(self):
        """Test that the ``__repr__`` method works correctly."""
        num_qubits = 2

        qc = QuantumCircuit(num_qubits)
        qc.h(0)

        measurement = ClassicalShadows(num_qubits=num_qubits)
        qc_composed = measurement.compose_circuits(qc)

        povm_metadata = POVMMetadata(measurement, qc_composed)
        self.assertEqual(
            f"{povm_metadata}",
            "POVMMetadata(povm_implementation=ClassicalShadows(num_qubits=2), composed_circuit="
            f"<qiskit.circuit.quantumcircuit.QuantumCircuit object at {hex(id(qc_composed))}>)",
        )
