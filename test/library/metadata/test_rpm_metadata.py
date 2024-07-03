# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the RPMMetadata class."""

from unittest import TestCase

import numpy as np
from povm_toolbox.library import ClassicalShadows
from povm_toolbox.library.metadata import RPMMetadata
from qiskit import QuantumCircuit


class TestRPMMetadata(TestCase):
    def test_init(self):
        """Test the POVMMetadata class initialization."""
        num_qubits = 2

        qc = QuantumCircuit(num_qubits)
        qc.h(0)

        measurement = ClassicalShadows(num_qubits=num_qubits)
        qc_composed = measurement.compose_circuits(qc)

        shape = (4, 5, 3, num_qubits)
        pvm_keys = np.arange(np.prod(shape)).reshape(shape)

        rpm_metadata = RPMMetadata(measurement, qc_composed, pvm_keys)
        self.assertIs(rpm_metadata.povm_implementation, measurement)
        self.assertIs(rpm_metadata.composed_circuit, qc_composed)
        self.assertIs(rpm_metadata.pvm_keys, pvm_keys)

        rpm_metadata = RPMMetadata(
            composed_circuit=qc_composed, pvm_keys=pvm_keys, povm_implementation=measurement
        )
        self.assertIs(rpm_metadata.povm_implementation, measurement)
        self.assertIs(rpm_metadata.composed_circuit, qc_composed)
        self.assertIs(rpm_metadata.pvm_keys, pvm_keys)

    def test_repr(self):
        """Test that the ``__repr__`` method works correctly."""
        num_qubits = 2

        qc = QuantumCircuit(num_qubits)
        qc.h(0)

        measurement = ClassicalShadows(num_qubits=num_qubits)
        qc_composed = measurement.compose_circuits(qc)

        shape = (4, 5, 3, num_qubits)
        pvm_keys = np.arange(np.prod(shape)).reshape(shape)

        rpm_metadata = RPMMetadata(measurement, qc_composed, pvm_keys)
        self.assertEqual(
            f"{rpm_metadata}",
            "RPMMetadata(povm_implementation=ClassicalShadows(num_qubits=2), composed_circuit="
            f"<qiskit.circuit.quantumcircuit.QuantumCircuit object at {hex(id(qc_composed))}>,"
            f" pvm_keys=np.ndarray<4,5,3,{num_qubits}>)",
        )
