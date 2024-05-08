# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the RandomizedProjectiveMeasurements class."""

from unittest import TestCase

import numpy as np
from povm_toolbox.library import ClassicalShadows, RandomizedProjectiveMeasurements
from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


class TestPOVMImplementation(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.n_qubits = 3
        self.povm = ClassicalShadows(n_qubit=self.n_qubits)

        # 3-qubit circuit preparing state |+01>
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.h(0)
        self.circuit.x(2)

        self.composed_circuit = self.povm.compose_circuits(self.circuit)

    def test_appended_classical_registers(self):
        """Test that the classical registers are correctly composed."""

        with self.subTest("Test removing final measurements."):
            qc2 = self.circuit.copy()
            qc2.measure_all()
            composed_qc2 = self.povm.compose_circuits(qc2)
            self.assertEqual(composed_qc2.num_clbits, self.n_qubits)
            self.assertEqual(self.composed_circuit.clbits, composed_qc2.clbits)

            qc2 = self.circuit.copy()
            qc2.measure_active()
            composed_qc2 = self.povm.compose_circuits(qc2)
            self.assertEqual(composed_qc2.num_clbits, self.n_qubits)
            self.assertEqual(self.composed_circuit.clbits, composed_qc2.clbits)

        with self.subTest("Test adding classical register."):
            qc2 = self.circuit.copy()
            qc2.add_register(ClassicalRegister(5, "creg"))
            composed_qc2 = self.povm.compose_circuits(qc2)
            self.assertEqual(composed_qc2.num_clbits, 5 + self.n_qubits)

            qc2 = self.circuit.copy()
            cr = ClassicalRegister(3, "creg")
            qc2.add_register(cr)
            qc2.measure(list(range(self.n_qubits)), cr)
            composed_qc2 = self.povm.compose_circuits(qc2)
            self.assertEqual(composed_qc2.num_clbits, self.n_qubits)

    def test_errors_raised(self):
        """Test that the proper errors are raised in specific situations."""

        with self.subTest(
            "Error when adding already existing classical register."
        ) and self.assertRaises(CircuitError):
            qc2 = self.circuit.copy()
            qc2.add_register(ClassicalRegister(self.n_qubits, self.povm.classical_register_name))
            self.povm.compose_circuits(qc2)

        with self.subTest("Error when number of qubits is not matching."):
            with self.assertRaises(ValueError):
                povm2 = ClassicalShadows(n_qubit=2)
                povm2.compose_circuits(self.circuit)
            with self.assertRaises(ValueError):
                povm2 = ClassicalShadows(n_qubit=4)
                povm2.compose_circuits(self.circuit)

    def test_composed_circuits(self):
        """Test the composition of the input circuit with the measurement circuit."""

        with self.subTest("Composed circuit."):
            # define a ZZX-measurement (inverse qubit order)
            bias = np.array([1.0])
            angles = np.array([[0.5 * np.pi, 0.0], [0.0, 0.0], [0.0, 0.0]])
            pvm = RandomizedProjectiveMeasurements(n_qubit=3, bias=bias, angles=angles)

            # compose circuits
            composed_circuit = pvm.compose_circuits(self.circuit)
            parameter_values = pvm._get_pvm_bindings_array(np.array([[0, 0, 0]]))
            sampler = StatevectorSampler()
            job = sampler.run([(composed_circuit, parameter_values)])

            # assert the final state is |+01>
            self.assertTrue(
                set(pvm._get_bitarray(job.result()[0].data).get_counts().keys()) == {"100"}
            )

        with self.subTest("Composed after transpilation of input circuit."):
            pm = generate_preset_pass_manager(optimization_level=0, initial_layout=[2, 0, 1])
            routed_circuit = pm.run(self.circuit)

            # define a ZZX-measurement (inverse qubit order)
            bias = np.array([1.0])
            angles = np.array([[0.5 * np.pi, 0.0], [0.0, 0.0], [0.0, 0.0]])
            pvm = RandomizedProjectiveMeasurements(n_qubit=3, bias=bias, angles=angles)

            # compose circuits after the input circuit has been transpiled
            composed_circuit = pvm.compose_circuits(routed_circuit)
            parameter_values = pvm._get_pvm_bindings_array(np.array([[0, 0, 0]]))
            sampler = StatevectorSampler()
            job = sampler.run([(composed_circuit, parameter_values)])

            # assert the final state is |+01>
            self.assertTrue(
                set(pvm._get_bitarray(job.result()[0].data).get_counts().keys()) == {"100"}
            )
