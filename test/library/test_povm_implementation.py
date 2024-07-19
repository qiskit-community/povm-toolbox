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

from povm_toolbox.library import ClassicalShadows
from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.converters import circuit_to_dag
from qiskit.primitives import StatevectorSampler
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ApplyLayout, RemoveBarriers, SetLayout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


class TestPOVMImplementation(TestCase):
    SEED = 42

    def setUp(self) -> None:
        self.num_qubits = 3
        self.povm = ClassicalShadows(num_qubits=self.num_qubits)

        # 3-qubit circuit preparing state |+01>
        self.circuit = QuantumCircuit(self.num_qubits)
        self.circuit.h(0)
        self.circuit.x(2)

        self.composed_circuit = self.povm.compose_circuits(self.circuit)

    def test_appended_classical_registers(self):
        """Test that the classical registers are correctly composed."""

        with self.subTest("Test removing final measurements."):
            qc2 = self.circuit.copy()
            qc2.measure_all()
            composed_qc2 = self.povm.compose_circuits(qc2)
            self.assertEqual(composed_qc2.num_clbits, self.num_qubits)
            self.assertEqual(self.composed_circuit.clbits, composed_qc2.clbits)

            qc2 = self.circuit.copy()
            qc2.measure_active()
            composed_qc2 = self.povm.compose_circuits(qc2)
            self.assertEqual(composed_qc2.num_clbits, self.num_qubits)
            self.assertEqual(self.composed_circuit.clbits, composed_qc2.clbits)

        with self.subTest("Test adding classical register."):
            qc2 = self.circuit.copy()
            qc2.add_register(ClassicalRegister(5, "creg"))
            composed_qc2 = self.povm.compose_circuits(qc2)
            self.assertEqual(composed_qc2.num_clbits, 5 + self.num_qubits)

            qc2 = self.circuit.copy()
            cr = ClassicalRegister(3, "creg")
            qc2.add_register(cr)
            qc2.measure(list(range(self.num_qubits)), cr)
            composed_qc2 = self.povm.compose_circuits(qc2)
            self.assertEqual(composed_qc2.num_clbits, self.num_qubits)

    def test_errors_raised(self):
        """Test that the proper errors are raised in specific situations."""

        with self.subTest(
            "Error when adding already existing classical register."
        ) and self.assertRaises(CircuitError):
            qc2 = self.circuit.copy()
            qc2.add_register(ClassicalRegister(self.num_qubits, self.povm.classical_register_name))
            self.povm.compose_circuits(qc2)

        with self.subTest("Error when number of qubits is not matching."):
            with self.assertRaises(ValueError):
                povm2 = ClassicalShadows(num_qubits=2)
                povm2.compose_circuits(self.circuit)
            with self.assertRaises(ValueError):
                povm2 = ClassicalShadows(num_qubits=4)
                povm2.compose_circuits(self.circuit)

    def test_composed_circuits(self):
        """Test the composition of the input circuit with the measurement circuit."""

        sampler = StatevectorSampler(seed=self.SEED)

        with self.subTest("Composed circuit."):
            pvm = ClassicalShadows(3, seed=self.SEED)

            # compose circuits
            composed_circuit = pvm.compose_circuits(self.circuit)

            # obtain measurement parameter values
            pvm_idx = pvm._sample_pvm_idxs(BindingsArray(), shots=128)
            parameter_values = pvm._get_pvm_bindings_array(pvm_idx)

            # sample composed circuit
            job = sampler.run([(composed_circuit, parameter_values)], shots=1)
            result = pvm._get_bitarray(job.result()[0].data)

            # validate outcome
            expected = {"101": 28, "110": 66, "111": 17, "100": 17}
            self.assertEqual(result.get_counts(), expected)

        with self.subTest("Composed after transpilation of input circuit."):
            pm = generate_preset_pass_manager(optimization_level=0, initial_layout=[2, 0, 1])
            routed_circuit = pm.run(self.circuit)

            pvm = ClassicalShadows(3, seed=self.SEED)

            # compose circuits
            composed_circuit = pvm.compose_circuits(routed_circuit)

            # obtain measurement parameter values
            pvm_idx = pvm._sample_pvm_idxs(BindingsArray(), shots=128)
            parameter_values = pvm._get_pvm_bindings_array(pvm_idx)

            # sample composed circuit
            job = sampler.run([(composed_circuit, parameter_values)], shots=1)
            result = pvm._get_bitarray(job.result()[0].data)

            # validate outcome
            expected = {"101": 68, "110": 26, "111": 17, "100": 17}
            self.assertEqual(result.get_counts(), expected)

        with self.subTest("With a TranspileLayout present"):
            layout = [2, 0, 1]
            pm = PassManager(
                [
                    SetLayout(layout),
                    ApplyLayout(),
                ]
            )
            layout_circuit = pm.run(self.circuit)

            pvm = ClassicalShadows(3, seed=self.SEED)

            # compose circuits
            composed_circuit = pvm.compose_circuits(layout_circuit)

            # obtain measurement parameter values
            pvm_idx = pvm._sample_pvm_idxs(BindingsArray(), shots=128)
            parameter_values = pvm._get_pvm_bindings_array(pvm_idx)

            # sample composed circuit
            job = sampler.run([(composed_circuit, parameter_values)], shots=1)
            result = pvm._get_bitarray(job.result()[0].data)

            # validate outcome
            expected = {"101": 68, "110": 26, "111": 17, "100": 17}
            self.assertEqual(result.get_counts(), expected)

        with self.subTest("Using measurement_layout"):
            measurement_layout = [0, 2]
            pvm = ClassicalShadows(2, seed=self.SEED, measurement_layout=measurement_layout)

            # compose circuits
            composed_circuit = pvm.compose_circuits(self.circuit)

            # due to our chosen measurement_layout, we know that for this circuit, qubit at index 1
            # remains idle
            dag = circuit_to_dag(composed_circuit)
            self.assertEqual(list(dag.idle_wires()), [composed_circuit.qubits[1]])

            # obtain measurement parameter values
            pvm_idx = pvm._sample_pvm_idxs(BindingsArray(), shots=128)
            parameter_values = pvm._get_pvm_bindings_array(pvm_idx)

            # sample composed circuit
            job = sampler.run([(composed_circuit, parameter_values)], shots=1)
            result = pvm._get_bitarray(job.result()[0].data)

            # validate outcome
            expected = {"10": 43, "11": 85}
            self.assertEqual(result.get_counts(), expected)

        with self.subTest("Test the insert_barriers option"):
            pvm = ClassicalShadows(3, seed=self.SEED)
            composed_circuit = pvm.compose_circuits(self.circuit)
            composed_circuit.assign_parameters([0, 0, 0, 0, 0, 0], inplace=True)

            pvm_barriers = ClassicalShadows(3, seed=self.SEED, insert_barriers=True)
            composed_circuit_with_barrier = pvm_barriers.compose_circuits(self.circuit)
            composed_circuit_with_barrier.assign_parameters([0, 0, 0, 0, 0, 0], inplace=True)

            pm = PassManager([RemoveBarriers()])

            self.assertEqual(composed_circuit, pm.run(composed_circuit_with_barrier))
