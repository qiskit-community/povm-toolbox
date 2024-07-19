# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the DilationMeasurements class."""

from unittest import TestCase

import numpy as np
from povm_toolbox.library import (
    DilationMeasurements,
)
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.sampler import POVMSampler
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2


class TestDilationMeasurements(TestCase):
    SEED = 9128346

    def setUp(self) -> None:
        super().setUp()

    def test_init_errors(self):
        """Test that the ``__init__`` method raises errors correctly."""
        # Sanity check
        measurement = DilationMeasurements(1, parameters=np.random.uniform(0, 1, size=8))
        self.assertIsInstance(measurement, DilationMeasurements)
        with self.subTest(
            "Test invalid shape for ``parameters``, not enough parameters."
        ) and self.assertRaises(ValueError):
            DilationMeasurements(1, parameters=np.ones(7))
        with self.subTest(
            "Test invalid shape for ``parameters``, number of qubits not matching."
        ) and self.assertRaises(ValueError):
            DilationMeasurements(1, parameters=np.ones((2, 8)))
        with self.subTest(
            "Test invalid shape for ``parameters``, too many dimensions."
        ) and self.assertRaises(ValueError):
            DilationMeasurements(1, parameters=np.ones((1, 1, 8)))

    def test_repr(self):
        """Test that the ``__repr__`` method works correctly."""
        mub_str = (
            "DilationMeasurements(num_qubits=1, parameters=array"
            "([[0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]))"
        )
        povm = DilationMeasurements(1, parameters=0.1 * np.arange(8))
        self.assertEqual(povm.__repr__(), mub_str)

    def test_to_sampler_pub(self):
        """Test that the ``to_sampler_pub`` method works correctly."""
        num_qubits = 2
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)

        backend = FakeManilaV2()
        backend.set_options(seed_simulator=self.SEED)
        povm_sampler = POVMSampler(sampler=RuntimeSampler(mode=backend))

        pm = generate_preset_pass_manager(
            optimization_level=2, backend=backend, seed_transpiler=self.SEED
        )

        measurement = DilationMeasurements(
            num_qubits,
            parameters=np.array(
                [
                    0.75,
                    0.30408673,
                    0.375,
                    0.40678524,
                    0.32509973,
                    0.25000035,
                    0.49999321,
                    0.83333313,
                ]
            ),
        )

        job = povm_sampler.run([qc], shots=128, povm=measurement, pass_manager=pm)
        pub_result = job.result()[0]

        post_processor = POVMPostProcessor(pub_result)

        observable = SparsePauliOp(["ZI"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 0.09375100309506113)
        self.assertAlmostEqual(std, 0.15820620317705336)
        observable = SparsePauliOp(["IZ"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 0.06250168079223002)
        self.assertAlmostEqual(std, 0.15676579098034854)
        observable = SparsePauliOp(["XI"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 1.0275137444747091)
        self.assertAlmostEqual(std, 0.1612840231061351)

    def test_definition(self):
        """Test that the ``definition`` method works correctly."""
        num_qubits = 1

        # parameters defining a SIC-POVM
        sic_parameters = np.array(
            [[0.75, 0.30408673, 0.375, 0.40678524, 0.32509973, 0.25000035, 0.49999321, 0.83333313]]
        )
        # define measurement and the quantum-informational POVM
        measurement = DilationMeasurements(num_qubits, parameters=sic_parameters)
        povm = measurement.definition()[(0,)]

        with self.subTest("Test effects"):
            effects = np.empty((4, 2, 2), dtype=complex)
            effects[0] = np.array(
                [
                    [5.00000000e-01 + 0.00000000e00j, 3.25620884e-09 - 4.42295492e-06j],
                    [3.25620884e-09 + 4.42295492e-06j, 3.91250817e-11 - 2.67501659e-30j],
                ]
            )
            effects[1] = np.array(
                [
                    [0.16666667 + 0.00000000e00j, 0.23570227 + 4.41719108e-06j],
                    [0.23570227 - 4.41719108e-06j, 0.33333335 - 9.43454930e-23j],
                ]
            )
            effects[2] = np.array(
                [
                    [0.16666666 + 0.00000000e00j, -0.11785114 - 2.04124135e-01j],
                    [-0.11785114 + 2.04124135e-01j, 0.33333334 - 8.73465760e-18j],
                ]
            )
            effects[3] = np.array(
                [
                    [0.16666667 + 0.00000000e00j, -0.11785113 + 2.04124140e-01j],
                    [-0.11785113 - 2.04124140e-01j, 0.33333331 - 6.84480542e-18j],
                ]
            )
            for effect, povm_operator in zip(effects, povm.operators):
                self.assertTrue(np.allclose(povm_operator.data, effect))

        with self.subTest("Test bloch vectors"):
            bloch_vectors_check = np.array(
                [
                    [
                        0.0,
                        0.0,
                        0.5,
                    ],
                    [0.47140454, 0.0, -0.16666668],
                    [-0.23570228, 0.40824827, -0.16666668],
                    [-0.23570226, -0.40824828, -0.16666664],
                ]
            )
            self.assertTrue(np.allclose(povm.get_bloch_vectors(), bloch_vectors_check))

    def test_compose_circuit(self):
        """Test that the ``compose_circuit`` method works correctly."""
        sampler = StatevectorSampler(seed=self.SEED)
        povm_sampler = POVMSampler(sampler)
        measurement = DilationMeasurements(
            num_qubits=2,
            parameters=np.array(
                [
                    0.75,
                    0.30408673,
                    0.375,
                    0.40678524,
                    0.32509973,
                    0.25000035,
                    0.49999321,
                    0.83333313,
                ]
            ),
        )
        with self.subTest("No idle qubits in input circuit."):
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            job = povm_sampler.run([qc], shots=32, povm=measurement)
            pub_result = job.result()[0]
            self.assertEqual(pub_result.metadata.composed_circuit.num_qubits, 4)
            self.assertEqual(pub_result.metadata.composed_circuit.num_ancillas, 2)
            observable = SparsePauliOp(["XI", "XX", "YY", "ZX"], coeffs=[1, 1, -1, 1])
            post_processor = POVMPostProcessor(pub_result)
            exp_value, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, 2.0517605907028327)
            self.assertAlmostEqual(std, 1.1158789520748584)
        with self.subTest("Not enough idle qubits in input circuit."):
            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 1)
            measurement.measurement_layout = [0, 1]
            job = povm_sampler.run([qc], shots=32, povm=measurement)
            pub_result = job.result()[0]
            self.assertEqual(pub_result.metadata.composed_circuit.num_qubits, 4)
            self.assertEqual(pub_result.metadata.composed_circuit.num_ancillas, 1)
            observable = SparsePauliOp(["XI", "XX", "YY", "ZX"], coeffs=[1, 1, -1, 1])
            post_processor = POVMPostProcessor(pub_result)
            exp_value, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, 2.0517605907028327)
            self.assertAlmostEqual(std, 1.1158789520748584)
        with self.subTest("Exactly enough idle qubits in input circuit."):
            qc = QuantumCircuit(4)
            qc.h(0)
            qc.cx(0, 1)
            measurement.measurement_layout = [0, 1]
            job = povm_sampler.run([qc], shots=32, povm=measurement)
            pub_result = job.result()[0]
            self.assertEqual(pub_result.metadata.composed_circuit.num_qubits, 4)
            self.assertEqual(pub_result.metadata.composed_circuit.num_ancillas, 0)
            observable = SparsePauliOp(["XI", "XX", "YY", "ZX"], coeffs=[1, 1, -1, 1])
            post_processor = POVMPostProcessor(pub_result)
            exp_value, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, 2.0517605907028327)
            self.assertAlmostEqual(std, 1.1158789520748584)
        with self.subTest("Too many idle qubits in input circuit."):
            qc = QuantumCircuit(5)
            qc.h(0)
            qc.cx(0, 1)
            measurement.measurement_layout = [0, 1]
            job = povm_sampler.run([qc], shots=32, povm=measurement)
            pub_result = job.result()[0]
            self.assertEqual(pub_result.metadata.composed_circuit.num_qubits, 5)
            self.assertEqual(pub_result.metadata.composed_circuit.num_ancillas, 0)
            observable = SparsePauliOp(["XI", "XX", "YY", "ZX"], coeffs=[1, 1, -1, 1])
            post_processor = POVMPostProcessor(pub_result)
            exp_value, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, 2.0517605907028327)
            self.assertAlmostEqual(std, 1.1158789520748584)
