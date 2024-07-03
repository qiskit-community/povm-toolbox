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
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.sampler import POVMSampler
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp


class TestRandomizedPMs(TestCase):
    RNG_SEED = 239486

    def setUp(self) -> None:
        super().setUp()

        basis_0 = np.array([1.0, 0], dtype=complex)
        basis_1 = np.array([0, 1.0], dtype=complex)
        basis_plus = 1.0 / np.sqrt(2) * (basis_0 + basis_1)
        basis_minus = 1.0 / np.sqrt(2) * (basis_0 - basis_1)
        basis_plus_i = 1.0 / np.sqrt(2) * (basis_0 + 1.0j * basis_1)
        basis_minus_i = 1.0 / np.sqrt(2) * (basis_0 - 1.0j * basis_1)

        self.Z0 = np.outer(basis_0, basis_0.conj())
        self.Z1 = np.outer(basis_1, basis_1.conj())
        self.X0 = np.outer(basis_plus, basis_plus.conj())
        self.X1 = np.outer(basis_minus, basis_minus.conj())
        self.Y0 = np.outer(basis_plus_i, basis_plus_i.conj())
        self.Y1 = np.outer(basis_minus_i, basis_minus_i.conj())

    def test_init_errors(self):
        """Test that the ``__init__`` method raises errors correctly."""
        # Sanity check
        measurement = RandomizedProjectiveMeasurements(
            1, bias=np.array([0.5, 0.5]), angles=np.array([0.0, 0.0, 0.5, 0.0])
        )
        self.assertIsInstance(measurement, RandomizedProjectiveMeasurements)
        with self.subTest("Incompatible ``bias`` and ``angles`` shapes.") and self.assertRaises(
            ValueError
        ):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([0.5, 0.5]), angles=np.array([0.0, 0.0, 0.5, 0.0, 0.4])
            )
        with self.subTest(
            "Shape of ``bias`` incompatible with number of qubits."
        ) and self.assertRaises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([[0.5, 0.5], [0.5, 0.5]]), angles=np.array([0.0, 0.0, 0.5, 0.0])
            )
        with self.subTest("Too many dims in ``bias``.") and self.assertRaises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([[[0.5, 0.5]]]), angles=np.array([0.0, 0.0, 0.5, 0.0])
            )
        with self.subTest("Negative value in ``bias``.") and self.assertRaises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([1.5, -0.5]), angles=np.array([0.0, 0.0, 0.5, 0.0])
            )
        with self.subTest("``bias`` not summing up to one.") and self.assertRaises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([0.5, 0.4]), angles=np.array([0.0, 0.0, 0.5, 0.0])
            )
        with self.subTest("``bias`` not summing up to one.") and self.assertRaises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([[0.5, 0.4], [0.5, 0.6]]), angles=np.array([0.0, 0.0, 0.5, 0.0])
            )
        with self.subTest(
            "Shape of ``angles`` incompatible with number of qubits."
        ) and self.assertRaises(ValueError):
            RandomizedProjectiveMeasurements(
                1,
                bias=np.array([0.5, 0.5]),
                angles=np.array([[0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.5, 0.0]]),
            )
        with self.subTest("Too many dims in ``angles``.") and self.assertRaises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([0.5, 0.5]), angles=np.array([[[0.0, 0.0, 0.5, 0.0]]])
            )
        with self.subTest("Invalid type for ``seed_rng``.") and self.assertRaises(TypeError):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([0.5, 0.5]), angles=np.array([0.0, 0.0, 0.5, 0.0]), seed_rng=1.2
            )

    def test_qc_build(self):
        """Test if we can build a QuantumCircuit."""
        for num_qubits in range(1, 11):
            q = np.random.uniform(0, 5, size=3 * num_qubits).reshape((num_qubits, 3))
            q /= q.sum(axis=1)[:, np.newaxis]

            angles = np.array([0.0, 0.0, 0.5 * np.pi, 0.0, 0.5 * np.pi, 0.5 * np.pi])

            cs_implementation = RandomizedProjectiveMeasurements(
                num_qubits=num_qubits, bias=q, angles=angles
            )

            qc = cs_implementation._build_qc()

            self.assertEqual(qc.num_qubits, num_qubits)

    def test_to_sampler_pub(self):
        """Test that the ``to_sampler_pub`` method works correctly."""
        _ = RandomizedProjectiveMeasurements(
            1,
            bias=np.array([0.3, 0.7]),
            angles=np.array([0.0, 0.0, 0.5, 0.0]),
            seed_rng=self.RNG_SEED,
        )
        qc = QuantumCircuit(2)
        qc.h(0)

        # TODO

    def test_twirling(self):
        """Test if the twirling option works correctly."""
        qc = QuantumCircuit(2)
        qc.h(0)

        num_qubits = qc.num_qubits
        measurement = ClassicalShadows(num_qubits, seed_rng=self.RNG_SEED, measurement_twirl=True)

        povm_sampler = POVMSampler(sampler=StatevectorSampler(seed=self.RNG_SEED))

        job = povm_sampler.run([qc], shots=128, povm=measurement)
        pub_result = job.result()[0]

        post_processor = POVMPostProcessor(pub_result)

        observable = SparsePauliOp(["ZI"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 1.0078125000000002)
        self.assertAlmostEqual(std, 0.1257341109337995)
        observable = SparsePauliOp(["IZ"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 0.32812500000000006)
        self.assertAlmostEqual(std, 0.15333777198692233)
        observable = SparsePauliOp(["ZY"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, -0.42187500000000006)
        self.assertAlmostEqual(std, 0.24164416287543897)
        observable = SparsePauliOp(["IX"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 1.1718749999999998)
        self.assertAlmostEqual(std, 0.12987983496490826)

    def test_shot_repetitions(self):
        """Test if the twirling option works correctly."""
        qc = QuantumCircuit(2)
        qc.h(0)

        num_qubits = qc.num_qubits
        measurement = ClassicalShadows(num_qubits, seed_rng=self.RNG_SEED, shot_repetitions=7)

        povm_sampler = POVMSampler(sampler=StatevectorSampler(seed=self.RNG_SEED))

        job = povm_sampler.run([qc], shots=128, povm=measurement)
        pub_result = job.result()[0]

        self.assertEqual(sum(pub_result.get_counts()[0].values()), 128 * 7)
        self.assertEqual(pub_result.data.povm_measurement_creg.num_shots, 128 * 7)

        post_processor = POVMPostProcessor(pub_result)

        observable = SparsePauliOp(["ZI"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 0.9843750000000002)
        self.assertAlmostEqual(std, 0.04708403113719653)

    # TODO: write a unittest for each public method of RandomizedProjectiveMeasurements

    # TODO: write a unittest to assert the correct handling of invalid inputs (i.e. verify that
    # errors are raised properly)

    def test_error_private_methods(self):
        """Test that errors in private methods are raised correctly."""
        measurement = RandomizedProjectiveMeasurements(
            2,
            bias=np.array([0.3, 0.4, 0.3]),
            angles=np.array([0.0, 0.0, 0.5, 0.0, 0.5, 0.5]),
            seed_rng=self.RNG_SEED,
        )
        qc = QuantumCircuit(2)
        qc.h(0)
        povm_sampler = POVMSampler(sampler=StatevectorSampler(seed=self.RNG_SEED))
        job = povm_sampler.run([qc], shots=128, povm=measurement)
        _ = job.result()[0]

        # TODO
