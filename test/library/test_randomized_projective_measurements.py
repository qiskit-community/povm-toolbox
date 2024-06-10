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
from numpy.random import default_rng
from povm_toolbox.library import ClassicalShadows, RandomizedProjectiveMeasurements
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.sampler import POVMSampler
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp


class TestRandomizedPMs(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

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

    def test_qc_build(self):
        """Test if we can build a QuantumCircuit."""

        for n_qubit in range(1, 11):
            q = np.random.uniform(0, 5, size=3 * n_qubit).reshape((n_qubit, 3))
            q /= q.sum(axis=1)[:, np.newaxis]

            angles = np.array([0.0, 0.0, 0.5 * np.pi, 0.0, 0.5 * np.pi, 0.5 * np.pi])

            cs_implementation = RandomizedProjectiveMeasurements(
                n_qubit=n_qubit, bias=q, angles=angles
            )

            qc = cs_implementation._build_qc()

            self.assertEqual(qc.num_qubits, n_qubit)

    def test_twirling(self):
        """Test if the twirling option works correctly."""
        rng = default_rng(13)

        qc = QuantumCircuit(2)
        qc.h(0)

        n_qubit = qc.num_qubits
        measurement = ClassicalShadows(n_qubit, seed_rng=rng, measurement_twirl=True)

        rng2 = default_rng(26)

        sampler = StatevectorSampler(seed=rng2)
        povm_sampler = POVMSampler(sampler=sampler)

        job = povm_sampler.run([qc], shots=128, povm=measurement)
        pub_result = job.result()[0]

        post_processor = POVMPostProcessor(pub_result)

        observable = SparsePauliOp(["ZI"], coeffs=[1.0])
        exp_value, _ = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 0.9843749999999998)
        observable = SparsePauliOp(["IZ"], coeffs=[1.0])
        exp_value, _ = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 0.07031249999999983)
        observable = SparsePauliOp(["ZY"], coeffs=[1.0])
        exp_value, _ = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 0.0)
        observable = SparsePauliOp(["IX"], coeffs=[1.0])
        exp_value, _ = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 1.1718749999999998)

    def test_shot_repetitions(self):
        """Test if the twirling option works correctly."""
        rng = default_rng(65)

        qc = QuantumCircuit(2)
        qc.h(0)

        n_qubit = qc.num_qubits
        measurement = ClassicalShadows(n_qubit, seed_rng=rng, shot_repetitions=7)

        rng2 = default_rng(56)

        sampler = StatevectorSampler(seed=rng2)
        povm_sampler = POVMSampler(sampler=sampler)

        job = povm_sampler.run([qc], shots=128, povm=measurement)
        pub_result = job.result()[0]

        self.assertEqual(sum(pub_result.get_counts()[0].values()), 128 * 7)
        self.assertEqual(pub_result.data.povm_measurement_creg.num_shots, 128 * 7)

        post_processor = POVMPostProcessor(pub_result)

        observable = SparsePauliOp(["ZI"], coeffs=[1.0])
        exp_value, _ = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 1.0312499999999998)

    # TODO: write a unittest for each public method of RandomizedProjectiveMeasurements

    # TODO: write a unittest to assert the correct handling of invalid inputs (i.e. verify that
    # errors are raised properly)
