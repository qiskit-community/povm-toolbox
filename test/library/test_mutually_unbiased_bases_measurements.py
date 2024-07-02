# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the MutuallyUnbiasedBasesMeasurements class."""

from unittest import TestCase

import numpy as np
from numpy.random import default_rng
from povm_toolbox.library import MutuallyUnbiasedBasesMeasurements
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.sampler import POVMSampler
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, SGate, UGate
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector


class TestMutuallyUnbiasedBasesMeasurements(TestCase):
    def test_init(self):
        """Test the implementation of mutually-unbiased-bases POVMs."""
        rng = default_rng(13)

        qc = QuantumCircuit(2)
        qc.h(0)

        num_qubits = qc.num_qubits

        with self.subTest("Test uniform angles across qubits."):
            measurement = MutuallyUnbiasedBasesMeasurements(
                num_qubits,
                bias=np.ones(3) / 3,
                angles=np.array([0.75, -np.pi / 3, 0.2]),
                seed_rng=rng,
            )
            sampler = StatevectorSampler(seed=rng)
            povm_sampler = POVMSampler(sampler=sampler)

            job = povm_sampler.run([qc], shots=128, povm=measurement)
            pub_result = job.result()[0]

            post_processor = POVMPostProcessor(pub_result)

            observable = SparsePauliOp(["ZI"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, 1.0162351045000342)
            self.assertAlmostEqual(std, 0.1292712036194854)
            observable = SparsePauliOp(["ZY"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, 0.1504502666075513)
            self.assertAlmostEqual(std, 0.27055245411115403)

        with self.subTest("Test specific angles for each qubit."):
            measurement = MutuallyUnbiasedBasesMeasurements(
                num_qubits,
                bias=np.ones(3) / 3,
                angles=np.array([[1.2, 0.0, 0.4], [3.5, -0.4, 0.8]]),
                seed_rng=rng,
            )
            sampler = StatevectorSampler(seed=rng)
            povm_sampler = POVMSampler(sampler=sampler)

            job = povm_sampler.run([qc], shots=128, povm=measurement)
            pub_result = job.result()[0]

            post_processor = POVMPostProcessor(pub_result)

            observable = SparsePauliOp(["ZI"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, 0.8341406005724791)
            self.assertAlmostEqual(std, 0.1284174176451904)
            observable = SparsePauliOp(["ZY"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, -0.27772649739166505)
            self.assertAlmostEqual(std, 0.24894801839627123)

    def test_init_errors(self):
        """Test that the ``__init__`` method raises errors correctly."""
        with self.subTest("Test invalid shape for ``bias``.") and self.assertRaises(ValueError):
            MutuallyUnbiasedBasesMeasurements(1, bias=np.ones(2) / 2, angles=np.zeros(3))
        with self.subTest("Test invalid shape for ``angles``.") and self.assertRaises(ValueError):
            MutuallyUnbiasedBasesMeasurements(1, bias=np.ones(3) / 3, angles=np.zeros(2))

    def test_repr(self):
        """Test that the ``__repr__`` method works correctly."""
        mub_str = "MutuallyUnbiasedBasesMeasurements(num_qubits=1, bias=array([[0.2, 0.3, 0.5]]), angles=array([0, 1, 2]))"
        povm = MutuallyUnbiasedBasesMeasurements(
            1, bias=np.array([0.2, 0.3, 0.5]), angles=np.arange(3)
        )
        self.assertEqual(povm.__repr__(), mub_str)

    def test_process_angles(self):
        """Test if the processing of the angles works correctly."""
        rng = default_rng()

        set_angles = 2 * np.pi * (rng.random(12).reshape((4, 3)) - 0.5)

        for angles in set_angles:
            theta, phi, lam = angles

            processed_angles_1 = MutuallyUnbiasedBasesMeasurements._process_angles(theta, phi, lam)

            H = HGate().to_matrix()
            S = SGate().to_matrix()
            U = UGate(theta, phi, lam).to_matrix()

            rotated_Z_msmt = U @ np.array([1, 0])
            rotated_X_msmt = U @ H @ np.array([1, 0])
            rotated_Y_msmt = U @ S @ H @ np.array([1, 0])
            rotated_msmts = [rotated_Z_msmt, rotated_X_msmt, rotated_Y_msmt]

            bloch_vectors = np.real_if_close(
                [
                    [
                        Statevector(msmt).expectation_value(Operator.from_label(op))
                        for op in ["X", "Y", "Z"]
                    ]
                    for msmt in rotated_msmts
                ]
            )

            thetas = np.arctan2(np.linalg.norm(bloch_vectors[:, :2], axis=1), bloch_vectors[:, 2])
            phis = np.arctan2(bloch_vectors[:, 1], bloch_vectors[:, 0])
            processed_angles_2 = np.vstack((thetas, phis)).T.flatten()

            self.assertTrue(np.allclose(processed_angles_1, processed_angles_2))
