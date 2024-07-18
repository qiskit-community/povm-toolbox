# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the MedianOfMeans class."""

from unittest import TestCase

from numpy.random import default_rng
from povm_toolbox.library import ClassicalShadows
from povm_toolbox.post_processor import MedianOfMeans
from povm_toolbox.sampler import POVMSampler
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import SparsePauliOp


class TestMedianOfMeans(TestCase):
    """Test the methods and attributes of the :class:`.MedianOfMeans class`."""

    SEED = 3433

    def setUp(self) -> None:
        super().setUp()

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        povm_sampler = POVMSampler(sampler=Sampler(seed=self.SEED))
        self.measurement = ClassicalShadows(num_qubits=2, seed=self.SEED)

        job = povm_sampler.run([qc], shots=32, povm=self.measurement)
        result = job.result()
        self.pub_result = result[0]

    def test_init_errors(self):
        """Test that the ``__init__`` method raises errors correctly."""
        # sanity check
        with self.subTest("Valid initialization."):
            post_processor = MedianOfMeans(
                self.pub_result, num_batches=5, seed=default_rng(self.SEED)
            )
            self.assertEqual(post_processor.num_batches, 5)
            self.assertEqual(post_processor.delta_confidence, 0.1641699972477976)
        with self.subTest("Invalid type for ``seed``.") and self.assertRaises(TypeError):
            MedianOfMeans(self.pub_result, seed=1.2)

    def test_get_expectation_value(self):
        """Test that the ``get_expectation_value`` method works correctly."""
        with self.subTest("Test with default ``delta_confidence`` argument."):
            post_processor = MedianOfMeans(self.pub_result, seed=self.SEED)
            observable = SparsePauliOp(["ZZ", "XX", "YY"], coeffs=[1, 2, 3])
            exp_val, epsilon_coef = post_processor.get_expectation_value(observable)
            self.assertEqual(post_processor.num_batches, 8)
            self.assertEqual(post_processor.delta_confidence, 0.03663127777746836)
            self.assertAlmostEqual(exp_val, 1.125)
            self.assertAlmostEqual(epsilon_coef, 2.9154759474226504)

        with self.subTest("Test with specified ``delta_confidence`` argument."):
            post_processor = MedianOfMeans(
                self.pub_result, upper_delta_confidence=0.1, seed=self.SEED
            )
            observable = SparsePauliOp(["ZZ", "XX", "YY"], coeffs=[1, 2, 3])
            exp_val, epsilon_coef = post_processor.get_expectation_value(observable)
            self.assertEqual(post_processor.num_batches, 6)
            self.assertEqual(post_processor.delta_confidence, 0.09957413673572789)
            self.assertAlmostEqual(exp_val, -0.7500000000000003)
            self.assertAlmostEqual(epsilon_coef, 2.6076809620810595)

        with self.subTest("Test with specified ``num_batches`` argument."):
            post_processor = MedianOfMeans(self.pub_result, num_batches=4, seed=self.SEED)
            observable = SparsePauliOp(["ZZ", "XX", "YY"], coeffs=[1, 2, 3])
            exp_val, epsilon_coef = post_processor.get_expectation_value(observable)
            self.assertEqual(post_processor.num_batches, 4)
            self.assertEqual(post_processor.delta_confidence, 0.2706705664732254)
            self.assertAlmostEqual(exp_val, -4.440892098500626e-16)
            self.assertAlmostEqual(epsilon_coef, 2.0615528128088303)

        with self.subTest("Test with random ``seed`` argument."):
            post_processor = MedianOfMeans(self.pub_result, num_batches=8)
            self.assertEqual(post_processor.num_batches, 8)
            self.assertEqual(post_processor.delta_confidence, 0.03663127777746836)
            observable = SparsePauliOp(["ZZ", "XX", "YY"], coeffs=[1, 2, 3])
            _, epsilon_coef = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(epsilon_coef, 2.9154759474226504)

    def test_delta_confidence(self):
        """Test that the ``delta_confidence`` property and setter work correctly."""
        post_processor = MedianOfMeans(self.pub_result, num_batches=5)
        self.assertEqual(post_processor.num_batches, 5)
        self.assertEqual(post_processor.delta_confidence, 0.1641699972477976)
        post_processor.delta_confidence = 0.098
        self.assertEqual(post_processor.num_batches, 7)
        self.assertEqual(post_processor.delta_confidence, 0.060394766844637)
