# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the POVMPostProcessor class."""

from unittest import TestCase

import numpy as np
from povm_toolbox.library import ClassicalShadows, LocallyBiasedClassicalShadows
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.quantum_info.product_dual import ProductDual
from povm_toolbox.sampler import POVMSampler
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import SparsePauliOp


class TestPostProcessor(TestCase):
    """Test the methods and attributes of the :class:`.POVMPostProcessor class`."""

    SEED = 42

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

    def test_init(self):
        """Test that ``__init__`` works correctly."""
        with self.subTest("Initialization with default ``dual``."):
            post_processor = POVMPostProcessor(self.pub_result)
            self.assertIsInstance(post_processor, POVMPostProcessor)
            self.assertIsNone(post_processor._dual)

        with self.subTest("Initialization with valid ``dual`` argument."):
            povm = self.measurement.definition()
            dual = ProductDual.build_dual_from_frame(povm)
            post_processor = POVMPostProcessor(self.pub_result, dual=dual)
            self.assertIsInstance(post_processor, POVMPostProcessor)
            self.assertIs(post_processor._dual, dual)

        with self.subTest("Initialization with invalid ``dual`` argument.") and self.assertRaises(
            ValueError
        ):
            povm = LocallyBiasedClassicalShadows(
                num_qubits=2, bias=np.array([0.8, 0.1, 0.1])
            ).definition()
            dual = ProductDual.build_dual_from_frame(povm)
            post_processor = POVMPostProcessor(self.pub_result, dual=dual)

    def test_dual(self):
        """Test that the ``dual`` property and setter work correctly."""
        with self.subTest("Test default ``dual``."):
            post_processor = POVMPostProcessor(self.pub_result)
            self.assertIsNone(post_processor._dual)
            self.assertIsInstance(post_processor.dual, ProductDual)
            self.assertIsInstance(post_processor._dual, ProductDual)
        with self.subTest("Test setting ``dual`` after initialization."):
            post_processor = POVMPostProcessor(self.pub_result)
            self.assertIsNone(post_processor._dual)
            povm = self.measurement.definition()
            dual = ProductDual.build_dual_from_frame(povm, alphas=((1, 2, 2, 2, 2, 2), None))
            post_processor.dual = dual
            self.assertIs(post_processor._dual, dual)
            self.assertIs(post_processor.dual, dual)
        with self.subTest(
            "Test setting invalid ``dual`` after initialization."
        ) and self.assertRaises(ValueError):
            post_processor = POVMPostProcessor(self.pub_result)
            povm = LocallyBiasedClassicalShadows(
                num_qubits=2, bias=np.array([0.8, 0.1, 0.1])
            ).definition()
            dual = ProductDual.build_dual_from_frame(povm)
            post_processor.dual = dual

    def test_get_decomposition_weights(self):
        """Test that the ``get_decomposition_weights`` method works correctly."""
        observable = SparsePauliOp(["ZZ", "XX", "YY"], coeffs=[1, 2, 3])
        post_processor = POVMPostProcessor(self.pub_result)
        weights = post_processor.get_decomposition_weights(
            observable, set([(0, 0), (0, 1), (1, 1), (0, 2), (2, 2), (2, 3), (5, 5), (5, 0)])
        )
        self.assertAlmostEqual(weights[(0, 0)], 1 * 9)
        self.assertAlmostEqual(weights[(0, 1)], -1 * 9)
        self.assertAlmostEqual(weights[(1, 1)], 1 * 9)
        self.assertAlmostEqual(weights[(0, 2)], 0)
        self.assertAlmostEqual(weights[(2, 2)], 2 * 9)
        self.assertAlmostEqual(weights[(2, 3)], -2 * 9)
        self.assertAlmostEqual(weights[(5, 5)], 3 * 9)
        self.assertAlmostEqual(weights[(5, 0)], 0)

    def test_get_expectation_value(self):
        """Test that the ``get_expectation_value`` method works correctly."""
        post_processor = POVMPostProcessor(self.pub_result)
        with self.subTest("Test with default ``loc`` for un-parametrized circuit."):
            observable = SparsePauliOp(["ZZ", "XX", "YY"], coeffs=[1, 2, 3])
            exp_val, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_val, -2.2499999999999987)
            self.assertAlmostEqual(std, 2.3563572213988917)
        with self.subTest("Test with specified ``loc`` argument."):
            observable = SparsePauliOp(["IZ", "XX", "ZY"], coeffs=[-0.5, 1, -2])
            exp_val, std = post_processor.get_expectation_value(observable, loc=0)
            self.assertAlmostEqual(exp_val, -1.6406249999999998)
            self.assertAlmostEqual(std, 1.3442744428582185)
        with self.subTest("Test with default ``loc`` for parametrized circuit."):
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            qc.ry(theta=Parameter("theta"), qubit=0)
            povm_sampler = POVMSampler(sampler=Sampler(seed=self.SEED))
            measurement = ClassicalShadows(num_qubits=2, seed=self.SEED)
            job = povm_sampler.run(
                [(qc, np.array(2 * [[0, np.pi / 3, np.pi]]))], shots=32, povm=measurement
            )
            pub_result = job.result()[0]
            post_processor = POVMPostProcessor(pub_result)
            exp_val, std = post_processor.get_expectation_value(observable)
            self.assertIsInstance(exp_val, np.ndarray)
            self.assertEqual(exp_val.shape, (2, 3))
            self.assertTrue(
                np.allclose(
                    exp_val,
                    np.array(
                        [
                            [-4.171875, 0.703125, -2.578125],
                            [-0.5625, -0.5625, -2.15625],
                        ]
                    ),
                )
            )
            self.assertIsInstance(std, np.ndarray)
            self.assertEqual(std.shape, (2, 3))
            self.assertTrue(
                np.allclose(
                    std,
                    np.array(
                        [
                            [1.59914439, 0.41510017, 0.8915795],
                            [1.46287216, 1.11232782, 1.04977856],
                        ]
                    ),
                )
            )

    def test_single_exp_value_and_std(self):
        """Test that the ``_single_exp_value_and_std`` method works correctly."""
        observable = SparsePauliOp(["ZX", "XZ", "YY"], coeffs=[1.2, 2, -3])
        post_processor = POVMPostProcessor(self.pub_result)
        exp_val, std = post_processor._single_exp_value_and_std(observable, loc=0)
        self.assertAlmostEqual(exp_val, 6.862499999999998)
        self.assertAlmostEqual(std, 1.9438371907630394)
