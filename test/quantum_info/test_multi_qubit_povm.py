# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the MultiQubitPOVM class."""

from unittest import TestCase

import numpy as np
from povm_toolbox.quantum_info.multi_qubit_povm import MultiQubitPOVM
from povm_toolbox.quantum_info.single_qubit_povm import SingleQubitPOVM
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator


class TestMultiQubitPOVM(TestCase):
    """Test that we can create valid POVM and get warnings if invalid."""

    def test_invalid_operators(self):
        """Test that errors are correctly raised if invalid operators are supplied."""
        with self.subTest("Non Hermitian operators") and self.assertRaises(ValueError):
            ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))
            while np.abs(ops[0, 0, 0].imag) < 1e-6:
                ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(
                    -1, 1, (6, 2, 2)
                )
            _ = MultiQubitPOVM(list_operators=[Operator(op) for op in ops])
        with self.subTest("Operators with negative eigenvalues") and self.assertRaises(ValueError):
            op = np.array([[-0.5, 0], [0, 0]])
            _ = MultiQubitPOVM(list_operators=[Operator(op), Operator(np.eye(2) - op)])
        with self.subTest("Operators not summing up to identity") and self.assertRaises(ValueError):
            _ = MultiQubitPOVM(
                list_operators=[0.9 * Operator.from_label("0"), Operator.from_label("1")]
            )

    def test_dimension(self):
        """Test dimension attribute"""
        for dim in range(1, 10):
            povm = MultiQubitPOVM(3 * [Operator(1.0 / 3.0 * np.eye(dim))])
            self.assertEqual(dim, povm.dimension)
            self.assertEqual(dim, povm._dimension)
            self.assertEqual((dim, dim), povm.operators[1].dim)

    def test_num_outcomes(self):
        """Test the number of outcomes, with both `num_outcomes` attribute and `__len__` method."""
        for n in range(1, 10):
            for dim in range(1, 10):
                povm = MultiQubitPOVM(n * [Operator(1.0 / n * np.eye(dim))])
                self.assertEqual(n, povm.num_outcomes)
                self.assertEqual(n, len(povm))
                self.assertEqual(n, len(povm.operators))

    def test_getitem(self):
        """Test the `__getitem__` method."""
        n = 6
        povm = MultiQubitPOVM(
            [Operator(2 * i / (n * (n + 1)) * np.eye(4)) for i in range(1, n + 1)]
        )
        self.assertIsInstance(povm[2], Operator)
        self.assertIsInstance(povm[1:2], list)
        self.assertIsInstance(povm[:2], list)
        self.assertEqual(povm[0], povm.operators[0])
        self.assertListEqual(povm[3:], [povm.operators[3], povm.operators[4], povm.operators[5]])
        self.assertListEqual(povm[::2], [povm.operators[0], povm.operators[2], povm.operators[4]])
        self.assertListEqual(povm[2::-1], [povm.operators[2], povm.operators[1], povm.operators[0]])

    def test_informationally_complete(self):
        """Test whether a POVM is informationally complete or not."""
        with self.subTest("SIC-POVM"):
            import cmath

            vecs = np.sqrt(1.0 / 2.0) * np.array(
                [
                    [1, 0],
                    [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0)],
                    [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0) * cmath.exp(2.0j * np.pi / 3)],
                    [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0) * cmath.exp(4.0j * np.pi / 3)],
                ]
            )
            sic_povm = MultiQubitPOVM.from_vectors(vecs)
            self.assertTrue(sic_povm.informationally_complete)

        with self.subTest("CS-POVM"):
            coef = 1.0 / 3.0
            cs_povm = MultiQubitPOVM(
                [
                    coef * Operator.from_label("0"),
                    coef * Operator.from_label("1"),
                    coef * Operator.from_label("+"),
                    coef * Operator.from_label("-"),
                    coef * Operator.from_label("r"),
                    coef * Operator.from_label("l"),
                ]
            )
            self.assertTrue(cs_povm.informationally_complete)

        with self.subTest("Non IC-POVM"):
            coef = 1.0 / 2.0
            povm = MultiQubitPOVM(
                [
                    coef * Operator.from_label("0"),
                    coef * Operator.from_label("1"),
                    coef * Operator.from_label("+"),
                    coef * Operator.from_label("-"),
                ]
            )
            self.assertFalse(povm.informationally_complete)

    def test_repr(self):
        """Test the ``__repr__`` method."""
        with self.subTest("Single-qubit case."):
            povm = MultiQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
            self.assertEqual(povm.__repr__(), f"MultiQubitPOVM<2> at {hex(id(povm))}")
            povm = MultiQubitPOVM(
                2 * [0.5 * Operator.from_label("0"), 0.5 * Operator.from_label("1")]
            )
            self.assertEqual(povm.__repr__(), f"MultiQubitPOVM<4> at {hex(id(povm))}")
            povm = SingleQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
            self.assertEqual(povm.__repr__(), f"SingleQubitPOVM<2> at {hex(id(povm))}")
        with self.subTest("Multi-qubit case."):
            povm = MultiQubitPOVM(
                [
                    Operator.from_label("00"),
                    Operator.from_label("01"),
                    Operator.from_label("10"),
                    Operator.from_label("11"),
                ]
            )
            self.assertEqual(povm.__repr__(), f"MultiQubitPOVM(num_qubits=2)<4> at {hex(id(povm))}")

    def test_operators_setter(self):
        """Test the ``__repr__`` method."""
        povm = MultiQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
        with self.subTest("Non-square operators") and self.assertRaises(ValueError):
            povm.operators = [
                Operator(np.array([[1, 0, 0], [0, 0, 0]])),
                Operator(np.array([[0, 0, 0], [0, 1, 0]])),
            ]

    def test_pauli_operators(self):
        """Test errors are raised  correctly for the ``pauli_operators`` attribute."""
        povm = MultiQubitPOVM([Operator(np.eye(3))])
        with self.subTest("Non-qubit operators") and self.assertRaises(QiskitError):
            _ = povm.pauli_operators

    def test_analysis(self):
        povm = MultiQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
        operator = Operator([[0.8, 0], [0, 0.2]])
        with self.subTest("Get a single frame coefficient."):
            self.assertEqual(povm.analysis(operator, 0), 0.8)
            self.assertEqual(povm.analysis(operator, 1), 0.2)
        with self.subTest("Get a set of frame coefficients."):
            frame_coefficients = povm.analysis(operator, set([0]))
            self.assertIsInstance(frame_coefficients, dict)
            self.assertEqual(frame_coefficients[0], 0.8)
            frame_coefficients = povm.analysis(operator, set([1, 0]))
            self.assertIsInstance(frame_coefficients, dict)
            self.assertEqual(frame_coefficients[0], 0.8)
            self.assertEqual(frame_coefficients[1], 0.2)
        with self.subTest("Get all frame coefficients."):
            frame_coefficients = povm.analysis(operator)
            self.assertIsInstance(frame_coefficients, np.ndarray)
            self.assertTrue(np.allclose(frame_coefficients, np.array([0.8, 0.2])))
        with self.subTest("Invalid type for ``frame_op_idx``.") and self.assertRaises(TypeError):
            _ = povm.analysis(operator, (0, 1))

    def test_draw_bloch(self):
        with self.assertRaises(NotImplementedError):
            povm = MultiQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
            povm.draw_bloch()
