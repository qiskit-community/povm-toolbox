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
from povm_toolbox.quantum_info.multi_qubit_povm import MultiQubitFrame
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator


class TestMultiQubitFrame(TestCase):
    """Test that we can create valid frame and get warnings if invalid."""

    def test_invalid_operators(self):
        """Test that errors are correctly raised if invalid operators are supplied."""
        with self.subTest("Non Hermitian operators") and self.assertRaises(ValueError):
            ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))
            while np.abs(ops[0, 0, 0].imag) < 1e-6:
                ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(
                    -1, 1, (6, 2, 2)
                )
            _ = MultiQubitFrame(list_operators=[Operator(op) for op in ops])

    def test_dimension(self):
        """Test dimension attribute."""
        for dim in range(1, 10):
            frame = MultiQubitFrame(3 * [Operator(1.0 / 3.0 * np.eye(dim))])
            self.assertEqual(dim, frame.dimension)
            self.assertEqual(dim, frame._dimension)
            self.assertEqual((dim, dim), frame.operators[1].dim)

    def test_getitem(self):
        """Test the ``__getitem__`` method."""
        n = 6
        frame = MultiQubitFrame(
            [Operator(2 * i / (n * (n + 1)) * np.eye(4)) for i in range(1, n + 1)]
        )
        with self.subTest("1d shape"):
            self.assertIsInstance(frame[2], Operator)
            self.assertIsInstance(frame[1:2], list)
            self.assertIsInstance(frame[:2], list)
            self.assertEqual(frame[0], frame.operators[0])
            self.assertListEqual(
                frame[3:], [frame.operators[3], frame.operators[4], frame.operators[5]]
            )
            self.assertListEqual(
                frame[::2], [frame.operators[0], frame.operators[2], frame.operators[4]]
            )
            self.assertListEqual(
                frame[2::-1], [frame.operators[2], frame.operators[1], frame.operators[0]]
            )
        with self.subTest("2d shape"):
            frame.shape = (2, 3)
            self.assertIsInstance(frame[0, 2], Operator)
            self.assertEqual(frame[0, 0], frame.operators[0])
            self.assertEqual(frame[(0, 1)], frame.operators[1])
            self.assertEqual(frame[1, 2], frame.operators[5])
        with self.subTest("index too short") and self.assertRaises(ValueError):
            frame[0]
        with self.subTest("index too long") and self.assertRaises(ValueError):
            frame[0, 0, 0]
        with self.subTest("out of bound index") and self.assertRaises(ValueError):
            frame[0, 4]

    def test_informationally_complete(self):
        """Test whether a frame is informationally complete or not."""
        paulis = ["I", "X", "Y", "Z"]
        with self.subTest("IC frame"):
            frame = MultiQubitFrame([Operator.from_label(label) for label in paulis])
            self.assertTrue(frame.informationally_complete)

        with self.subTest("non-IC frame"):
            frame = MultiQubitFrame([Operator.from_label(label) for label in paulis[1:]])
            self.assertFalse(frame.informationally_complete)

    def test_repr(self):
        """Test the ``__repr__`` method."""
        with self.subTest("Single-qubit case."):
            frame = MultiQubitFrame([Operator.from_label("0"), Operator.from_label("1")])
            self.assertEqual(frame.__repr__(), f"MultiQubitFrame<2> at {hex(id(frame))}")
            frame = MultiQubitFrame(
                2 * [0.5 * Operator.from_label("0"), 0.5 * Operator.from_label("1")]
            )
            self.assertEqual(frame.__repr__(), f"MultiQubitFrame<4> at {hex(id(frame))}")
            frame.shape = (2, 2)
            self.assertEqual(frame.__repr__(), f"MultiQubitFrame<2,2> at {hex(id(frame))}")
        with self.subTest("Multi-qubit case."):
            frame = MultiQubitFrame(
                [
                    Operator.from_label("II"),
                    Operator.from_label("IX"),
                    Operator.from_label("XI"),
                    Operator.from_label("XX"),
                ]
            )
            self.assertEqual(
                frame.__repr__(), f"MultiQubitFrame(num_qubits=2)<4> at {hex(id(frame))}"
            )
            frame.shape = (2, 2)
            self.assertEqual(
                frame.__repr__(), f"MultiQubitFrame(num_qubits=2)<2,2> at {hex(id(frame))}"
            )

    def test_operators_setter(self):
        """Test the ``__repr__`` method."""
        with self.subTest("Non-square operators") and self.assertRaises(ValueError):
            frame = MultiQubitFrame([Operator.from_label("0"), Operator.from_label("1")])
            frame.operators = [
                Operator(np.array([[1, 0, 0], [0, 0, 0]])),
                Operator(np.array([[0, 0, 0], [0, 1, 0]])),
            ]
        with self.subTest("Change number of operators"):
            frame = MultiQubitFrame([Operator.from_label("0"), Operator.from_label("1")])
            frame.operators = [Operator.from_label("I")]
            self.assertEqual(frame.shape, (1,))
        with self.subTest("Number of operators incompatible with shape.") and self.assertRaises(
            ValueError
        ):
            frame = MultiQubitFrame(
                [Operator.from_label("0"), Operator.from_label("1")], shape=(2,)
            )
            frame.operators = [Operator.from_label("I")]

    def test_pauli_operators(self):
        """Test errors are raised  correctly for the ``pauli_operators`` attribute."""
        frame = MultiQubitFrame([Operator(np.eye(3))])
        with self.subTest("Non-qubit operators") and self.assertRaises(QiskitError):
            _ = frame.pauli_operators

    def test_analysis(self):
        """Test that the ``analysis`` method works correctly."""
        frame = MultiQubitFrame([Operator.from_label(label) for label in ["0", "1", "I", "Z"]])
        frame_shaped = MultiQubitFrame(
            [Operator.from_label(label) for label in ["0", "1", "I", "Z"]], shape=(2, 2)
        )
        operator = Operator([[0.8, 0], [0, 0.2]])
        with self.subTest("Get a single frame coefficient."):
            self.assertAlmostEqual(frame.analysis(operator, 0), 0.8)
            self.assertAlmostEqual(frame.analysis(operator, 1), 0.2)
            self.assertAlmostEqual(frame.analysis(operator, 2), 1.0)
            self.assertAlmostEqual(frame.analysis(operator, 3), 0.6)
            self.assertAlmostEqual(frame_shaped.analysis(operator, (0, 0)), 0.8)
            self.assertAlmostEqual(frame_shaped.analysis(operator, (0, 1)), 0.2)
            self.assertAlmostEqual(frame_shaped.analysis(operator, (1, 0)), 1.0)
            self.assertAlmostEqual(frame_shaped.analysis(operator, (1, 1)), 0.6)
        with self.subTest("Get a set of frame coefficients."):
            frame_coefficients = frame.analysis(operator, set([0]))
            self.assertIsInstance(frame_coefficients, dict)
            self.assertAlmostEqual(frame_coefficients[0], 0.8)
            frame_coefficients = frame.analysis(operator, set([1, 0]))
            self.assertIsInstance(frame_coefficients, dict)
            self.assertAlmostEqual(frame_coefficients[0], 0.8)
            self.assertAlmostEqual(frame_coefficients[1], 0.2)
            frame_coefficients = frame_shaped.analysis(operator, set([(1, 0), (0, 0)]))
            self.assertIsInstance(frame_coefficients, dict)
            self.assertAlmostEqual(frame_coefficients[0, 0], 0.8)
            self.assertAlmostEqual(frame_coefficients[(1, 0)], 1.0)
        with self.subTest("Get all frame coefficients."):
            frame_coefficients = frame.analysis(operator)
            self.assertIsInstance(frame_coefficients, np.ndarray)
            self.assertTrue(np.allclose(frame_coefficients, np.array([0.8, 0.2, 1.0, 0.6])))
            frame_coefficients = frame_shaped.analysis(operator)
            self.assertIsInstance(frame_coefficients, np.ndarray)
            self.assertTrue(np.allclose(frame_coefficients, np.array([0.8, 0.2, 1.0, 0.6])))
        with self.subTest("Invalid value for ``frame_op_idx``.") and self.assertRaises(ValueError):
            _ = frame.analysis(operator, (0, 1))
        with self.subTest("Invalid type for ``frame_op_idx``.") and self.assertRaises(TypeError):
            _ = frame.analysis(operator, [0])

    def test_shape(self):
        """Test that the ``shape`` property works correctly."""
        paulis = ["I", "X", "Y", "Z"]
        with self.subTest("Test works correctly"):
            frame = MultiQubitFrame([Operator.from_label(label) for label in paulis], shape=(2, 2))
            self.assertEqual(frame.shape, (2, 2))
            frame.shape = (4, 1)
            self.assertEqual(frame.shape, (4, 1))
            frame.shape = None
            self.assertEqual(frame.shape, (4,))
        with self.subTest("Test raises errors correctly") and self.assertRaises(ValueError):
            frame = MultiQubitFrame([Operator.from_label(label) for label in paulis])
            frame.shape = (2, 3)
        with self.subTest("Index incompatible with shape") and self.assertRaises(ValueError):
            frame = MultiQubitFrame([Operator.from_label(label) for label in paulis], shape=(2, 2))
            frame._ravel_index(0)
