# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the ProductFrame class."""

from unittest import TestCase

import numpy as np
from povm_toolbox.quantum_info.multi_qubit_frame import MultiQubitFrame
from povm_toolbox.quantum_info.product_frame import ProductFrame
from qiskit.quantum_info import Operator, SparsePauliOp


class TestProductFrame(TestCase):
    """Tests for the ``ProductFrame`` class."""

    def test_analysis(self):
        """Test that the ``analysis`` method works correctly."""
        paulis = ["I", "X", "Y", "Z"]
        frame_1_qubit = MultiQubitFrame([Operator.from_label(label) for label in paulis])

        num_qubit_max = 3

        for num_qubit in range(1, num_qubit_max + 1):
            frame_product = ProductFrame.from_list(frames=num_qubit * [frame_1_qubit])

            product_paulis = []
            for idx in np.ndindex(frame_product.shape):
                product_paulis.append("".join([paulis[i] for i in idx]))

            frame_n_qubit = ProductFrame.from_list(
                [MultiQubitFrame([Operator.from_label(label) for label in product_paulis])]
            )

            for i, pauli_string in enumerate(product_paulis):
                decomposition_weights_n_qubit = frame_n_qubit.analysis(
                    SparsePauliOp(pauli_string[::-1])
                )
                decomposition_weights_product = frame_product.analysis(
                    SparsePauliOp(pauli_string[::-1])
                ).flatten()
                check = np.zeros(len(product_paulis))
                check[i] = 2**num_qubit
                self.assertTrue(np.allclose(decomposition_weights_n_qubit, check))
                self.assertTrue(np.allclose(decomposition_weights_product, check))

            decomposition_weights_n_qubit = frame_n_qubit.analysis(
                SparsePauliOp(
                    [pauli_string[::-1] for pauli_string in product_paulis],
                    np.ones(len(product_paulis)),
                )
            )
            decomposition_weights_product = frame_product.analysis(
                SparsePauliOp(
                    [pauli_string[::-1] for pauli_string in product_paulis],
                    np.ones(len(product_paulis)),
                )
            ).flatten()
            check = np.ones(len(product_paulis)) * 2**num_qubit
            self.assertTrue(np.allclose(decomposition_weights_n_qubit, check))
            self.assertTrue(np.allclose(decomposition_weights_product, check))

    def test_shape(self):
        """Test that the ``shape`` property works correctly."""
        paulis = ["I", "X", "Y", "Z"]

        frame_0 = MultiQubitFrame([Operator.from_label(label) for label in paulis], shape=(2, 2))
        frame_1 = MultiQubitFrame([Operator.from_label(label) for label in paulis])
        prod_frame = ProductFrame.from_list([frame_0, frame_1])

        with self.subTest("Test shape."):
            self.assertEqual(prod_frame.shape, (2, 2, 4))
            self.assertEqual(prod_frame._sub_shapes, ((2, 2), (4,)))

        with self.subTest("Test reshape."):
            prod_frame[(1,)].shape = (1, 2, 2)
            self.assertEqual(prod_frame.shape, (2, 2, 1, 2, 2))
            self.assertEqual(prod_frame._sub_shapes, ((2, 2), (1, 2, 2)))

    def test_custom_shape_and_analysis(self):
        """Test that the ``analysis`` method works correctly when the frame has a custom shape."""
        paulis = ["I", "X", "Y", "Z"]

        frame_0 = MultiQubitFrame([Operator.from_label(label) for label in paulis], shape=(2, 2))
        frame_1 = MultiQubitFrame([Operator.from_label(label) for label in paulis])
        prod_frame = ProductFrame.from_list([frame_0, frame_1])

        with self.subTest("Test analysis method with specific index."):
            val = prod_frame.analysis(
                hermitian_op=Operator.from_label("II"), frame_op_idx=(0, 0, 0)
            )
            self.assertAlmostEqual(val, 4.0)
            val = prod_frame.analysis(
                hermitian_op=Operator.from_label("II"), frame_op_idx=(0, 0, 1)
            )
            self.assertAlmostEqual(val, 0.0)
            val = prod_frame.analysis(
                hermitian_op=Operator.from_label("XI"), frame_op_idx=(0, 0, 1)
            )
            self.assertAlmostEqual(val, 4.0)
            val = prod_frame.analysis(
                hermitian_op=Operator.from_label("II"), frame_op_idx=(1, 0, 1)
            )
            self.assertAlmostEqual(val, 0.0)
            val = prod_frame.analysis(
                hermitian_op=Operator.from_label("XY"), frame_op_idx=(1, 0, 1)
            )
            self.assertAlmostEqual(val, 4.0)
            val = prod_frame.analysis(
                hermitian_op=Operator.from_label("ZZ"), frame_op_idx=(1, 1, 3)
            )
            self.assertAlmostEqual(val, 4.0)

        with self.subTest("Test index out of bounds.") and self.assertRaises(ValueError):
            prod_frame.analysis(hermitian_op=Operator.from_label("II"), frame_op_idx=(2, 0, 0))

        with self.subTest("Test index too short.") and self.assertRaises(ValueError):
            prod_frame.analysis(hermitian_op=Operator.from_label("II"), frame_op_idx=(0, 0))

        with self.subTest("Test index too long.") and self.assertRaises(ValueError):
            prod_frame.analysis(hermitian_op=Operator.from_label("II"), frame_op_idx=(0, 0, 0, 0))

        with self.subTest("Test analysis method with set of indices.") and self.assertRaises(
            KeyError
        ):
            val = prod_frame.analysis(
                hermitian_op=Operator.from_label("II"), frame_op_idx={(0, 0, 0), (0, 1, 2)}
            )
            self.assertIsInstance(val, dict)
            self.assertAlmostEqual(val[0, 0, 0], 4.0)
            self.assertAlmostEqual(val[0, 1, 2], 0.0)
            val = prod_frame.analysis(
                hermitian_op=Operator.from_label("YZ"), frame_op_idx={(0, 0, 0), (1, 1, 2)}
            )
            self.assertAlmostEqual(val[0, 0, 0], 0.0)
            self.assertAlmostEqual(val[1, 1, 2], 4.0)
            val[0, 1, 2]

        with self.subTest("Test analysis method with all indices."):
            val = prod_frame.analysis(
                hermitian_op=Operator.from_label("XZ") + Operator.from_label("II"),
                frame_op_idx=None,
            )
            self.assertIsInstance(val, np.ndarray)
            self.assertEqual(val.shape, (2, 2, 4))
            expected_val = np.zeros(shape=(2, 2, 4))
            expected_val[0, 0, 0] = expected_val[1, 1, 1] = 4.0
            self.assertTrue(np.allclose(val, expected_val))

        frame_2 = MultiQubitFrame([Operator.from_label(label) for label in paulis], shape=(1, 2, 2))
        prod_frame = ProductFrame.from_list([frame_0, frame_2])
        with self.subTest("Test analysis method with special shape."):
            val = prod_frame.analysis(
                hermitian_op=Operator.from_label("II"), frame_op_idx=(0, 1, 0, 1, 0)
            )
            self.assertAlmostEqual(val, 0.0)
            val = prod_frame.analysis(
                hermitian_op=Operator.from_label("YX"), frame_op_idx=(0, 1, 0, 1, 0)
            )
            self.assertAlmostEqual(val, 4.0)

    def test_get_operator(self):
        """Test that the ``get_operator`` method works correctly."""
        frame_0 = MultiQubitFrame([Operator.from_label(label) for label in ["I", "X", "Y", "Z"]])
        frame_1 = MultiQubitFrame([Operator.from_label(label) for label in ["0", "1"]])
        frame_product = ProductFrame.from_list(frames=[frame_0, frame_1])

        with self.subTest("Test method works correctly"):
            frame_op_idx = (0, 1)
            expected_snapshot = {(0,): Operator.from_label("I"), (1,): Operator.from_label("1")}
            snapshot = frame_product.get_operator(frame_op_idx)
            self.assertDictEqual(snapshot, expected_snapshot)

            frame_op_idx = (2, 0)
            expected_snapshot = {(0,): Operator.from_label("Y"), (1,): Operator.from_label("0")}
            snapshot = frame_product.get_operator(frame_op_idx)
            self.assertDictEqual(snapshot, expected_snapshot)

        with self.subTest("invalid frame_op_idx") and self.assertRaises(IndexError):
            frame_op_idx = (10, 20)
            frame_product.get_operator(frame_op_idx)
