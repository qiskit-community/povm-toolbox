# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the ProductPOVM class."""

from unittest import TestCase

import numpy as np
from numpy.random import default_rng
from povm_toolbox.quantum_info.multi_qubit_dual import MultiQubitDual
from povm_toolbox.quantum_info.multi_qubit_povm import MultiQubitPOVM
from povm_toolbox.quantum_info.product_dual import ProductDual
from povm_toolbox.quantum_info.product_povm import ProductPOVM
from povm_toolbox.quantum_info.single_qubit_povm import SingleQubitPOVM
from qiskit.quantum_info import Operator


class TestProductPOVM(TestCase):
    """Test that we can create valid product POVM and get warnings if invalid."""

    SEED = 12

    def test_is_dual_to(self):
        sq_povm = SingleQubitPOVM(
            [
                1.0 / 3 * Operator.from_label("0"),
                1.0 / 3 * Operator.from_label("1"),
                1.0 / 3 * Operator.from_label("+"),
                1.0 / 3 * Operator.from_label("-"),
                1.0 / 3 * Operator.from_label("r"),
                1.0 / 3 * Operator.from_label("l"),
            ]
        )

        sq_dual = MultiQubitDual(
            [
                Operator([[2.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]]),
                Operator([[-1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 2.0 + 0.0j]]),
                Operator([[0.5 + 0.0j, 1.5 + 0.0j], [1.5 + 0.0j, 0.5 + 0.0j]]),
                Operator([[0.5 + 0.0j, -1.5 + 0.0j], [-1.5 + 0.0j, 0.5 + 0.0j]]),
                Operator([[0.5 + 0.0j, 0.0 - 1.5j], [0.0 + 1.5j, 0.5 + 0.0j]]),
                Operator([[0.5 + 0.0j, 0.0 + 1.5j], [0.0 - 1.5j, 0.5 + 0.0j]]),
            ]
        )

        povm = ProductPOVM.from_list([sq_povm])

        with self.subTest("Test valid dual."):
            dual = ProductDual.from_list([sq_dual])
            self.assertTrue(dual.is_dual_to(povm))

        with self.subTest("Test not implemented type for ``frame``.") and self.assertRaises(
            NotImplementedError
        ):
            self.assertFalse(dual.is_dual_to(sq_povm))

        with self.subTest("Test invalid dual."):
            invalid_dual = ProductDual.from_list(
                [
                    MultiQubitDual(
                        [
                            Operator.from_label("0"),
                            Operator.from_label("1"),
                            Operator.from_label("+"),
                            Operator.from_label("-"),
                            Operator.from_label("r"),
                            Operator.from_label("l"),
                        ]
                    )
                ]
            )
            self.assertFalse(invalid_dual.is_dual_to(povm))

    def test_build_dual(self):
        num_qubits = 3
        rng = default_rng(self.SEED)
        q = rng.uniform(0, 5, size=3 * num_qubits).reshape((num_qubits, 3))
        q /= q.sum(axis=1)[:, np.newaxis]

        povm_list = []
        for i in range(num_qubits):
            povm_list.append(
                SingleQubitPOVM(
                    [
                        q[i, 0] * Operator.from_label("0"),
                        q[i, 0] * Operator.from_label("1"),
                        q[i, 1] * Operator.from_label("+"),
                        q[i, 1] * Operator.from_label("-"),
                        q[i, 2] * Operator.from_label("r"),
                        q[i, 2] * Operator.from_label("l"),
                    ]
                )
            )
        prod_povm = ProductPOVM.from_list(povm_list)

        with self.subTest("Default ``alphas``."):
            dual_1 = ProductDual.build_dual_from_frame(prod_povm)
            self.assertTrue(dual_1.is_dual_to(prod_povm))
        with self.subTest("Specified ``alphas``."):
            dual_2 = ProductDual.build_dual_from_frame(
                prod_povm,
                alphas=tuple(
                    (q[i, 0], q[i, 0], q[i, 1], q[i, 1], q[i, 2], q[i, 2])
                    for i in range(num_qubits)
                ),
            )
            self.assertTrue(dual_2.is_dual_to(prod_povm))
            for i in range(num_qubits):
                for k in range(dual_2[(i,)].num_outcomes):
                    self.assertTrue(np.allclose(dual_1[(i,)][k], dual_2[(i,)][k]))
        with self.subTest("Invalid ``alphas``.") and self.assertRaises(ValueError):
            _ = ProductDual.build_dual_from_frame(
                prod_povm, alphas=((1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1))
            )
        with self.subTest("Not implemented type for ``frame``.") and self.assertRaises(
            NotImplementedError
        ):
            multi_qubit_povm = MultiQubitPOVM(
                [
                    1.0 / 3 * Operator.from_label("0"),
                    1.0 / 3 * Operator.from_label("1"),
                    1.0 / 3 * Operator.from_label("+"),
                    1.0 / 3 * Operator.from_label("-"),
                    1.0 / 3 * Operator.from_label("r"),
                    1.0 / 3 * Operator.from_label("l"),
                ]
            )
            _ = ProductDual.build_dual_from_frame(multi_qubit_povm)

    def test_repr(self):
        """Test the ``__repr__`` method."""
        dual_1 = MultiQubitDual([Operator.from_label("0"), Operator.from_label("1")])
        dual_2 = MultiQubitDual([Operator.from_label("0"), Operator.from_label("1")])
        prod_dual = ProductDual.from_list([dual_1])
        self.assertEqual(
            prod_dual.__repr__(),
            (
                "ProductDual(num_subsystems=1)<2>:\n   (0,): MultiQubitDual<2>"
                f" at {hex(id(dual_1))}"
            ),
        )
        prod_dual = ProductDual.from_list([dual_1, dual_2])
        self.assertEqual(
            prod_dual.__repr__(),
            (
                "ProductDual(num_subsystems=2)<2,2>:\n   (0,): MultiQubitDual<2> at "
                f"{hex(id(dual_1))}\n   (1,): MultiQubitDual<2> at {hex(id(dual_2))}"
            ),
        )
