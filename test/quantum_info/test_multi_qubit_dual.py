# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the MultiQubitDual class."""

from unittest import TestCase

import numpy as np
from povm_toolbox.quantum_info.multi_qubit_dual import MultiQubitDual
from povm_toolbox.quantum_info.multi_qubit_povm import MultiQubitPOVM
from povm_toolbox.quantum_info.product_povm import ProductPOVM
from qiskit.quantum_info import Operator, random_hermitian


class TestMultiQubitDual(TestCase):
    """Test that we can create valid dual frame and get warnings if invalid."""

    def test_invalid_operators(self):
        """Test that errors are correctly raised if invalid operators are supplied."""
        with self.subTest("Non Hermitian operators") and self.assertRaises(ValueError):
            ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))
            while np.abs(ops[0, 0, 0].imag) < 1e-6:
                ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(
                    -1, 1, (6, 2, 2)
                )
            _ = MultiQubitDual(list_operators=[Operator(op) for op in ops])

    def test_num_outcomes(self):
        dual = MultiQubitDual(
            [
                Operator.from_label("0"),
                Operator.from_label("1"),
                Operator.from_label("+"),
            ]
        )
        self.assertEqual(dual.num_operators, 3)
        self.assertEqual(dual.num_outcomes, 3)
        self.assertEqual(len(dual), 3)

    def test_is_dual_to(self):
        povm = MultiQubitPOVM(
            [
                1.0 / 3 * Operator.from_label("0"),
                1.0 / 3 * Operator.from_label("1"),
                1.0 / 3 * Operator.from_label("+"),
                1.0 / 3 * Operator.from_label("-"),
                1.0 / 3 * Operator.from_label("r"),
                1.0 / 3 * Operator.from_label("l"),
            ]
        )
        with self.subTest("Test valid dual."):
            dual = MultiQubitDual(
                [
                    Operator([[2.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]]),
                    Operator([[-1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 2.0 + 0.0j]]),
                    Operator([[0.5 + 0.0j, 1.5 + 0.0j], [1.5 + 0.0j, 0.5 + 0.0j]]),
                    Operator([[0.5 + 0.0j, -1.5 + 0.0j], [-1.5 + 0.0j, 0.5 + 0.0j]]),
                    Operator([[0.5 + 0.0j, 0.0 - 1.5j], [0.0 + 1.5j, 0.5 + 0.0j]]),
                    Operator([[0.5 + 0.0j, 0.0 + 1.5j], [0.0 - 1.5j, 0.5 + 0.0j]]),
                ]
            )
            self.assertTrue(dual.is_dual_to(povm))
        with self.subTest("Test not implemented type for ``frame``.") and self.assertRaises(
            NotImplementedError
        ):
            self.assertFalse(dual.is_dual_to(ProductPOVM.from_list([povm])))
        with self.subTest("Test invalid dual."):
            dual = MultiQubitDual(
                [
                    Operator.from_label("0"),
                    Operator.from_label("1"),
                    Operator.from_label("+"),
                    Operator.from_label("-"),
                    Operator.from_label("r"),
                    Operator.from_label("l"),
                ]
            )
            self.assertFalse(dual.is_dual_to(povm))

    def test_build_dual_from_frame(self):
        povm = MultiQubitPOVM(
            [
                1.0 / 2 * Operator.from_label("0"),
                1.0 / 2 * Operator.from_label("1"),
                1.0 / 3 * Operator.from_label("+"),
                1.0 / 3 * Operator.from_label("-"),
                1.0 / 6 * Operator.from_label("r"),
                1.0 / 6 * Operator.from_label("l"),
            ]
        )
        with self.subTest("Default ``alphas``."):
            dual_1 = MultiQubitDual.build_dual_from_frame(povm)
            self.assertTrue(dual_1.is_dual_to(povm))
        with self.subTest("Specified ``alphas``."):
            dual_2 = MultiQubitDual.build_dual_from_frame(
                povm, alphas=(0.5, 0.5, 1 / 3, 1 / 3, 1 / 6, 1 / 6)
            )
            self.assertTrue(dual_2.is_dual_to(povm))
            for k in range(dual_2.num_outcomes):
                self.assertTrue(np.allclose(dual_1[k], dual_2[k]))
        with self.subTest("Invalid ``alphas``.") and self.assertRaises(ValueError):
            _ = MultiQubitDual.build_dual_from_frame(povm, alphas=(0.5, 0.5, 1 / 3, 1 / 3))
        with self.subTest("Not implemented type for ``frame``.") and self.assertRaises(
            NotImplementedError
        ):
            prod_povm = ProductPOVM.from_list([povm])
            _ = MultiQubitDual.build_dual_from_frame(prod_povm)

    def test_get_omegas(self):
        with self.subTest("Single-qubit case"):
            povm = MultiQubitPOVM(
                [
                    1.0 / 2 * Operator.from_label("0"),
                    1.0 / 2 * Operator.from_label("1"),
                    1.0 / 3 * Operator.from_label("+"),
                    1.0 / 3 * Operator.from_label("-"),
                    1.0 / 6 * Operator.from_label("r"),
                    1.0 / 6 * Operator.from_label("l"),
                ]
            )
            obs = random_hermitian(dims=2**1)
            dual = MultiQubitDual.build_dual_from_frame(povm)
            omegas = dual.get_omegas(obs)
            dec = np.zeros((2, 2), dtype=complex)
            for k in range(povm.num_outcomes):
                dec += omegas[k] * povm[k].data
            self.assertTrue(np.allclose(obs, dec))
