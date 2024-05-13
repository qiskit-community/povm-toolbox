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
from povm_toolbox.quantum_info.multi_qubit_dual import MultiQubitDUAL
from povm_toolbox.quantum_info.multi_qubit_povm import MultiQubitPOVM
from qiskit.quantum_info import Operator, random_hermitian


class TestMultiQubitPOVM(TestCase):
    """Test that we can create valid POVM and get warnings if invalid."""

    def test_random_operators(self):
        """Test that an error is raised if the operators are not Hermitian."""

        ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))

        while np.abs(ops[0, 0, 0].imag) < 1e-6:
            ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))

        with self.assertRaises(ValueError):
            povm1 = MultiQubitPOVM(list_operators=[Operator(op) for op in ops])
            print(povm1[0])

    def test_dimension(self):
        """Test dimension attribute"""
        for dim in range(1, 10):
            povm = MultiQubitPOVM(3 * [Operator(1.0 / 3.0 * np.eye(dim))])
            self.assertEqual(dim, povm.dimension)
            self.assertEqual(dim, povm._dimension)
            self.assertEqual((dim, dim), povm.operators[1].dim)

    def test_n_outcomes(self):
        """Test the number of outcomes, with both `n_outcome` attribute and `__len__` method."""
        for n in range(1, 10):
            for dim in range(1, 10):
                povm = MultiQubitPOVM(n * [Operator(1.0 / n * np.eye(dim))])
                self.assertEqual(n, povm.n_outcomes)
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

    # TODO
    def test_build_from_vectors(self):
        """Test that we can correctly instantiate a POVM from Bloch vectors."""
        if True:
            self.assertTrue(True)

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
            dual = MultiQubitDUAL.build_dual_from_frame(povm)
            omegas = dual.get_omegas(obs)
            dec = np.zeros((2, 2), dtype=complex)
            for k in range(povm.n_outcomes):
                dec += omegas[k] * povm[k].data
            self.assertTrue(np.allclose(obs, dec))

        # TODO
        with self.subTest("Multi-qubit case"):
            self.assertTrue(True)

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

    # TODO: write a unittest for each public method of MultiQubitPOVM

    # TODO: write a unittest to assert the correct handling of invalid inputs (i.e. verify that
    # errors are raised properly)
