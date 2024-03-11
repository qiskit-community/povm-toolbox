"""Tests for the MultiQubitPOVM class."""

from unittest import TestCase

import numpy as np
from povms.quantum_info.multi_qubit_povm import MultiQubitPOVM
from qiskit.quantum_info import Operator


class TestMultiQubitPOVM(TestCase):
    """Test that we can create valid POVM and get warnings if invalid."""

    def test_random_operators(self):
        """Test"""

        ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))

        while np.abs(ops[0, 0, 0].imag) < 1e-6:
            ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))

        with self.assertRaises(ValueError):
            povm1 = MultiQubitPOVM(povm_ops=[Operator(op) for op in ops])
            print(povm1[0])

    def test_dimension(self):
        for dim in range(10):
            povm = MultiQubitPOVM(3 * [Operator(1.0 / 3.0 * np.eye(dim))])
            self.assertEqual(dim, povm.dimension)
            self.assertEqual(dim, povm._dimension)
            self.assertEqual((dim, dim), povm.povm_operators[1].dim)

    def test_n_outcomes(self):
        for n in range(1, 10):
            for dim in range(1, 10):
                povm = MultiQubitPOVM(n * [Operator(1.0 / n * np.eye(dim))])
                self.assertEqual(n, povm.n_outcomes)
                self.assertEqual(n, povm._n_outcomes)
                self.assertEqual(n, len(povm))
                self.assertEqual(n, len(povm.povm_operators))

    def test_getitem(self):
        n = 6
        povm = MultiQubitPOVM(
            [Operator(2 * i / (n * (n + 1)) * np.eye(4)) for i in range(1, n + 1)]
        )
        self.assertIsInstance(povm[2], Operator)
        self.assertIsInstance(povm[1:2], list)
        self.assertIsInstance(povm[:2], list)
        self.assertEqual(povm[0], povm.povm_operators[0])
        self.assertListEqual(
            povm[3:], [povm.povm_operators[3], povm.povm_operators[4], povm.povm_operators[5]]
        )
        self.assertListEqual(
            povm[::2], [povm.povm_operators[0], povm.povm_operators[2], povm.povm_operators[4]]
        )
        self.assertListEqual(
            povm[2::-1], [povm.povm_operators[2], povm.povm_operators[1], povm.povm_operators[0]]
        )

    # TODO
    def test_build_from_vectors(self):
        if True:
            self.assertTrue(True)

    # TODO: write a unittest for each public method of MultiQubitPOVM

    # TODO: write a unittest to assert the correct handling of invalid inputs (i.e. verify that
    # errors are raised properly)
