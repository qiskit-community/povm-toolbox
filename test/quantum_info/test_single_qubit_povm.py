# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the SingleQubitPOVM class."""

from collections import defaultdict
from unittest import TestCase

import numpy as np
from povm_toolbox.library import ClassicalShadows, RandomizedProjectiveMeasurements
from povm_toolbox.quantum_info.single_qubit_povm import SingleQubitPOVM
from qiskit.quantum_info import Operator
from scipy.stats import unitary_group


class TestSingleQubitPovm(TestCase):
    """Test that we can create valid single qubit POVM and get warnings if invalid."""

    def test_random_operators(self):
        """Test"""

        ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))

        while np.abs(ops[0, 0, 0].imag) < 1e-6:
            ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))

        with self.assertRaises(ValueError):
            povm1 = SingleQubitPOVM(list_operators=[Operator(op) for op in ops])
            povm1.check_validity()

    def test_pauli_decomposition(self):
        """Test"""

        # TODO : select a random POVM ...
        dim = 2
        eigval = np.random.uniform(low=-5, high=5, size=dim)
        x = unitary_group.rvs(dim)  # , random_state=seed_obs[i])
        obs = x @ np.diag(eigval) @ x.T.conj()

        _, V = np.linalg.eigh(obs)

        sqpovm = SingleQubitPOVM.from_vectors(V)

        summed = defaultdict(complex)
        for pauli_op in sqpovm.pauli_operators:
            for pauli, coeff in pauli_op.items():
                summed[pauli] += coeff

        with self.subTest("I"):
            self.assertAlmostEqual(summed["I"], 1.0)
        with self.subTest("X"):
            self.assertAlmostEqual(summed["X"], 0.0)
        with self.subTest("Y"):
            self.assertAlmostEqual(summed["Y"], 0.0)
        with self.subTest("Z"):
            self.assertAlmostEqual(summed["Z"], 0.0)

        # also check that the decomposition is correct TODO

    def test_get_bloch_vectors(self):
        """Test the method :method:`.SingleQubitPOVM.get_bloch_vectors`."""

        sqpovm = SingleQubitPOVM(
            [
                0.8 * Operator.from_label("0"),
                0.8 * Operator.from_label("1"),
                0.2 * Operator.from_label("+"),
                0.2 * Operator.from_label("-"),
            ]
        )
        vectors = np.array([[0, 0, 0.8], [0, 0, -0.8], [0.2, 0, 0], [-0.2, 0, 0]])
        self.assertTrue(np.allclose(sqpovm.get_bloch_vectors(), vectors))

        sqpovm = SingleQubitPOVM(
            [
                0.4 * Operator.from_label("-"),
                0.4 * Operator.from_label("+"),
                0.6 * Operator.from_label("r"),
                0.6 * Operator.from_label("l"),
            ]
        )
        vectors = np.array([[-0.4, 0, 0], [0.4, 0, 0], [0, 0.6, 0], [0, -0.6, 0]])
        self.assertTrue(np.allclose(sqpovm.get_bloch_vectors(), vectors))

        sqpovm = SingleQubitPOVM(
            [
                0.4 * Operator(np.eye(2)),
                0.6 * Operator.from_label("r"),
                0.6 * Operator.from_label("l"),
            ]
        )
        with self.assertRaises(ValueError):
            sqpovm.get_bloch_vectors()

        sqpovm = ClassicalShadows(1).definition()[(0,)]
        vectors = np.array(
            [
                [0, 0, 1 / 3],
                [0, 0, -1 / 3],
                [1 / 3, 0, 0],
                [-1 / 3, 0, 0],
                [0, 1 / 3, 0],
                [0, -1 / 3, 0],
            ]
        )
        self.assertTrue(np.allclose(sqpovm.get_bloch_vectors(), vectors))

        sqpovm = RandomizedProjectiveMeasurements(
            1,
            bias=np.array([0.2, 0.8]),
            angles=np.array([0.25 * np.pi, 0, 0.25 * np.pi, 0.25 * np.pi]),
        ).definition()[(0,)]
        vectors = np.sqrt(0.5) * np.array(
            [
                [0.2, 0, 0.2],
                [-0.2, 0, -0.2],
                [0.8 * np.sqrt(0.5), 0.8 * np.sqrt(0.5), 0.8],
                [-0.8 * np.sqrt(0.5), -0.8 * np.sqrt(0.5), -0.8],
            ]
        )
        self.assertTrue(np.allclose(sqpovm.get_bloch_vectors(), vectors))

    # TODO: write a unittest for each public method of SingleQubitPOVM

    # TODO: write a unittest to assert the correct handling of invalid inputs (i.e. verify that
    # errors are raised properly)
