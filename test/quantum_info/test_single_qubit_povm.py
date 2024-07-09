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

    def test_invalid_dimension(self):
        """Test that an error is raised if the dimension of the operators is different than 2 (single-qubit system)."""
        with self.subTest("Dimension less than 2.") and self.assertRaises(ValueError):
            SingleQubitPOVM([Operator(np.eye(1))])
        with self.subTest("Dimension greater than 2.") and self.assertRaises(ValueError):
            SingleQubitPOVM([Operator(np.eye(4))])

    def test_pauli_decomposition(self):
        """Test the pauli decomposition of the POVM effects."""
        dim = 2
        eigval = np.random.uniform(low=-5, high=5, size=dim)
        x = unitary_group.rvs(dim)
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
