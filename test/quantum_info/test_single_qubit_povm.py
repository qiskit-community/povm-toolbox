"""Tests for the SingleQubitPOVM class."""

from collections import defaultdict
from unittest import TestCase

import numpy as np
from povms.quantum_info.single_qubit_povm import SingleQubitPOVM
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
            povm1 = SingleQubitPOVM(povm_ops=[Operator(op) for op in ops])
            povm1.check_validity()

    def test_pauli_decomposition(self):
        """Test"""

        # TODO : select a random POVM ...
        dim = 2
        eival = np.random.uniform(low=-5, high=5, size=dim)
        x = unitary_group.rvs(dim)  # , random_state=seed_obs[i])
        obs = x @ np.diag(eival) @ x.T.conj()

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

    # TODO: write a unittest for each public method of SingleQubitPOVM

    # TODO: write a unittest to assert the correct handling of invalid inputs (i.e. verify that
    # errors are raised properly)
