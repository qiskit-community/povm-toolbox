"""Tests for the RandomizedPMs class."""

from unittest import TestCase

import numpy as np
from povms.library.pm_sim_implementation import ClassicalShadows, LocallyBiased, RandomizedPMs
from povms.quantum_info.single_qubit_povm import SingleQubitPOVM
from qiskit.quantum_info import Operator


class TestRandomizedPMs(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        basis_0 = np.array([1.0, 0], dtype=complex)
        basis_1 = np.array([0, 1.0], dtype=complex)
        basis_plus = 1.0 / np.sqrt(2) * (basis_0 + basis_1)
        basis_minus = 1.0 / np.sqrt(2) * (basis_0 - basis_1)
        basis_plus_i = 1.0 / np.sqrt(2) * (basis_0 + 1.0j * basis_1)
        basis_minus_i = 1.0 / np.sqrt(2) * (basis_0 - 1.0j * basis_1)

        self.Z0 = np.outer(basis_0, basis_0.conj())
        self.Z1 = np.outer(basis_1, basis_1.conj())
        self.X0 = np.outer(basis_plus, basis_plus.conj())
        self.X1 = np.outer(basis_minus, basis_minus.conj())
        self.Y0 = np.outer(basis_plus_i, basis_plus_i.conj())
        self.Y1 = np.outer(basis_minus_i, basis_minus_i.conj())

    def test_CS_build(self):
        """Test if we can build a standard Classical Shadow POVM from the generic class"""

        q = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        sqpovm = SingleQubitPOVM(
            [
                q[0] * Operator.from_label("0"),
                q[0] * Operator.from_label("1"),
                q[1] * Operator.from_label("+"),
                q[1] * Operator.from_label("-"),
                q[2] * Operator.from_label("r"),
                q[2] * Operator.from_label("l"),
            ]
        )

        for n_qubit in range(1, 11):
            cs_implementation = ClassicalShadows(n_qubit=n_qubit)
            self.assertEqual(n_qubit, cs_implementation.n_qubit)
            cs_povm = cs_implementation.to_povm()
            for i in range(n_qubit):
                self.assertEqual(cs_povm._povms[(i,)].n_outcomes, sqpovm.n_outcomes)
                for k in range(sqpovm.n_outcomes):
                    self.assertAlmostEqual(cs_povm._povms[(i,)][k], sqpovm[k])

    def test_LBCS_build(self):
        """Test if we can build a LB Classical Shadow POVM from the generic class"""

        for n_qubit in range(1, 11):
            q = np.random.uniform(0, 5, size=3 * n_qubit).reshape((n_qubit, 3))
            q /= q.sum(axis=1)[:, np.newaxis]

            cs_implementation = LocallyBiased(n_qubit=n_qubit, bias=q)
            self.assertEqual(n_qubit, cs_implementation.n_qubit)
            cs_povm = cs_implementation.to_povm()
            for i in range(n_qubit):
                sqpovm = SingleQubitPOVM(
                    [
                        q[i, 0] * Operator.from_label("0"),
                        q[i, 0] * Operator.from_label("1"),
                        q[i, 1] * Operator.from_label("+"),
                        q[i, 1] * Operator.from_label("-"),
                        q[i, 2] * Operator.from_label("r"),
                        q[i, 2] * Operator.from_label("l"),
                    ]
                )
                self.assertEqual(cs_povm._povms[(i,)].n_outcomes, sqpovm.n_outcomes)
                for k in range(sqpovm.n_outcomes):
                    self.assertTrue(np.allclose(cs_povm._povms[(i,)][k], sqpovm[k]))

    def test_qc_build(self):
        """Test if we can build a QunatumCircuit."""

        for n_qubit in range(1, 11):
            q = np.random.uniform(0, 5, size=3 * n_qubit).reshape((n_qubit, 3))
            q /= q.sum(axis=1)[:, np.newaxis]

            angles = np.array([0.0, 0.0, 0.5 * np.pi, 0.0, 0.5 * np.pi, 0.5 * np.pi])

            cs_implementation = RandomizedPMs(n_qubit=n_qubit, bias=q, angles=angles)

            qc = cs_implementation._build_qc()

            self.assertEqual(qc.num_qubits, n_qubit)

    def test_get_parameters_and_shot(self):
        """Test `get_parameters_and_shot` method."""

        for n_qubit in range(1, 11):
            q = np.random.uniform(0, 5, size=3 * n_qubit).reshape((n_qubit, 3))
            q /= q.sum(axis=1)[:, np.newaxis]

            angles = np.array([0.0, 0.0, 0.5 * np.pi, 0.0, 0.5 * np.pi, 0.5 * np.pi])

            cs_implementation = RandomizedPMs(n_qubit=n_qubit, bias=q, angles=angles)

            summed_shots = 0
            for shots in cs_implementation.distribute_shots(1000).values():
                summed_shots += shots

            self.assertEqual(summed_shots, 1000)

    # TODO: write a unittest for each public method of RandomizedPMs

    # TODO: write a unittest to assert the correct handling of invalid inputs (i.e. verify that
    # errors are raised properly)