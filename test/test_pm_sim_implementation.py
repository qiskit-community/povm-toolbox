"""Tests for the PMSimImplementation class."""

from unittest import TestCase

import numpy as np
from povms.pm_sim_implementation import PMSimImplementation
from povms.single_qubit_povm import SingleQubitPOVM


class TestPMSimImplementation(TestCase):
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
            np.array(
                [
                    q[0] * self.Z0,
                    q[0] * self.Z1,
                    q[1] * self.X0,
                    q[1] * self.X1,
                    q[2] * self.Y0,
                    q[2] * self.Y1,
                ]
            )
        )

        for n_qubit in range(1, 11):
            parameters = np.array(
                n_qubit * [0.0, 0.0, 0.5 * np.pi, 0.0, 0.5 * np.pi, 0.5 * np.pi, 1, 1]
            )
            cs_implementation = PMSimImplementation(n_qubit=n_qubit, parameters=parameters)
            self.assertEqual(n_qubit, cs_implementation.n_qubit)
            cs_povm = cs_implementation.to_povm()
            for i in range(n_qubit):
                self.assertEqual(cs_povm._povm_list[i].n_outcomes, sqpovm.n_outcomes)
                for k in range(sqpovm.n_outcomes):
                    self.assertAlmostEqual(cs_povm._povm_list[i][k], sqpovm[k])

    def test_LBCS_build(self):
        """Test if we can build a LB Classical Shadow POVM from the generic class"""

        for n_qubit in range(1, 11):
            q = np.random.uniform(0, 5, size=3 * n_qubit).reshape((n_qubit, 3))
            q /= q.sum(axis=1)[:, np.newaxis]

            parameters = np.array(
                n_qubit * [0.0, 0.0, 0.5 * np.pi, 0.0, 0.5 * np.pi, 0.5 * np.pi, 1, 1]
            )
            for i in range(n_qubit):
                parameters[i * 8 + 6] = q[i, 0] / q[i, 2]
                parameters[i * 8 + 7] = q[i, 1] / q[i, 2]

            cs_implementation = PMSimImplementation(n_qubit=n_qubit, parameters=parameters)
            self.assertEqual(n_qubit, cs_implementation.n_qubit)
            cs_povm = cs_implementation.to_povm()
            for i in range(n_qubit):
                sqpovm = SingleQubitPOVM(
                    np.array(
                        [
                            q[i, 0] * self.Z0,
                            q[i, 0] * self.Z1,
                            q[i, 1] * self.X0,
                            q[i, 1] * self.X1,
                            q[i, 2] * self.Y0,
                            q[i, 2] * self.Y1,
                        ]
                    )
                )
                self.assertEqual(cs_povm._povm_list[i].n_outcomes, sqpovm.n_outcomes)
                for k in range(sqpovm.n_outcomes):
                    self.assertTrue(np.allclose(cs_povm._povm_list[i][k], sqpovm[k]))

    # TODO: write a unittest for each public method of PMSimImplementation

    # TODO: write a unittest to assert the correct handling of invalid inputs (i.e. verify that
    # errors are raised properly)
