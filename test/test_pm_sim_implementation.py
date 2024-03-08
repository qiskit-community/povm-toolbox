"""Tests for the PMSimImplementation class."""

from unittest import TestCase

import numpy as np
from povms.pm_sim_implementation import PMSimImplementation
from povms.single_qubit_povm import SingleQubitPOVM
from qiskit.quantum_info import Operator


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
            parameters = np.array(
                n_qubit * [0.0, 0.0, 0.5 * np.pi, 0.0, 0.5 * np.pi, 0.5 * np.pi, 1, 1]
            )
            cs_implementation = PMSimImplementation(n_qubit=n_qubit, parameters=parameters)
            self.assertEqual(n_qubit, cs_implementation.n_qubit)
            cs_povm = cs_implementation.to_povm()
            for i in range(n_qubit):
                self.assertEqual(cs_povm._povm_list[i].n_outcomes, sqpovm.n_outcomes)
                for k in range(sqpovm.n_outcomes):
                    self.assertAlmostEqual(cs_povm[i, k], sqpovm[k])

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
                    [
                        q[i, 0] * Operator.from_label("0"),
                        q[i, 0] * Operator.from_label("1"),
                        q[i, 1] * Operator.from_label("+"),
                        q[i, 1] * Operator.from_label("-"),
                        q[i, 2] * Operator.from_label("r"),
                        q[i, 2] * Operator.from_label("l"),
                    ]
                )
                self.assertEqual(cs_povm._povm_list[i].n_outcomes, sqpovm.n_outcomes)
                for k in range(sqpovm.n_outcomes):
                    self.assertTrue(np.allclose(cs_povm._povm_list[i][k], sqpovm[k]))

    def test_qc_build(self):
        """Test if we can build a QunatumCircuit."""

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

            qc = cs_implementation._build_qc()

            self.assertEqual(qc.num_qubits, n_qubit)

    def test_get_parameters_and_shot(self):
        """Test `get_parameters_and_shot` method."""

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

            summed_shots = 0
            for _, shots in cs_implementation.get_parameter_and_shot(1000):
                summed_shots += shots

            self.assertEqual(summed_shots, 1000)

    # TODO: write a unittest for each public method of PMSimImplementation

    # TODO: write a unittest to assert the correct handling of invalid inputs (i.e. verify that
    # errors are raised properly)
