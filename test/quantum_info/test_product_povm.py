"""Tests for the ProductPOVM class."""

from unittest import TestCase

import numpy as np
from povms.quantum_info.multi_qubit_povm import MultiQubitPOVM
from povms.quantum_info.product_povm import ProductPOVM
from povms.quantum_info.single_qubit_povm import SingleQubitPOVM
from qiskit.quantum_info import DensityMatrix, Operator, random_density_matrix


class TestProductPOVM(TestCase):
    """Test that we can create valid product POVM and get warnings if invalid."""

    # TODO: write a unittest for each public method of ProductPOVM

    # TODO: write a unittest to assert the correct handling of invalid inputs (i.e. verify that
    # errors are raised properly)
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

    # TODO
    def test_random_operators(self):
        """Test"""
        if True:
            self.assertTrue(True)

    # TODO
    def test_init(self):
        """Test the ``ProductPOVM.__init__`` method."""
        sqp = SingleQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
        mqp = MultiQubitPOVM(
            [
                Operator.from_label("00"),
                Operator.from_label("01"),
                Operator.from_label("10"),
                Operator.from_label("11"),
            ]
        )
        with self.subTest("SingleQubitPOVM objects"):
            povms = {(0,): sqp, (1,): sqp, (2,): sqp}
            product = ProductPOVM(povms)
            self.assertEqual(product.dimension, 8)
            self.assertEqual(product.n_outcomes, 8)
            self.assertEqual(product.n_operators, 6)
            self.assertEqual(product.n_subsystems, 3)
        with self.subTest("MultiQubitPOVM objects"):
            povms = {(0, 1): mqp, (2, 3): mqp}
            product = ProductPOVM(povms)
            self.assertEqual(product.dimension, 16)
            self.assertEqual(product.n_outcomes, 16)
            self.assertEqual(product.n_operators, 8)
            self.assertEqual(product.n_subsystems, 4)
        with self.subTest("SingleQubitPOVM + MultiQubitPOVM objects"):
            povms = {(0,): sqp, (1,): sqp, (2, 3): mqp}
            product = ProductPOVM(povms)
            self.assertEqual(product.dimension, 16)
            self.assertEqual(product.n_outcomes, 16)
            self.assertEqual(product.n_operators, 8)
            self.assertEqual(product.n_subsystems, 4)
        with self.subTest("Invalid POVM subsystem indices"), self.assertRaises(ValueError):
            _ = ProductPOVM({(0, 0): mqp})
        with self.subTest("Duplicate POVM subsystem indices"), self.assertRaises(ValueError):
            _ = ProductPOVM({(0,): sqp, (0, 1): mqp})
        with self.subTest("Mismatching POVM size: SingleQubitPOVM"), self.assertRaises(ValueError):
            _ = ProductPOVM({(0, 1): sqp})
        with self.subTest("Mismatching POVM size: MultiQubitPOVM"), self.assertRaises(ValueError):
            _ = ProductPOVM({(0,): mqp})

    def test_from_list(self):
        """Test the ``ProductPOVM.from_list`` constructor method."""
        sqp = SingleQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
        mqp = MultiQubitPOVM(
            [
                Operator.from_label("00"),
                Operator.from_label("01"),
                Operator.from_label("10"),
                Operator.from_label("11"),
            ]
        )
        with self.subTest("SingleQubitPOVM objects"):
            expected = {(0,): sqp, (1,): sqp}
            product = ProductPOVM.from_list([sqp, sqp])
            self.assertEqual(expected, product._povms)
        with self.subTest("MultiQubitPOVM objects"):
            expected = {(0, 1): mqp, (2, 3): mqp}
            product = ProductPOVM.from_list([mqp, mqp])
            self.assertEqual(expected, product._povms)
        with self.subTest("SingleQubitPOVM + MultiQubitPOVM objects"):
            expected = {(0,): sqp, (1,): sqp, (2, 3): mqp}
            product = ProductPOVM.from_list([sqp, sqp, mqp])
            self.assertEqual(expected, product._povms)
        with self.subTest("SingleQubitPOVM + MultiQubitPOVM objects - interleaved"):
            expected = {(0,): sqp, (1, 2): mqp, (3,): sqp}
            product = ProductPOVM.from_list([sqp, mqp, sqp])
            self.assertEqual(expected, product._povms)

    # TODO
    def test_clean_povm(self):
        """Test"""
        if True:
            self.assertTrue(True)

    def test_dimension_after_clean_povm(self):
        """Test the number of outcomes is updated after a cleaning process.

        In particular, if we construct a ProductPOVM with a LocalPOVM that is subsequently cleaned,
        the ProductPOVM is also cleaned. We check that the number of outcomes of the ProductPOVM is updated.
        (Is this really what we want ? Or should the product POVM have a copy of the local POVM ?)
        """

        q = [0.51, 0.1, 0.2, 0.2]
        ops = [
            0.15 * Operator.from_label("+"),
            q[0] * Operator.from_label("0"),
            q[0] * Operator.from_label("1"),
            0.01 * Operator.from_label("+"),
            0.08 * Operator.from_label("+"),
            0.01 * Operator.from_label("l"),
            0.29 * Operator.from_label("-"),
            q[2] * Operator.from_label("r"),
            0.09 * Operator.from_label("l"),
            0.05 * Operator.from_label("+"),
            0.05 * Operator.from_label("l"),
            0.05 * Operator.from_label("l"),
        ]
        n_ops = len(ops)

        sqpovm = SingleQubitPOVM(ops)
        self.assertEqual(sqpovm.n_outcomes, n_ops)

        prod_povm = ProductPOVM.from_list(5 * [sqpovm])
        self.assertEqual(prod_povm.n_outcomes, n_ops**5)
        self.assertEqual(len(prod_povm), n_ops**5)

        sqpovm = SingleQubitPOVM.clean_povm_operators(sqpovm)

        self.assertEqual(sqpovm.n_outcomes, 6)
        self.assertEqual(len(sqpovm), 6)

        n_outcome_check = 1
        for povm in prod_povm._povms.values():
            n_outcome_check *= povm.n_outcomes

        self.assertEqual(prod_povm.n_outcomes, n_outcome_check)
        self.assertEqual(len(prod_povm), n_outcome_check)

    def test_get_prob(self):
        """Test if we can build a LB Classical Shadow POVM from the generic class"""

        with self.subTest("Product of single-qubit POVMs test"):
            checks = np.load("test/quantum_info/probabilities_ProdOfSingleQubitPOVMs.npz")

            seed = 14
            for n_qubit in range(1, 4):
                rng = np.random.RandomState(seed)
                q = rng.uniform(0, 5, size=3 * n_qubit).reshape((n_qubit, 3))
                q /= q.sum(axis=1)[:, np.newaxis]

                povm_list = []
                for i in range(n_qubit):
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
                rho = random_density_matrix(dims=2**n_qubit, seed=seed)
                p = prod_povm.get_prob(rho)
                self.assertTrue(np.allclose(a=np.array(checks[f"n_{n_qubit}"]), b=np.array(p)))

        with self.subTest("Product of single-qubit and multi-qubit POVMs test"):
            npzfile = np.load("test/quantum_info/probabilities_ProdOfMultiQubitPOVMs.npz")

            sqp1 = SingleQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
            sqp2 = SingleQubitPOVM([Operator.from_label("+"), Operator.from_label("-")])
            bell_states = (
                1.0
                / np.sqrt(2)
                * np.array(
                    [
                        [1.0, 0.0, 0.0, +1.0],
                        [1.0, 0.0, 0.0, -1.0],
                        [0.0, 1.0, +1.0, 0.0],
                        [0.0, 1.0, -1.0, 0.0],
                    ]
                )
            )
            mqp1 = MultiQubitPOVM.from_vectors(bell_states)

            prod_povm1 = ProductPOVM.from_list([sqp1, mqp1, sqp2])
            prod_povm2 = ProductPOVM({(0, 2): mqp1, (1,): sqp1, (3,): sqp2})

            rho1 = DensityMatrix(Operator.from_label("0000"))
            self.assertTrue(np.allclose(prod_povm1.get_prob(rho1), npzfile["prob_1_1"]))
            self.assertTrue(np.allclose(prod_povm2.get_prob(rho1), npzfile["prob_2_1"]))

            rho2 = DensityMatrix(Operator.from_label("101+"))
            self.assertTrue(np.allclose(prod_povm1.get_prob(rho2), npzfile["prob_1_2"]))
            self.assertTrue(np.allclose(prod_povm2.get_prob(rho2), npzfile["prob_2_2"]))

            rho3_vec = np.zeros(2**4)
            rho3_vec[0] += 1.0
            rho3_vec[6] += 1.0
            rho3 = DensityMatrix(0.5 * np.outer(rho3_vec, rho3_vec.conj()))
            self.assertTrue(np.allclose(prod_povm1.get_prob(rho3), npzfile["prob_1_3"]))
            self.assertTrue(np.allclose(prod_povm2.get_prob(rho3), npzfile["prob_2_3"]))

    # TODO
    def test_build_from_vectors(self):
        if True:
            self.assertTrue(True)
