# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the ProductPOVM class."""

from unittest import TestCase

import numpy as np
from povm_toolbox.quantum_info.multi_qubit_povm import MultiQubitPOVM
from povm_toolbox.quantum_info.product_povm import ProductPOVM
from povm_toolbox.quantum_info.single_qubit_povm import SingleQubitPOVM
from qiskit.quantum_info import DensityMatrix, Operator, random_density_matrix


class TestProductPOVM(TestCase):
    """Test that we can create valid product POVM and get warnings if invalid."""

    def setUp(self) -> None:
        super().setUp()

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

    def test_invalid_operators(self):
        """Test that errors are correctly raised if invalid operators are supplied."""
        # create two valid POVMs
        mq_povm1 = MultiQubitPOVM([Operator(np.eye(2))])
        mq_povm2 = MultiQubitPOVM([Operator(np.eye(2))])
        with self.subTest("Non Hermitian operators") and self.assertRaises(ValueError):
            ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))
            while np.abs(ops[0, 0, 0].imag) < 1e-6:
                ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(
                    -1, 1, (6, 2, 2)
                )
            # artificially make the 2nd povm invalid and bypass the private `check_validity` method
            mq_povm2._operators = [Operator(op) for op in ops]
            _ = ProductPOVM({(0,): mq_povm1, (1,): mq_povm2})
        with self.subTest("Operators with negative eigenvalues") and self.assertRaises(ValueError):
            op = np.array([[-0.5, 0], [0, 0]])
            # artificially make the 2nd povm invalid and bypass the private `check_validity` method
            mq_povm2._operators = [Operator(op), Operator(np.eye(2) - op)]
            _ = ProductPOVM({(0,): mq_povm1, (1,): mq_povm2})
        with self.subTest("Operators not summing up to identity") and self.assertRaises(ValueError):
            # artificially make the 2nd povm invalid and bypass the private `check_validity` method
            mq_povm2._operators = [0.9 * Operator.from_label("0"), Operator.from_label("1")]
            _ = ProductPOVM({(0,): mq_povm1, (1,): mq_povm2})

    def test_init(self):
        """Test the ``__init__`` method."""
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
            self.assertEqual(product.num_outcomes, 8)
            self.assertEqual(product.num_subsystems, 3)
        with self.subTest("MultiQubitPOVM objects"):
            povms = {(0, 1): mqp, (2, 3): mqp}
            product = ProductPOVM(povms)
            self.assertEqual(product.dimension, 16)
            self.assertEqual(product.num_outcomes, 16)
            self.assertEqual(product.num_subsystems, 4)
        with self.subTest("SingleQubitPOVM + MultiQubitPOVM objects"):
            povms = {(0,): sqp, (1,): sqp, (2, 3): mqp}
            product = ProductPOVM(povms)
            self.assertEqual(product.dimension, 16)
            self.assertEqual(product.num_outcomes, 16)
            self.assertEqual(product.num_subsystems, 4)
        with self.subTest("Invalid POVM subsystem indices"), self.assertRaises(ValueError):
            _ = ProductPOVM({(0, 0): mqp})
        with self.subTest("Duplicate POVM subsystem indices"), self.assertRaises(ValueError):
            _ = ProductPOVM({(0,): sqp, (0, 1): mqp})
        with self.subTest("Mismatching POVM size: SingleQubitPOVM"), self.assertRaises(ValueError):
            _ = ProductPOVM({(0, 1): sqp})
        with self.subTest("Mismatching POVM size: MultiQubitPOVM"), self.assertRaises(ValueError):
            _ = ProductPOVM({(0,): mqp})
        with self.subTest("Invalid type of internal frame."), self.assertRaises(TypeError):
            _ = ProductPOVM({(0,): ProductPOVM({(0,): sqp})})

    def test_num_outcomes(self):
        """Test the number of outcomes, with both `num_outcomes` attribute and `__len__` method."""
        for num_outcomes in range(1, 10):
            povm = SingleQubitPOVM(num_outcomes * [Operator(1.0 / num_outcomes * np.eye(2))])
            for num_qubits in range(1, 4):
                prod_povm = ProductPOVM.from_list(num_qubits * [povm])
                self.assertEqual(num_outcomes**num_qubits, prod_povm.num_outcomes)
                self.assertEqual(num_outcomes**num_qubits, len(prod_povm))
                self.assertEqual(num_outcomes**num_qubits, prod_povm.num_operators)

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
            self.assertEqual(expected, product._frames)
        with self.subTest("MultiQubitPOVM objects"):
            expected = {(0, 1): mqp, (2, 3): mqp}
            product = ProductPOVM.from_list([mqp, mqp])
            self.assertEqual(expected, product._frames)
        with self.subTest("SingleQubitPOVM + MultiQubitPOVM objects"):
            expected = {(0,): sqp, (1,): sqp, (2, 3): mqp}
            product = ProductPOVM.from_list([sqp, sqp, mqp])
            self.assertEqual(expected, product._frames)
        with self.subTest("SingleQubitPOVM + MultiQubitPOVM objects - interleaved"):
            expected = {(0,): sqp, (1, 2): mqp, (3,): sqp}
            product = ProductPOVM.from_list([sqp, mqp, sqp])
            self.assertEqual(expected, product._frames)

    def test_get_prob(self):
        """Test if we can build a LB Classical Shadow POVM from the generic class"""

        with self.subTest("Product of single-qubit POVMs test"):
            checks = np.load("test/quantum_info/probabilities_ProdOfSingleQubitPOVMs.npz")

            seed = 14
            for num_qubits in range(1, 4):
                rng = np.random.RandomState(seed)
                q = rng.uniform(0, 5, size=3 * num_qubits).reshape((num_qubits, 3))
                q /= q.sum(axis=1)[:, np.newaxis]

                povm_list = []
                for i in range(num_qubits):
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
                rho = random_density_matrix(dims=2**num_qubits, seed=seed)
                p = prod_povm.get_prob(rho)
                self.assertTrue(np.allclose(a=np.array(checks[f"n_{num_qubits}"]), b=np.array(p)))
                if num_qubits >= 2:
                    for n_idx in range(2, 5):
                        outcome_idx = np.random.randint(low=0, high=6, size=(n_idx, num_qubits))
                        outcome_idx = set({tuple(idx) for idx in outcome_idx})
                        # TODO: correct bug if two index tuples are the same in sequence. For now
                        # we avoid that by skipping the test when it's the case.
                        if len(outcome_idx) != len(set(outcome_idx)):
                            break
                        p = prod_povm.get_prob(rho, outcome_idx)
                        check = checks[f"n_{num_qubits}"]
                        for idx in outcome_idx:
                            self.assertTrue(np.allclose(p[idx], check[idx]))

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
            self.assertTrue(
                np.allclose(prod_povm2.get_prob(rho1, (3, 0, 0)), npzfile["prob_2_1"][3, 0, 0])
            )

            rho2 = DensityMatrix(Operator.from_label("101+"))
            self.assertTrue(np.allclose(prod_povm1.get_prob(rho2), npzfile["prob_1_2"]))
            p = prod_povm1.get_prob(rho2, {(0, 2, 1), (1, 2, 0)})
            self.assertTrue(np.allclose(p[(0, 2, 1)], npzfile["prob_1_2"][0, 2, 1]))
            self.assertTrue(np.allclose(p[(1, 2, 0)], npzfile["prob_1_2"][1, 2, 0]))
            self.assertTrue(np.allclose(prod_povm2.get_prob(rho2), npzfile["prob_2_2"]))

            rho3_vec = np.zeros(2**4)
            rho3_vec[0] += 1.0
            rho3_vec[6] += 1.0
            rho3 = DensityMatrix(0.5 * np.outer(rho3_vec, rho3_vec.conj()))
            self.assertTrue(np.allclose(prod_povm1.get_prob(rho3), npzfile["prob_1_3"]))
            self.assertTrue(np.allclose(prod_povm2.get_prob(rho3), npzfile["prob_2_3"]))

    def test_informationally_complete(self):
        """Test whether a POVM is informationally complete or not."""
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

        coef = 1.0 / 2.0
        non_ic_povm = MultiQubitPOVM(
            [
                coef * Operator.from_label("0"),
                coef * Operator.from_label("1"),
                coef * Operator.from_label("+"),
                coef * Operator.from_label("-"),
            ]
        )

        self.assertTrue(ProductPOVM.from_list([sic_povm, cs_povm]).informationally_complete)
        self.assertFalse(ProductPOVM.from_list([sic_povm, non_ic_povm]).informationally_complete)
        self.assertFalse(ProductPOVM.from_list([non_ic_povm, non_ic_povm]).informationally_complete)

        self.assertTrue(
            ProductPOVM.from_list([sic_povm, cs_povm, cs_povm]).informationally_complete
        )
        self.assertFalse(
            ProductPOVM.from_list([sic_povm, cs_povm, non_ic_povm]).informationally_complete
        )

    def test_repr(self):
        """Test the ``__repr__`` method."""
        povm_1 = SingleQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
        povm_2 = MultiQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
        prod_povm = ProductPOVM.from_list([povm_1])
        self.assertEqual(
            prod_povm.__repr__(),
            (
                "ProductPOVM(num_subsystems=1)<2>:\n   (0,): SingleQubitPOVM<2>"
                f" at {hex(id(povm_1))}"
            ),
        )
        prod_povm = ProductPOVM.from_list([povm_1, povm_2])
        self.assertEqual(
            prod_povm.__repr__(),
            (
                "ProductPOVM(num_subsystems=2)<2,2>:\n   (0,): SingleQubitPOVM<2> at "
                f"{hex(id(povm_1))}\n   (1,): MultiQubitPOVM<2> at {hex(id(povm_2))}"
            ),
        )

    def test_errors_analysis(self):
        sq_povm = SingleQubitPOVM(
            [
                1 / 3 * Operator.from_label("0"),
                1 / 3 * Operator.from_label("1"),
                1 / 3 * Operator.from_label("+"),
                1 / 3 * Operator.from_label("-"),
                1 / 3 * Operator.from_label("r"),
                1 / 3 * Operator.from_label("l"),
            ]
        )
        prod_povm = ProductPOVM.from_list(3 * [sq_povm])
        with self.subTest("Test mismatch in number of qubits.") and self.assertRaises(ValueError):
            observable = Operator.from_label("XX")
            _ = prod_povm.analysis(observable)
        with self.subTest("Test invalid type for ``frame_op_idx``.") and self.assertRaises(
            TypeError
        ):
            observable = Operator.from_label("XXX")
            _ = prod_povm.analysis(observable, frame_op_idx=0)
        with self.subTest("Test warning for non-real frame coefficients.") and self.assertWarns(
            Warning
        ):
            observable = 1.0j * Operator.from_label("XXX")
            _ = prod_povm.analysis(observable)
        with self.subTest("Test invalid shape for ``frame_op_idx``.") and self.assertRaises(
            IndexError
        ):
            observable = Operator.from_label("ZZZ")
            _ = prod_povm.analysis(observable, frame_op_idx=(0, 0))
        with self.subTest(
            "Test invalid ``frame_op_idx`` argument (out of range)."
        ) and self.assertRaises(IndexError):
            observable = Operator.from_label("ZZZ")
            _ = prod_povm.analysis(observable, frame_op_idx=(0, 0, 6))
        with self.subTest(
            "Test invalid ``frame_op_idx`` argument (negative out of range)."
        ) and self.assertRaises(IndexError):
            observable = Operator.from_label("ZZZ")
            _ = prod_povm.analysis(observable, frame_op_idx=(0, 0, -10))
