"""Tests for the ProductPOVM class."""

from unittest import TestCase

import numpy as np
from povms.quantum_info.product_povm import ProductPOVM
from povms.quantum_info.single_qubit_povm import SingleQubitPOVM
from qiskit.quantum_info import Operator, random_density_matrix


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
        """Test"""
        if True:
            self.assertTrue(True)

    # TODO
    def test_clean_povm(self):
        """Test"""
        if True:
            self.assertTrue(True)

    def test_dimension_after_clean_povm(self):
        """Test the number of outcomes is updated after a cleaning process.

        In particular, if we construct a ProductPOVM with a LocalPOVM that is subsequently cleaned,
        the ProductPOVM is also cleaned. We check that the number of outcomes of the ProductPOVM is updated.
        (Is this really what we want ? Or should the product POVM have a copy of the local POVM ?)"""

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

        prod_povm = ProductPOVM(5 * [sqpovm])
        self.assertEqual(prod_povm.n_outcomes, n_ops**5)
        self.assertEqual(len(prod_povm), n_ops**5)

        sqpovm = SingleQubitPOVM.clean_povm_operators(sqpovm)

        self.assertEqual(sqpovm.n_outcomes, 6)
        self.assertEqual(len(sqpovm), 6)

        n_outcome_check = 1
        for povm in prod_povm._povm_list:
            n_outcome_check *= povm.n_outcomes

        self.assertEqual(prod_povm.n_outcomes, n_outcome_check)
        self.assertEqual(len(prod_povm), n_outcome_check)

    def test_get_prob(self):
        """Test if we can build a LB Classical Shadow POVM from the generic class"""

        # fmt: off
        checks = [[0.05161737, 0.18659108, 0.23876207, 0.11959349, 0.27488718, 0.1285488],
        [0.00059614, 0.00016942, 0.01016329, 0.01930394, 0.04579875, 0.04530444,
        0.00043272, 0.00030468, 0.01186327, 0.01651997, 0.05578419, 0.03196765,
        0.00075619, 0.00055843, 0.01369828, 0.03690285, 0.08847944, 0.06796295,
        0.00079159, 0.00015481, 0.01943799, 0.01698986, 0.06433969, 0.04828349,
        0.0014374,  0.00027004, 0.02443322, 0.04128798, 0.12173772, 0.08145107,
        0.00030509, 0.00053292, 0.01287153, 0.01938431, 0.05030575, 0.04941894],
        [8.20916330e-05, 1.02162504e-04, 6.98883197e-05, 4.44038492e-05,
        9.70244628e-05, 9.65958512e-05, 9.14297604e-05, 6.13364834e-05,
        6.85749152e-05, 2.61854286e-05, 5.94217454e-05, 1.01110054e-04,
        3.97906200e-03, 3.63850156e-03, 2.98300782e-03, 1.74213907e-03,
        4.59142605e-03, 3.41336058e-03, 2.69996784e-03, 2.65475481e-03,
        2.34659483e-03, 9.74920097e-04, 1.43036108e-03, 4.19655780e-03,
        1.26491934e-02, 8.48741707e-03, 8.15222604e-03, 4.95873633e-03,
        7.04382408e-03, 1.51672222e-02, 8.00021607e-03, 1.09693050e-02,
        8.32518909e-03, 3.44153566e-03, 1.15736026e-02, 8.36019486e-03,
        4.39752431e-05, 7.06941911e-05, 3.15393409e-05, 3.95896909e-05,
        2.39996465e-05, 9.64987708e-05, 4.57780363e-05, 6.51994013e-05,
        2.23718885e-05, 4.64670112e-05, 7.71306687e-05, 3.94880770e-05,
        2.27785339e-03, 2.08354638e-03, 1.12431881e-03, 1.58104155e-03,
        1.57050821e-03, 3.01259428e-03, 1.17684934e-03, 3.14714790e-03,
        9.50783878e-04, 1.73137584e-03, 2.32210918e-03, 2.22168950e-03,
        4.47397037e-03, 7.88251415e-03, 4.12940984e-03, 3.53527244e-03,
        4.85611351e-03, 8.12848730e-03, 6.20685798e-03, 8.28910814e-03,
        2.28613945e-03, 6.70565546e-03, 7.17860595e-03, 8.05423259e-03,
        8.79140860e-05, 1.35032304e-04, 7.55365578e-05, 6.27562747e-05,
        8.65192753e-05, 1.47760133e-04, 1.31291710e-04, 7.38821586e-05,
        6.21610192e-05, 6.51075827e-05, 8.96366264e-05, 1.25966830e-04,
        5.80697006e-03, 4.43578484e-03, 3.30711129e-03, 3.04643221e-03,
        3.73523127e-03, 7.02819286e-03, 2.63050281e-03, 3.60556328e-03,
        1.99302034e-03, 1.87518871e-03, 3.04520370e-03, 3.50785989e-03,
        1.30959835e-02, 8.31230629e-03, 6.74230627e-03, 6.53717782e-03,
        8.51809250e-03, 1.39784434e-02, 1.29899655e-02, 1.65489526e-02,
        9.64399398e-03, 8.67888919e-03, 1.24448290e-02, 1.85956389e-02,
        1.01738147e-04, 1.25009502e-04, 7.70489818e-05, 6.36017579e-05,
        9.55467306e-05, 1.42727166e-04, 7.51206792e-05, 1.16475641e-04,
        7.46573501e-05, 4.41891481e-05, 1.15789819e-04, 8.55459027e-05,
        3.60579571e-03, 4.17233811e-03, 2.87186047e-03, 1.95288756e-03,
        5.53464698e-03, 2.63887216e-03, 3.20169561e-03, 5.12269142e-03,
        2.96748359e-03, 2.19610332e-03, 2.59992995e-03, 6.14761006e-03,
        1.26637271e-02, 1.63142579e-02, 1.17339164e-02, 6.24102197e-03,
        9.38391165e-03, 2.10671091e-02, 8.38284390e-03, 1.24229802e-02,
        6.31945507e-03, 6.58632175e-03, 1.57655807e-02, 6.09786443e-03,
        9.88215345e-05, 1.67688756e-04, 8.26060009e-05, 8.27093599e-05,
        1.02292128e-04, 1.77765661e-04, 1.36091190e-04, 6.37239669e-05,
        7.71516563e-05, 4.67929564e-05, 1.09145728e-04, 1.00826618e-04,
        6.30304739e-03, 4.12524258e-03, 2.97518916e-03, 3.49344107e-03,
        4.69853675e-03, 6.25985374e-03, 2.73900249e-03, 4.78208838e-03,
        3.17405930e-03, 1.49124594e-03, 3.43993956e-03, 4.46347039e-03,
        1.37636165e-02, 1.96948211e-02, 1.14021107e-02, 9.35203608e-03,
        1.06211778e-02, 2.45380502e-02, 1.41914899e-02, 7.84377787e-03,
        7.60938527e-03, 6.05901097e-03, 1.45403702e-02, 8.61501484e-03,
        1.14688574e-04, 1.25065795e-04, 8.91744977e-05, 5.95442631e-05,
        1.02677421e-04, 1.49264365e-04, 9.62874692e-05, 1.50580467e-04,
        7.68781932e-05, 7.62530870e-05, 1.22122958e-04, 1.37293998e-04,
        4.29382569e-03, 5.56576520e-03, 3.98108503e-03, 2.13478323e-03,
        5.73747384e-03, 4.62330893e-03, 3.82687495e-03, 5.04416349e-03,
        2.41046622e-03, 3.09220657e-03, 2.91534073e-03, 6.40663858e-03,
        1.52366147e-02, 8.02971609e-03, 9.39838415e-03, 5.03363596e-03,
        9.53286302e-03, 1.49161635e-02, 9.86997664e-03, 2.47727668e-02,
        1.03622341e-02, 1.11265333e-02, 1.72188532e-02, 1.91848823e-02]]
        # fmt: on

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

            prod_povm = ProductPOVM(povm_list=povm_list)
            rho = random_density_matrix(dims=2**n_qubit, seed=seed)
            p = prod_povm.get_prob(rho)
            self.assertTrue(np.allclose(a=np.array(checks[n_qubit - 1]), b=np.array(p)))

    # TODO
    def test_build_from_vectors(self):
        if True:
            self.assertTrue(True)
