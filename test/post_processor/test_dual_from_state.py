# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the `dual_from_state` function."""

from unittest import TestCase

import numpy as np
from povm_toolbox.post_processor import dual_from_state
from povm_toolbox.quantum_info import MultiQubitPOVM, ProductPOVM, SingleQubitPOVM
from povm_toolbox.quantum_info.multi_qubit_dual import MultiQubitDual
from qiskit.quantum_info import Operator, random_density_matrix, random_hermitian


class TestDualFromState(TestCase):
    """Test that we can construct optimal dual of a POVM from a state."""

    SEED = 28

    def setUp(self) -> None:
        super().setUp()
        self.povm = MultiQubitPOVM(
            [
                0.3 * Operator.from_label("0"),
                0.3 * Operator.from_label("1"),
                0.6 * Operator.from_label("+"),
                0.6 * Operator.from_label("-"),
                0.1 * Operator.from_label("r"),
                0.1 * Operator.from_label("l"),
            ]
        )

    def test_not_implemented(self):
        """Test that errors are correctly raised."""
        prod_povm: ProductPOVM = ProductPOVM.from_list([self.povm, self.povm])
        state = random_density_matrix(4, seed=self.SEED)
        with self.assertRaises(NotImplementedError):
            _ = dual_from_state(prod_povm, state)

    def test_implemented(self):
        """Test that the method constructs a valid dual."""
        joint_povm: MultiQubitPOVM | SingleQubitPOVM = self.povm
        state = random_density_matrix(2, seed=self.SEED)
        dual = dual_from_state(joint_povm, state)
        self.assertTrue(dual.is_dual_to(joint_povm))

    def test_optimal_dual(self):
        """Test that the method constructs a valid dual."""
        joint_povm: MultiQubitPOVM | SingleQubitPOVM = self.povm
        state = random_density_matrix(2, seed=self.SEED)
        obs = random_hermitian(2, seed=self.SEED)
        probabilities = joint_povm.get_prob(state)
        exact_exp_val = state.expectation_value(obs).real

        with self.subTest("Test canonical dual."):
            canonical_dual = MultiQubitDual.build_dual_from_frame(joint_povm)
            canonical_weights = canonical_dual.get_omegas(obs)
            exp_val_canonical = np.dot(probabilities, canonical_weights)
            var_canonical = (
                np.dot(probabilities, np.power(canonical_weights, 2)) - exp_val_canonical**2
            )
            self.assertTrue(canonical_dual.is_dual_to(joint_povm))
            self.assertAlmostEqual(exp_val_canonical, exact_exp_val)
            self.assertAlmostEqual(
                var_canonical,
                np.dot(probabilities, np.power(canonical_weights, 2)) - exact_exp_val**2,
            )

        with self.subTest("Test optimal dual."):
            optimal_dual = dual_from_state(joint_povm, state)
            optimal_weights = optimal_dual.get_omegas(obs)
            exp_val_optimal = np.dot(probabilities, optimal_weights)
            var_optimal = np.dot(probabilities, np.power(optimal_weights, 2)) - exp_val_optimal**2
            self.assertTrue(optimal_dual.is_dual_to(joint_povm))
            self.assertAlmostEqual(exp_val_optimal, exact_exp_val)
            self.assertAlmostEqual(
                var_optimal, np.dot(probabilities, np.power(optimal_weights, 2)) - exact_exp_val**2
            )
