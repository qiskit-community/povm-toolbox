# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the MultiQubitPOVM class."""

from unittest import TestCase

import numpy as np
from povm_toolbox.library import ClassicalShadows, LocallyBiasedClassicalShadows
from povm_toolbox.post_processor import optimal_dual_from_state
from qiskit.quantum_info import DensityMatrix, random_density_matrix


class TestOptimalDualFromState(TestCase):
    """Test that we can construct optimal dual of a POVM from a state."""

    def test_not_implemented(self):
        """Test that errors are correctly raised."""
        prod_povm = ClassicalShadows(n_qubit=2).definition()
        state = DensityMatrix(np.eye(4) / 4)
        with self.assertRaises(NotImplementedError):
            _ = optimal_dual_from_state(prod_povm, state)

    def test_optimal_dual(self):
        """Test that the method constructs a valid dual."""
        povm = LocallyBiasedClassicalShadows(
            n_qubit=1, bias=np.array([0.3, 0.6, 0.1])
        ).definition()[(0,)]
        state = random_density_matrix(2, seed=12)
        dual = optimal_dual_from_state(povm, state)
        self.assertTrue(dual.is_dual_to(povm))
