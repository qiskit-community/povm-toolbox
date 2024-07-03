# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the utility functions."""

from unittest import TestCase

import numpy as np
from povm_toolbox.utilities import double_ket_to_matrix, matrix_to_double_ket


class TestUtilities(TestCase):
    def test_matrix_to_double_ket(self):
        """Test the ``matrix_to_double_ket`` utility function."""
        matrix = np.array([[0.5, 1, 2 + 1.2j], [3, 4, 5], [6, 7, 8]])
        double_ket = np.array([0.5, 3, 6, 1, 4, 7, 2 + 1.2j, 5, 8])
        self.assertTrue(np.all(matrix_to_double_ket(matrix) == double_ket))

    def test_double_ket_to_matrix(self):
        """Test the ``double_ket_to_matrix`` utility function."""
        double_ket = np.array([0.5, 3, 6, 1, 4, 7, 2 + 1.2j, 5, 8])
        matrix = np.array([[0.5, 1, 2 + 1.2j], [3, 4, 5], [6, 7, 8]])
        self.assertTrue(np.all(matrix == double_ket_to_matrix(double_ket)))
