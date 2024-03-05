"""Tests for Base POVM utils"""

from unittest import TestCase
import numpy as np

from povms.base_povm import Povm


class TestBasePovm(TestCase):
    """Test that we can create valid POVM and get warnings if invalid."""

    def test_random_operators(self):
        """Test"""

        ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))

        while np.abs(ops[0, 0, 0].imag) < 1e-6:
            ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))

        povm1 = Povm(povm_ops=ops)

        with self.assertRaises(ValueError):
            povm1.check_validity()

    # TODO
    def test_build_from_vectors(self):
        if True:
            self.assertTrue(True)
