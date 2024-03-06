"""Tests for Base POVM utils"""

from unittest import TestCase
import numpy as np

from povms.multi_qubit_povm import MultiQubitPOVM


class TestMultiQubitPOVM(TestCase):
    """Test that we can create valid POVM and get warnings if invalid."""

    def test_random_operators(self):
        """Test"""

        ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))

        while np.abs(ops[0, 0, 0].imag) < 1e-6:
            ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))

        with self.assertRaises(ValueError):
            povm1 = MultiQubitPOVM(povm_ops=ops)
            print(povm1[0])

    # TODO
    def test_build_from_vectors(self):
        if True:
            self.assertTrue(True)
