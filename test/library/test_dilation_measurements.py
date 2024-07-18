# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the DilationMeasurements class."""

from unittest import TestCase

import numpy as np
from povm_toolbox.library import DilationMeasurements


class TestDilationMeasurements(TestCase):
    SEED = 9128346

    def setUp(self) -> None:
        super().setUp()

    def test_init_errors(self):
        """Test that the ``__init__`` method raises errors correctly."""
        # Sanity check
        measurement = DilationMeasurements(1, parameters=np.random.uniform(0, 1, size=8))
        self.assertIsInstance(measurement, DilationMeasurements)
        with self.subTest(
            "Test invalid shape for ``parameters``, not enough parameters."
        ) and self.assertRaises(ValueError):
            DilationMeasurements(1, parameters=np.ones(7))
        with self.subTest(
            "Test invalid shape for ``parameters``, number of qubits not matching."
        ) and self.assertRaises(ValueError):
            DilationMeasurements(1, parameters=np.ones((2, 8)))
        with self.subTest(
            "Test invalid shape for ``parameters``, too many dimensions."
        ) and self.assertRaises(ValueError):
            DilationMeasurements(1, parameters=np.ones((1, 1, 8)))

    def test_repr(self):
        """Test that the ``__repr__`` method works correctly."""
        mub_str = (
            "DilationMeasurements(num_qubits=1, parameters=array"
            "([[0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]))"
        )
        povm = DilationMeasurements(1, parameters=0.1 * np.arange(8))
        self.assertEqual(povm.__repr__(), mub_str)

    def test_to_sampler_pub(self):
        # TODO
        return

    def test_definition(self):
        # TODO
        return

    def test_compose_circuit(self):
        # TODO
        return
