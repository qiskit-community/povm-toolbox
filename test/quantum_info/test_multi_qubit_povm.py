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
from povm_toolbox.quantum_info.multi_qubit_povm import MultiQubitPOVM
from povm_toolbox.quantum_info.single_qubit_povm import SingleQubitPOVM
from qiskit.quantum_info import Operator


class TestMultiQubitPOVM(TestCase):
    """Test that we can create valid POVM and get warnings if invalid."""

    def test_invalid_operators(self):
        """Test that errors are correctly raised if invalid operators are supplied."""
        with self.subTest("Operators with negative eigenvalues") and self.assertRaises(ValueError):
            op = np.array([[-0.5, 0], [0, 0]])
            _ = MultiQubitPOVM(list_operators=[Operator(op), Operator(np.eye(2) - op)])
        with self.subTest("Operators not summing up to identity") and self.assertRaises(ValueError):
            _ = MultiQubitPOVM(
                list_operators=[0.9 * Operator.from_label("0"), Operator.from_label("1")]
            )

    def test_num_outcomes(self):
        """Test the number of outcomes, with both `num_outcomes` attribute and `__len__` method."""
        for n in range(1, 10):
            for dim in range(1, 10):
                povm = MultiQubitPOVM(n * [Operator(1.0 / n * np.eye(dim))])
                self.assertEqual(n, povm.num_outcomes)
                self.assertEqual(n, len(povm))
                self.assertEqual(n, len(povm.operators))

    def test_informationally_complete(self):
        """Test whether a POVM is informationally complete or not."""
        with self.subTest("SIC-POVM"):
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
            self.assertTrue(sic_povm.informationally_complete)

        with self.subTest("CS-POVM"):
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
            self.assertTrue(cs_povm.informationally_complete)

        with self.subTest("Non IC-POVM"):
            coef = 1.0 / 2.0
            povm = MultiQubitPOVM(
                [
                    coef * Operator.from_label("0"),
                    coef * Operator.from_label("1"),
                    coef * Operator.from_label("+"),
                    coef * Operator.from_label("-"),
                ]
            )
            self.assertFalse(povm.informationally_complete)

    def test_repr(self):
        """Test the ``__repr__`` method."""
        with self.subTest("Single-qubit case."):
            povm = MultiQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
            self.assertEqual(povm.__repr__(), f"MultiQubitPOVM<2> at {hex(id(povm))}")
            povm = MultiQubitPOVM(
                2 * [0.5 * Operator.from_label("0"), 0.5 * Operator.from_label("1")]
            )
            self.assertEqual(povm.__repr__(), f"MultiQubitPOVM<4> at {hex(id(povm))}")
            povm = SingleQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
            self.assertEqual(povm.__repr__(), f"SingleQubitPOVM<2> at {hex(id(povm))}")
        with self.subTest("Multi-qubit case."):
            povm = MultiQubitPOVM(
                [
                    Operator.from_label("00"),
                    Operator.from_label("01"),
                    Operator.from_label("10"),
                    Operator.from_label("11"),
                ]
            )
            self.assertEqual(povm.__repr__(), f"MultiQubitPOVM(num_qubits=2)<4> at {hex(id(povm))}")

    def test_draw_bloch(self):
        with self.assertRaises(NotImplementedError):
            povm = MultiQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
            povm.draw_bloch()
