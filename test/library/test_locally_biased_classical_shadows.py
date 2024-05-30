# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the LocallyBiasedClassicalShadows class."""

from unittest import TestCase

import numpy as np
from povm_toolbox.library import LocallyBiasedClassicalShadows
from povm_toolbox.quantum_info.single_qubit_povm import SingleQubitPOVM
from qiskit.quantum_info import Operator


class TestRandomizedPMs(TestCase):
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

    def test_qc_build(self):
        """Test if we can build a LB Classical Shadow POVM from the generic class"""

        for n_qubit in range(1, 11):
            q = np.random.uniform(0, 5, size=3 * n_qubit).reshape((n_qubit, 3))
            q /= q.sum(axis=1)[:, np.newaxis]

            cs_implementation = LocallyBiasedClassicalShadows(n_qubit=n_qubit, bias=q)
            self.assertEqual(n_qubit, cs_implementation.n_qubit)
            cs_povm = cs_implementation.definition()
            for i in range(n_qubit):
                sqpovm = SingleQubitPOVM(
                    [
                        q[i, 0] * Operator.from_label("0"),
                        q[i, 0] * Operator.from_label("1"),
                        q[i, 1] * Operator.from_label("+"),
                        q[i, 1] * Operator.from_label("-"),
                        q[i, 2] * Operator.from_label("r"),
                        q[i, 2] * Operator.from_label("l"),
                    ]
                )
                self.assertEqual(cs_povm._povms[(i,)].n_outcomes, sqpovm.n_outcomes)
                for k in range(sqpovm.n_outcomes):
                    self.assertTrue(np.allclose(cs_povm._povms[(i,)][k], sqpovm[k]))

    def test_kwargs(self):
        """Test if we can build a copy of a :class:`.LocallyBiasedClassicalShadows` from the `.kwargs` attribute."""

        lbcs = LocallyBiasedClassicalShadows(n_qubit=5, bias=np.array([0.2, 0.7, 0.1]), seed_rng=34)
        lbcs_copy = LocallyBiasedClassicalShadows(**lbcs.kwargs)
        self.assertIsInstance(lbcs_copy, LocallyBiasedClassicalShadows)
        self.assertEqual(lbcs.kwargs, lbcs_copy.kwargs)
