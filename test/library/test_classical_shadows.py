# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the ClassicalShadows class."""

from unittest import TestCase

import numpy as np
from povm_toolbox.library import ClassicalShadows
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.quantum_info.single_qubit_povm import SingleQubitPOVM
from povm_toolbox.sampler import POVMSampler
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Operator, SparsePauliOp


class TestClassicalShadows(TestCase):
    SEED = 147234

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

    def test_init(self):
        """Test the implementation of classical shadow POVMs."""

        qc = QuantumCircuit(2)
        qc.h(0)

        num_qubits = qc.num_qubits

        measurement = ClassicalShadows(
            num_qubits,
            seed=self.SEED,
        )
        sampler = StatevectorSampler(seed=self.SEED)
        povm_sampler = POVMSampler(sampler=sampler)

        job = povm_sampler.run([qc], shots=32, povm=measurement)
        pub_result = job.result()[0]

        post_processor = POVMPostProcessor(pub_result)

        observable = SparsePauliOp(["ZI"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 0.84375)
        self.assertAlmostEqual(std, 0.24225659134146815)
        observable = SparsePauliOp(["ZY"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, -0.8437500000000002)
        self.assertAlmostEqual(std, 0.47116314336043985)

    def test_qc_build(self):
        """Test if we can build a standard Classical Shadow POVM from the generic class"""

        q = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        sqpovm = SingleQubitPOVM(
            [
                q[0] * Operator.from_label("0"),
                q[0] * Operator.from_label("1"),
                q[1] * Operator.from_label("+"),
                q[1] * Operator.from_label("-"),
                q[2] * Operator.from_label("r"),
                q[2] * Operator.from_label("l"),
            ]
        )

        for num_qubits in range(1, 11):
            cs_implementation = ClassicalShadows(num_qubits=num_qubits)
            self.assertEqual(num_qubits, cs_implementation.num_qubits)
            cs_povm = cs_implementation.definition()
            for i in range(num_qubits):
                self.assertEqual(cs_povm._frames[(i,)].num_outcomes, sqpovm.num_outcomes)
                for k in range(sqpovm.num_outcomes):
                    self.assertAlmostEqual(cs_povm._frames[(i,)][k], sqpovm[k])

    def test_repr(self):
        """Test that the ``__repr__`` method works correctly."""
        cs_str = "ClassicalShadows(num_qubits=1)"
        povm = ClassicalShadows(
            1,
        )
        self.assertEqual(povm.__repr__(), cs_str)
