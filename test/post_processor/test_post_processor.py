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

from povm_toolbox.library import ClassicalShadows
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.sampler import POVMSampler
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import SparsePauliOp


class TestPostProcessor(TestCase):
    """TODO."""

    def test_init(self):
        """TODO."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        sampler = Sampler()
        povm_sampler = POVMSampler(sampler=sampler)

        measurement = ClassicalShadows(num_qubits=2)

        job = povm_sampler.run([qc], shots=256, povm=measurement)
        result = job.result()
        pub_result = result[0]

        observable = SparsePauliOp(["II", "XX", "YY", "ZZ"], coeffs=[1, 1, -1, 1])

        post_processor = POVMPostProcessor(pub_result)

        exp_val, std = post_processor.get_expectation_value(observable)

        self.assertIsInstance(exp_val, float)
        self.assertIsInstance(std, float)
