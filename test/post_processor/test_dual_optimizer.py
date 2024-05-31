# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the DUALOptimizer class."""

from unittest import TestCase

from povm_toolbox.post_processor.dual_optimizer import DUALOptimizer


class TestDualOptimizer(TestCase):
    """TODO."""

    def test_init(self):
        """TODO."""

        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        from povm_toolbox.sampler import POVMSampler
        from qiskit.primitives import StatevectorSampler as Sampler

        sampler = Sampler()
        povm_sampler = POVMSampler(sampler=sampler)

        from povm_toolbox.library import ClassicalShadows

        measurement = ClassicalShadows(n_qubit=2)

        job = povm_sampler.run([qc], shots=32, povm=measurement)
        result = job.result()
        pub_result = result[0]

        from qiskit.quantum_info import SparsePauliOp

        observable = SparsePauliOp(["II", "XX", "YY", "ZZ"], coeffs=[1, 1, -1, 1])

        post_processor = DUALOptimizer(pub_result)

        post_processor.gammas[(0, 0)] = 1.0

        self.assertEqual(post_processor.gammas, {(0, 0): 1.0})

        self.assertIsInstance(post_processor.get_expectation_value(observable), float)
