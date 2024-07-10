# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the POVMPostProcessor class."""

from collections import Counter
from unittest import TestCase

import numpy as np
from povm_toolbox.library import ClassicalShadows
from povm_toolbox.sampler import POVMSampler
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorSampler as Sampler


class TestPostProcessor(TestCase):
    """Test the methods and attributes of the :class:`.POVMPostProcessor class`."""

    SEED = 42

    def setUp(self) -> None:
        super().setUp()

        # Load the circuit that was obtained through:
        #   from qiskit.circuit.random import random_circuit
        #   qc = random_circuit(num_qubits=num_qubits, depth=3, measure=False, seed=10)
        # for qiskit==1.1.1
        with open("test/sampler/random_circuits.qpy", "rb") as file:
            qc = qpy.load(file)[0]

        param = Parameter("a")
        qc.ry(param, 0)

        povm_sampler = POVMSampler(sampler=Sampler(seed=self.SEED))
        self.measurement = ClassicalShadows(num_qubits=2, seed=self.SEED)

        job = povm_sampler.run([(qc, [0.0, np.pi])], shots=10, povm=self.measurement)
        result = job.result()
        self.pub_result = result[0]

        self.samples_check = [
            [(4, 5), (2, 3), (4, 5), (4, 5), (1, 5), (4, 1), (4, 3), (4, 1), (1, 1), (2, 5)],
            [(3, 5), (4, 5), (2, 1), (4, 3), (2, 3), (0, 1), (2, 1), (1, 3), (4, 1), (3, 5)],
        ]

    def test_metadata(self):
        """Test that ``metadata`` property works correctly."""
        metadata = self.pub_result.metadata
        with self.subTest("Test `composed_circuit`."):
            self.assertIsInstance(metadata.composed_circuit, QuantumCircuit)
        with self.subTest("Test `povm_implementation`."):
            self.assertIs(metadata.povm_implementation, self.measurement)
        with self.subTest("Test `pvm_keys`."):
            self.assertTrue(
                np.all(
                    metadata.pvm_keys
                    == np.array(
                        [
                            [
                                [2, 2],
                                [1, 1],
                                [2, 2],
                                [2, 2],
                                [0, 2],
                                [2, 0],
                                [2, 1],
                                [2, 0],
                                [0, 0],
                                [1, 2],
                            ],
                            [
                                [1, 2],
                                [2, 2],
                                [1, 0],
                                [2, 1],
                                [1, 1],
                                [0, 0],
                                [1, 0],
                                [0, 1],
                                [2, 0],
                                [1, 2],
                            ],
                        ]
                    )
                )
            )

    def test_get_counts(self):
        """Test that the ``get_counts`` method works correctly."""
        with self.subTest("No loc"):
            counts = self.pub_result.get_counts()
            self.assertEqual(counts[0], Counter(self.samples_check[0]))
            self.assertEqual(counts[1], Counter(self.samples_check[1]))

        with self.subTest("With loc"):
            counts = self.pub_result.get_counts(loc=1)
            self.assertEqual(counts, Counter(self.samples_check[1]))

    def test_get_samples(self):
        """Test that the ``get_samples`` method works correctly."""
        with self.subTest("No loc"):
            samples = self.pub_result.get_samples()
            self.assertSequenceEqual(samples[0], self.samples_check[0])
            self.assertSequenceEqual(samples[1], self.samples_check[1])

        with self.subTest("With loc"):
            samples = self.pub_result.get_samples(loc=1)
            self.assertSequenceEqual(samples, self.samples_check[1])
