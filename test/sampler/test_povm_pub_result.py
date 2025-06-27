# (C) Copyright IBM 2024, 2025.
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

import numpy as np
import pytest
from povm_toolbox.library import ClassicalShadows
from povm_toolbox.sampler import POVMSampler
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorSampler as Sampler


class TestPostProcessor:
    """Test the methods and attributes of the :class:`.POVMPostProcessor class`."""

    SEED = 42

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
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

    def test_metadata(self, subtests):
        """Test that ``metadata`` property works correctly."""
        metadata = self.pub_result.metadata
        with subtests.test("Test `composed_circuit`."):
            assert isinstance(metadata.composed_circuit, QuantumCircuit)

        with subtests.test("Test `povm_implementation`."):
            assert metadata.povm_implementation is self.measurement

        with subtests.test("Test `pvm_keys`."):
            assert np.all(
                metadata.pvm_keys
                == np.asarray(
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

    def test_get_counts(self, subtests):
        """Test that the ``get_counts`` method works correctly."""
        with subtests.test("No loc"):
            counts = self.pub_result.get_counts(loc=...)
            assert counts[0] == Counter(self.samples_check[0])
            assert counts[1] == Counter(self.samples_check[1])

        with subtests.test("With loc"):
            counts = self.pub_result.get_counts(loc=1)
            assert counts == Counter(self.samples_check[1])

    def test_get_samples(self, subtests):
        """Test that the ``get_samples`` method works correctly."""
        with subtests.test("No loc"):
            samples = self.pub_result.get_samples(loc=...)
            assert samples[0] == self.samples_check[0]
            assert samples[1] == self.samples_check[1]

        with subtests.test("With loc"):
            samples = self.pub_result.get_samples(loc=1)
            assert samples == self.samples_check[1]
