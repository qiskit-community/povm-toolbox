# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the POVMSampler class."""

from unittest import TestCase

import numpy as np
from povm_toolbox.library import ClassicalShadows, LocallyBiasedClassicalShadows
from povm_toolbox.sampler import POVMSampler, POVMSamplerJob
from qiskit.circuit.random import random_circuit
from qiskit.primitives import BaseSamplerV2
from qiskit_aer.primitives import SamplerV2 as AerSampler


class TestPOVMSampler(TestCase):
    """Tests for the ``POVMSampler`` class."""

    def setUp(self) -> None:
        self.sampler = AerSampler()

    def test_initialization(self):
        povm_sampler = POVMSampler(sampler=self.sampler)
        self.assertIsInstance(povm_sampler.sampler, BaseSamplerV2)

    def test_run(self):
        povm_sampler = POVMSampler(sampler=self.sampler)
        num_qubits = 2
        qc_random1 = random_circuit(num_qubits=num_qubits, depth=3, measure=False, seed=42)
        qc_random2 = random_circuit(num_qubits=num_qubits, depth=3, measure=False, seed=43)
        cs_implementation = ClassicalShadows(num_qubits=num_qubits)
        lbcs_implementation = LocallyBiasedClassicalShadows(
            num_qubits=num_qubits, bias=np.array([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]])
        )
        cs_shots = 4096
        lbcs_shots = 2048
        cs_job1 = povm_sampler.run(
            [
                (qc_random1, None, None, cs_implementation),
                (qc_random1, None, None, lbcs_implementation),
            ],
            shots=cs_shots,
        )
        self.assertIsInstance(cs_job1, POVMSamplerJob)
        cs_job2 = povm_sampler.run(
            [qc_random1, (qc_random2, None, lbcs_shots, lbcs_implementation)],
            povm=cs_implementation,
            shots=cs_shots,
        )
        self.assertIsInstance(cs_job2, POVMSamplerJob)
