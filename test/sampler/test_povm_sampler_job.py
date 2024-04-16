# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the classes in the ``povm_sampler.py`` file."""

from unittest import TestCase

from povm_toolbox.library.pm_sim_implementation import ClassicalShadows
from povm_toolbox.sampler.job import POVMSamplerJob
from povm_toolbox.sampler.povm_sampler import POVMSampler
from povm_toolbox.sampler.result import POVMPubResult
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler


class TestPOVMSamplerJob(TestCase):
    """Tests for the ``POVMSampler`` class."""

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.backend = AerSimulator()
        self.sampler = Sampler(backend=self.backend)

    def test_initialization(self):
        povm_sampler = POVMSampler(sampler=self.sampler)
        n_qubit = 2
        qc_random = random_circuit(num_qubits=n_qubit, depth=3, measure=False, seed=42)
        cs_implementation = ClassicalShadows(n_qubit=n_qubit)
        cs_shots = 4096
        cs_job = povm_sampler.run([qc_random], shots=cs_shots, povm=cs_implementation)
        self.assertIsInstance(cs_job, POVMSamplerJob)

    def test_result(self):
        povm_sampler = POVMSampler(sampler=self.sampler)
        n_qubit = 2
        qc_random = random_circuit(num_qubits=n_qubit, depth=3, measure=False, seed=42)
        cs_implementation = ClassicalShadows(n_qubit=n_qubit)
        cs_shots = 4096
        cs_job = povm_sampler.run([qc_random], shots=cs_shots, povm=cs_implementation)
        result = cs_job.result()[0]
        self.assertIsInstance(result, POVMPubResult)
