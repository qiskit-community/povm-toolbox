"""Tests for the classes in the ``povm_sampler.py`` file."""

from unittest import TestCase

from povms.library.pm_sim_implementation import ClassicalShadows
from povms.sampler.job import POVMSamplerJob
from povms.sampler.povm_sampler import POVMSampler
from povms.sampler.result import POVMSamplerResult
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
        qc_random = random_circuit(num_qubits=n_qubit, depth=3, measure=False)
        cs_implementation = ClassicalShadows(n_qubit=n_qubit)
        cs_shots = 4096
        cs_job = povm_sampler.run([qc_random], shots=cs_shots, povm=cs_implementation)
        self.assertIsInstance(cs_job, POVMSamplerJob)

    def test_result(self):
        povm_sampler = POVMSampler(sampler=self.sampler)
        n_qubit = 2
        qc_random = random_circuit(num_qubits=n_qubit, depth=3, measure=False)
        cs_implementation = ClassicalShadows(n_qubit=n_qubit)
        cs_shots = 4096
        cs_job = povm_sampler.run([qc_random], shots=cs_shots, povm=cs_implementation)
        result = cs_job.result()[0]
        self.assertIsInstance(result, POVMSamplerResult)
