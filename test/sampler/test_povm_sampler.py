"""Tests for the classes in the ``povm_sampler.py`` file."""

from unittest import TestCase

from povms.library.pm_sim_implementation import ClassicalShadows
from povms.sampler.job import POVMSamplerJob
from povms.sampler.povm_sampler import POVMSampler
from qiskit.circuit.random import random_circuit
from qiskit.primitives import BaseSamplerV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler


class TestPOVMSampler(TestCase):
    """Tests for the ``POVMSampler`` class."""

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.backend = AerSimulator()
        self.sampler = Sampler(backend=self.backend)

    def test_initialization(self):
        povm_sampler = POVMSampler(sampler=self.sampler)
        self.assertIsInstance(povm_sampler.sampler, BaseSamplerV2)

    def test_run(self):
        povm_sampler = POVMSampler(sampler=self.sampler)
        n_qubit = 2
        qc_random = random_circuit(num_qubits=n_qubit, depth=3, measure=False)
        qc_random.draw()
        cs_implementation = ClassicalShadows(n_qubit=n_qubit)
        cs_shots = 4096
        cs_job = povm_sampler.run(cs_implementation, qc_random, shots=cs_shots)
        self.assertIsInstance(cs_job, POVMSamplerJob)
