"""Tests for the classes in the ``povm_sampler.py`` file."""

from unittest import TestCase

import numpy as np
from povms.library.pm_sim_implementation import ClassicalShadows, LocallyBiased
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
        qc_random1 = random_circuit(num_qubits=n_qubit, depth=3, measure=False, seed=42)
        qc_random2 = random_circuit(num_qubits=n_qubit, depth=3, measure=False, seed=42)
        cs_implementation = ClassicalShadows(n_qubit=n_qubit)
        lbcs_implementation = LocallyBiased(
            n_qubit=n_qubit, bias=np.array([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]])
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
        cs_job3 = povm_sampler.run(
            [qc_random1, (qc_random2, None, lbcs_shots, lbcs_implementation)],
            povm=cs_implementation,
            shots=cs_shots,
            multi_job=True,
        )
        self.assertIsInstance(cs_job3, list)
        self.assertIsInstance(cs_job3[0], POVMSamplerJob)
