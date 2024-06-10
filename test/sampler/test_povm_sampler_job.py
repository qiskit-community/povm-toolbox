# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the POVMSamplerJob class."""

from unittest import TestCase

from povm_toolbox.library import ClassicalShadows
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.sampler import POVMPubResult, POVMSampler, POVMSamplerJob
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke


class TestPOVMSamplerJob(TestCase):
    """Tests for the ``POVMSampler`` class."""

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.backend = AerSimulator()
        self.sampler = Sampler(backend=self.backend)
        self.pm = generate_preset_pass_manager(optimization_level=1, backend=self.backend)

    def test_initialization(self):
        povm_sampler = POVMSampler(sampler=self.sampler)
        n_qubit = 2
        qc_random = self.pm.run(random_circuit(num_qubits=n_qubit, depth=3, measure=False, seed=40))
        cs_implementation = ClassicalShadows(n_qubit=n_qubit)
        cs_shots = 32
        cs_job = povm_sampler.run([qc_random], shots=cs_shots, povm=cs_implementation)
        self.assertIsInstance(cs_job, POVMSamplerJob)

    def test_result(self):
        povm_sampler = POVMSampler(sampler=self.sampler)
        n_qubit = 2
        qc_random = self.pm.run(random_circuit(num_qubits=n_qubit, depth=3, measure=False, seed=41))
        cs_implementation = ClassicalShadows(n_qubit=n_qubit)
        cs_shots = 32
        cs_job = povm_sampler.run([qc_random], shots=cs_shots, povm=cs_implementation)
        result = cs_job.result()[0]
        self.assertIsInstance(result, POVMPubResult)

    def test_recover_job(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        backend = FakeSherbrooke()
        backend.set_options(seed_simulator=25)
        pm = generate_preset_pass_manager(optimization_level=2, backend=backend)

        qc_isa = pm.run(qc)

        measurement = ClassicalShadows(2, seed_rng=13)
        runtime_sampler = Sampler(backend=backend)
        povm_sampler = POVMSampler(runtime_sampler)
        job = povm_sampler.run(pubs=[qc_isa], shots=128, povm=measurement)
        job.save_metadata(filename="saved_metadata.pkl")
        tmp = job.base_job

        job_recovered = POVMSamplerJob.recover_job(filename="saved_metadata.pkl", base_job=tmp)
        self.assertIsInstance(job_recovered, POVMSamplerJob)
        result = job_recovered.result()
        pub_result = result[0]
        observable = SparsePauliOp(["II", "XX", "YY", "ZZ"], coeffs=[1, 1, -1, 1])
        post_processor = POVMPostProcessor(pub_result)
        exp_value, _ = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 3.53125)

        job.save_metadata()

        job_recovered = POVMSamplerJob.recover_job(
            filename=f"job_metadata_{job.base_job.job_id()}.pkl", base_job=tmp
        )
        self.assertIsInstance(job_recovered, POVMSamplerJob)
        result = job_recovered.result()
        pub_result = result[0]
        observable = SparsePauliOp(["II", "XX", "YY", "ZZ"], coeffs=[1, 1, -1, 1])
        post_processor = POVMPostProcessor(pub_result)
        exp_value, _ = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 3.53125)
