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

import os
from pathlib import Path
from unittest import TestCase

import pytest
from povm_toolbox.library import ClassicalShadows
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.sampler import POVMPubResult, POVMSampler, POVMSamplerJob
from qiskit import QuantumCircuit, qpy
from qiskit.primitives import PrimitiveResult
from qiskit.providers import JobStatus
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke


class TestPOVMSamplerJob(TestCase):
    """Tests for the ``POVMSamplerJob`` class."""

    SEED = 10

    def setUp(self) -> None:
        super().setUp()
        self.sampler = AerSampler(seed=self.SEED)

    def test_initialization(self):
        povm_sampler = POVMSampler(sampler=self.sampler)
        num_qubits = 2
        # Load the circuit that was obtained through:
        #   from qiskit.circuit.random import random_circuit
        #   qc = random_circuit(num_qubits=num_qubits, depth=3, measure=False, seed=10)
        # for qiskit==1.1.1
        with open("test/sampler/random_circuits.qpy", "rb") as file:
            qc_random = qpy.load(file)[0]
        cs_implementation = ClassicalShadows(num_qubits=num_qubits)
        cs_shots = 32
        cs_job = povm_sampler.run([qc_random], shots=cs_shots, povm=cs_implementation)
        self.assertIsInstance(cs_job, POVMSamplerJob)

    def test_result(self):
        povm_sampler = POVMSampler(sampler=self.sampler)
        num_qubits = 2
        # Load the circuit that was obtained through:
        #   from qiskit.circuit.random import random_circuit
        #   qc = random_circuit(num_qubits=num_qubits, depth=2, measure=False, seed=10)
        # for qiskit==1.1.1
        with open("test/sampler/random_circuits.qpy", "rb") as file:
            qc_random = qpy.load(file)[1]
        cs_implementation = ClassicalShadows(num_qubits=num_qubits)
        cs_shots = 32
        with self.subTest("Result for a single PUB."):
            cs_job = povm_sampler.run([qc_random], shots=cs_shots, povm=cs_implementation)
            result = cs_job.result()
            self.assertIsInstance(result, PrimitiveResult)
            self.assertIsInstance(result[0], POVMPubResult)
        with self.subTest("Result for multiple PUBs."):
            cs_job = povm_sampler.run(
                [qc_random, qc_random], shots=cs_shots, povm=cs_implementation
            )
            result = cs_job.result()
            self.assertIsInstance(result, PrimitiveResult)
            self.assertEqual(len(result), 2)
            self.assertIsInstance(result[0], POVMPubResult)
            self.assertIsInstance(result[1], POVMPubResult)
        with self.subTest(
            "Error raised if incompatible lengths of raw results and metadata."
        ) and self.assertRaises(ValueError):
            cs_job = povm_sampler.run(
                [qc_random, qc_random], shots=cs_shots, povm=cs_implementation
            )
            cs_job.metadata.pop()
            _ = cs_job.result()

    def test_recover_job(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        backend = FakeSherbrooke()
        backend.set_options(seed_simulator=self.SEED)
        pm = generate_preset_pass_manager(
            optimization_level=2, backend=backend, seed_transpiler=self.SEED
        )

        qc_isa = pm.run(qc)

        measurement = ClassicalShadows(2, seed=self.SEED)
        runtime_sampler = RuntimeSampler(mode=backend)
        povm_sampler = POVMSampler(runtime_sampler)
        job = povm_sampler.run(pubs=[qc_isa], shots=128, povm=measurement)
        tmp = job.base_job

        with self.subTest("Save job with specific filename."):
            filename = "saved_metadata.pkl"
            job.save_metadata(filename=filename)
            job_recovered = POVMSamplerJob.recover_job(filename=filename, base_job=tmp)
            try:
                self.assertIsInstance(job_recovered, POVMSamplerJob)
                result = job_recovered.result()
                pub_result = result[0]
                observable = SparsePauliOp(["II", "XX", "YY", "ZZ"], coeffs=[1, 1, -1, 1])
                post_processor = POVMPostProcessor(pub_result)
                exp_value, std = post_processor.get_expectation_value(observable)
                self.assertAlmostEqual(exp_value, 4.445312500000001)
                self.assertAlmostEqual(std, 0.3881881421165156)
            except BaseException as exc:  # catch anything
                raise exc
            finally:
                Path(filename).unlink(missing_ok=True)

        with self.subTest("Save job with default filename."):
            job.save_metadata()
            filename = f"job_metadata_{job.base_job.job_id()}.pkl"
            job_recovered = POVMSamplerJob.recover_job(filename=filename, base_job=tmp)
            try:
                self.assertIsInstance(job_recovered, POVMSamplerJob)
                result = job_recovered.result()
                pub_result = result[0]
                observable = SparsePauliOp(["II", "XX", "YY", "ZZ"], coeffs=[1, -2, 1, 1])
                post_processor = POVMPostProcessor(pub_result)
                exp_value, std = post_processor.get_expectation_value(observable)
                self.assertAlmostEqual(exp_value, -1.3906250000000009)
                self.assertAlmostEqual(std, 0.6732583954195841)
            except BaseException as exc:  # catch anything
                raise exc
            finally:
                Path(filename).unlink(missing_ok=True)

        with self.subTest(
            "Error if id of ``base_job`` does not match the one stored in the metadata file."
        ) and self.assertRaises(ValueError):
            filename = f"job_metadata_{job.base_job.job_id()}.pkl"
            job.save_metadata(filename=filename)
            try:
                job2 = povm_sampler.run(pubs=[qc_isa], shots=1, povm=measurement)
                _ = POVMSamplerJob.recover_job(filename=filename, base_job=job2.base_job)
            except BaseException as exc:  # catch anything
                raise exc
            finally:
                Path(filename).unlink(missing_ok=True)

    @pytest.mark.skipif(
        os.getenv("QISKIT_IBM_TOKEN") is None, reason="Missing QiskitRuntimeService configuration."
    )
    def test_recover_job_runtime(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        instance = os.getenv("QISKIT_IBM_INSTANCE")
        token = os.getenv("QISKIT_IBM_TOKEN")
        url = os.getenv("QISKIT_IBM_URL")
        if instance is None or token is None or url is None:
            pytest.skip("Missing QiskitRuntimeService configuration.")
        channel = "ibm_quantum" if url.find("quantum-computing.ibm.com") >= 0 else "ibm_cloud"
        service = QiskitRuntimeService(instance=instance, token=token, url=url, channel=channel)

        backend = service.least_busy(operational=True, simulator=True)
        pm = generate_preset_pass_manager(
            optimization_level=2, backend=backend, seed_transpiler=self.SEED
        )

        qc_isa = pm.run(qc)

        measurement = ClassicalShadows(2, seed=self.SEED)
        runtime_sampler = RuntimeSampler(mode=backend)
        runtime_sampler.options.simulator.seed_simulator = self.SEED
        povm_sampler = POVMSampler(runtime_sampler)
        job = povm_sampler.run(pubs=[qc_isa], shots=128, povm=measurement)

        try:
            job.save_metadata()
            filename = f"job_metadata_{job.base_job.job_id()}.pkl"
            job_recovered = POVMSamplerJob.recover_job(filename=filename, service=service)
            self.assertIsInstance(job_recovered, POVMSamplerJob)
            result = job_recovered.result()
            pub_result = result[0]
            observable = SparsePauliOp(["II", "XX", "YY", "ZZ"], coeffs=[1, -2, 1, 1])
            post_processor = POVMPostProcessor(pub_result)
            exp_value, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, -1.3906250000000009)
            self.assertAlmostEqual(std, 0.6732583954195841)
        except BaseException as exc:  # catch anything
            raise exc
        finally:
            Path(filename).unlink(missing_ok=True)

    def test_status(self):
        """Test the ``status`` and associated methods."""
        povm_sampler = POVMSampler(sampler=self.sampler)
        num_qubits = 2
        # Load the circuit that was obtained through:
        #   from qiskit.circuit.random import random_circuit
        #   qc = random_circuit(num_qubits=num_qubits, depth=3, measure=False, seed=10)
        # for qiskit==1.1.1
        with open("test/sampler/random_circuits.qpy", "rb") as file:
            qc_random = qpy.load(file)[0]
        cs_implementation = ClassicalShadows(num_qubits=num_qubits)
        cs_job = povm_sampler.run([qc_random], shots=100, povm=cs_implementation)
        job_status, is_done, is_running, is_cancelled, in_final = (
            cs_job.status(),
            cs_job.done(),
            cs_job.running(),
            cs_job.cancelled(),
            cs_job.in_final_state(),
        )
        if job_status == JobStatus.RUNNING:
            self.assertFalse(is_done)
            self.assertTrue(is_running)
            self.assertFalse(is_cancelled)
            self.assertFalse(in_final)

        _ = cs_job.result()
        self.assertEqual(cs_job.status(), JobStatus.DONE)
        self.assertTrue(cs_job.done())
        self.assertFalse(cs_job.running())
        self.assertFalse(cs_job.cancelled())
        self.assertTrue(cs_job.in_final_state())
        cs_job.cancel()
        self.assertFalse(cs_job.cancelled())
