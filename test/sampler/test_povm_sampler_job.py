# (C) Copyright IBM 2024, 2025.
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

import numpy as np
import pytest
from numpy.random import default_rng
from povm_toolbox.library import ClassicalShadows
from povm_toolbox.library.metadata import POVMMetadata
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.sampler import POVMPubResult, POVMSampler, POVMSamplerJob
from qiskit import QuantumCircuit, qpy
from qiskit.primitives import PrimitiveResult, StatevectorSampler
from qiskit.providers import JobStatus
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2


class TestPOVMSamplerJob:
    """Tests for the ``POVMSamplerJob`` class."""

    SEED = 10

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        rng = default_rng(self.SEED)
        self.sampler = StatevectorSampler(seed=rng)

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
        assert isinstance(cs_job, POVMSamplerJob)

    def test_result(self, subtests):
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
        with subtests.test("Result for a single PUB."):
            cs_job = povm_sampler.run([qc_random], shots=cs_shots, povm=cs_implementation)
            result = cs_job.result()
            assert isinstance(result, PrimitiveResult)
            assert isinstance(result[0], POVMPubResult)
        with subtests.test("Result for multiple PUBs."):
            cs_job = povm_sampler.run(
                [qc_random, qc_random], shots=cs_shots, povm=cs_implementation
            )
            result = cs_job.result()
            assert isinstance(result, PrimitiveResult)
            assert len(result) == 2
            assert isinstance(result[0], POVMPubResult)
            assert isinstance(result[1], POVMPubResult)
        with subtests.test(
            "Error raised if incompatible lengths of raw results and metadata."
        ) and pytest.raises(ValueError):
            cs_job = povm_sampler.run(
                [qc_random, qc_random], shots=cs_shots, povm=cs_implementation
            )
            cs_job.metadata.pop()
            _ = cs_job.result()

    def test_recover_job(self, subtests):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        backend = FakeManilaV2()
        backend.set_options(seed_simulator=self.SEED)
        pm = generate_preset_pass_manager(
            optimization_level=0,
            initial_layout=[0, 1],
            backend=backend,
            seed_transpiler=self.SEED,
        )

        qc_isa = pm.run(qc)

        measurement = ClassicalShadows(2, seed=self.SEED)
        runtime_sampler = RuntimeSampler(mode=backend)
        povm_sampler = POVMSampler(runtime_sampler)
        job = povm_sampler.run(pubs=[qc_isa], shots=128, povm=measurement)
        tmp = job.base_job

        with subtests.test("Save job with specific filename."):
            filename = "saved_metadata.pkl"
            job.save_metadata(filename=filename)
            job_recovered = POVMSamplerJob.recover_job(filename=filename, base_job=tmp)
            try:
                assert isinstance(job_recovered, POVMSamplerJob)
                result = job_recovered.result()
                pub_result = result[0]
                observable = SparsePauliOp(["II", "XX", "YY", "ZZ"], coeffs=[1, 1, -1, 1])
                post_processor = POVMPostProcessor(pub_result)
                exp_value, std = post_processor.get_expectation_value(observable)
                assert np.isclose(exp_value, 4.445312500000001)
                assert np.isclose(std, 0.3881881421165156)
            except BaseException as exc:  # catch anything
                raise exc
            finally:
                Path(filename).unlink(missing_ok=True)

        with subtests.test("Save job with default filename."):
            job.save_metadata()
            filename = f"job_metadata_{job.base_job.job_id()}.pkl"
            job_recovered = POVMSamplerJob.recover_job(filename=filename, base_job=tmp)
            try:
                assert isinstance(job_recovered, POVMSamplerJob)
                result = job_recovered.result()
                pub_result = result[0]
                observable = SparsePauliOp(["II", "XX", "YY", "ZZ"], coeffs=[1, -2, 1, 1])
                post_processor = POVMPostProcessor(pub_result)
                exp_value, std = post_processor.get_expectation_value(observable)
                assert np.isclose(exp_value, -1.3906250000000009)
                assert np.isclose(std, 0.6732583954195841)
            except BaseException as exc:  # catch anything
                raise exc
            finally:
                Path(filename).unlink(missing_ok=True)

        with subtests.test(
            "Error if id of ``base_job`` does not match the one stored in the metadata file."
        ) and pytest.raises(ValueError):
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
        not os.getenv("QISKIT_IBM_TOKEN"), reason="Missing QiskitRuntimeService configuration."
    )
    def test_recover_job_runtime(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        instance = os.getenv("QISKIT_IBM_INSTANCE")
        channel = os.getenv("QISKIT_IBM_CHANNEL")
        token = os.getenv("QISKIT_IBM_TOKEN")
        qpu = os.getenv("QISKIT_IBM_QPU")
        if instance is None or channel is None or token is None or qpu is None:
            pytest.skip("Missing QiskitRuntimeService configuration.")
        service = QiskitRuntimeService(instance=instance, channel=channel, token=token)

        backend = service.backend(name=qpu)
        pm = generate_preset_pass_manager(
            optimization_level=0,
            initial_layout=[0, 1],
            backend=backend,
            seed_transpiler=self.SEED,
        )

        qc_isa = pm.run(qc)

        measurement = ClassicalShadows(2, seed=self.SEED)
        runtime_sampler = RuntimeSampler(mode=backend)
        povm_sampler = POVMSampler(runtime_sampler)
        job = povm_sampler.run(pubs=[qc_isa], shots=128, povm=measurement)

        try:
            job.save_metadata()
            filename = f"job_metadata_{job.base_job.job_id()}.pkl"
            job_recovered = POVMSamplerJob.recover_job(filename=filename, service=service)
            assert isinstance(job_recovered, POVMSamplerJob)
            result = job_recovered.result()
            pub_result = result[0]
            assert isinstance(result, PrimitiveResult)
            assert isinstance(pub_result, POVMPubResult)
            assert isinstance(pub_result.metadata, POVMMetadata)
            assert pub_result.data.povm_measurement_creg.num_bits == 2
            assert pub_result.data.povm_measurement_creg.num_shots == 128
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
            assert not is_done
            assert is_running
            assert not is_cancelled
            assert not in_final

        _ = cs_job.result()
        assert cs_job.status() == JobStatus.DONE
        assert cs_job.done()
        assert not cs_job.running()
        assert not cs_job.cancelled()
        assert cs_job.in_final_state()
        cs_job.cancel()
        assert not cs_job.cancelled()
