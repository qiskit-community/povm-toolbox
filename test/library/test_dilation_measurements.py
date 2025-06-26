# (C) Copyright IBM 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the DilationMeasurements class."""

import numpy as np
import pytest
from numpy.random import default_rng
from povm_toolbox.library import (
    DilationMeasurements,
)
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.sampler import POVMSampler
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BackendSamplerV2 as Sampler
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeManilaV2


class TestDilationMeasurements:
    SEED = 9128346

    def test_init_errors(self, subtests):
        """Test that the ``__init__`` method raises errors correctly."""
        # Sanity check
        measurement = DilationMeasurements(1, parameters=np.random.uniform(0, 1, size=8))
        assert isinstance(measurement, DilationMeasurements)
        with subtests.test(
            "Test invalid shape for ``parameters``, not enough parameters."
        ) and pytest.raises(ValueError):
            DilationMeasurements(1, parameters=np.ones(7))
        with subtests.test(
            "Test invalid shape for ``parameters``, number of qubits not matching."
        ) and pytest.raises(ValueError):
            DilationMeasurements(1, parameters=np.ones((2, 8)))
        with subtests.test(
            "Test invalid shape for ``parameters``, too many dimensions."
        ) and pytest.raises(ValueError):
            DilationMeasurements(1, parameters=np.ones((1, 1, 8)))

    def test_repr(self):
        """Test that the ``__repr__`` method works correctly."""
        mub_str = (
            "DilationMeasurements(num_qubits=1, parameters=array"
            "([[0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]))"
        )
        povm = DilationMeasurements(1, parameters=0.1 * np.arange(8))
        assert povm.__repr__() == mub_str

    @pytest.mark.parametrize(
        ["pauli", "expected_exp_val", "expected_std"],
        [
            ("ZI", -0.18749696469140853, 0.1428020715587502),
            ("IZ", -0.18750068585105134, 0.1428022382724095),
            ("XI", 0.8949297904387798, 0.16958103337866398),
        ],
    )
    def test_to_sampler_pub(self, pauli: str, expected_exp_val: float, expected_std: float):
        """Test that the ``to_sampler_pub`` method works correctly."""
        num_qubits = 2
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        backend = FakeManilaV2()
        povm_sampler = POVMSampler(
            sampler=Sampler(backend=backend, options={"seed_simulator": self.SEED})
        )

        pm = generate_preset_pass_manager(
            optimization_level=0,
            initial_layout=[0, 1, 2, 3],
            backend=backend,
            seed_transpiler=self.SEED,
        )

        measurement = DilationMeasurements(
            num_qubits,
            parameters=np.asarray(
                [
                    0.75,
                    0.30408673,
                    0.375,
                    0.40678524,
                    0.32509973,
                    0.25000035,
                    0.49999321,
                    0.83333313,
                ]
            ),
        )

        job = povm_sampler.run([qc], shots=128, povm=measurement, pass_manager=pm)
        pub_result = job.result()[0]

        post_processor = POVMPostProcessor(pub_result)

        observable = SparsePauliOp([pauli], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        assert np.isclose(exp_value, expected_exp_val)
        assert np.isclose(std, expected_std)

    def test_definition(self, subtests):
        """Test that the ``definition`` method works correctly."""
        num_qubits = 1

        # parameters defining a SIC-POVM
        sic_parameters = np.asarray(
            [[0.75, 0.30408673, 0.375, 0.40678524, 0.32509973, 0.25000035, 0.49999321, 0.83333313]]
        )

        # define measurement and the quantum-informational POVM
        measurement = DilationMeasurements(num_qubits, parameters=sic_parameters)
        povm = measurement.definition()[(0,)]

        with subtests.test("Test effects"):
            effects = np.empty((4, 2, 2), dtype=complex)
            effects[0] = np.asarray(
                [
                    [5.00000000e-01 + 0.00000000e00j, 3.25620884e-09 - 4.42295492e-06j],
                    [3.25620884e-09 + 4.42295492e-06j, 3.91250817e-11 - 2.67501659e-30j],
                ]
            )
            effects[1] = np.asarray(
                [
                    [0.16666667 + 0.00000000e00j, 0.23570227 + 4.41719108e-06j],
                    [0.23570227 - 4.41719108e-06j, 0.33333335 - 9.43454930e-23j],
                ]
            )
            effects[2] = np.asarray(
                [
                    [0.16666666 + 0.00000000e00j, -0.11785114 - 2.04124135e-01j],
                    [-0.11785114 + 2.04124135e-01j, 0.33333334 - 8.73465760e-18j],
                ]
            )
            effects[3] = np.asarray(
                [
                    [0.16666667 + 0.00000000e00j, -0.11785113 + 2.04124140e-01j],
                    [-0.11785113 - 2.04124140e-01j, 0.33333331 - 6.84480542e-18j],
                ]
            )
            for effect, povm_operator in zip(effects, povm.operators):
                assert np.allclose(povm_operator.data, effect)

        with subtests.test("Test bloch vectors"):
            bloch_vectors_check = np.asarray(
                [
                    [
                        0.0,
                        0.0,
                        0.5,
                    ],
                    [0.47140454, 0.0, -0.16666668],
                    [-0.23570228, 0.40824827, -0.16666668],
                    [-0.23570226, -0.40824828, -0.16666664],
                ]
            )
            assert np.allclose(povm.get_bloch_vectors(), bloch_vectors_check)

    @pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
    def test_compose_circuit(self, num_qubits: int):
        """Test that the ``compose_circuit`` method works correctly."""
        sampler = StatevectorSampler(seed=default_rng(self.SEED))
        povm_sampler = POVMSampler(sampler)
        measurement = DilationMeasurements(
            num_qubits=2,
            parameters=np.asarray(
                [
                    0.75,
                    0.30408673,
                    0.375,
                    0.40678524,
                    0.32509973,
                    0.25000035,
                    0.49999321,
                    0.83333313,
                ]
            ),
        )
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)
        measurement.measurement_layout = [0, 1]
        job = povm_sampler.run([qc], shots=32, povm=measurement)
        pub_result = job.result()[0]
        assert pub_result.metadata.composed_circuit.num_qubits == max(4, num_qubits)
        assert pub_result.metadata.composed_circuit.num_ancillas == max(4 - num_qubits, 0)
        observable = SparsePauliOp(["XI", "XX", "YY", "ZX"], coeffs=[1, 1, -1, 1])
        post_processor = POVMPostProcessor(pub_result)
        exp_value, std = post_processor.get_expectation_value(observable)
        assert np.isclose(exp_value, 2.051760590702834)
        assert np.isclose(std, 1.115878952074859)
