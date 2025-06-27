# (C) Copyright IBM 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the LocallyBiasedClassicalShadows class."""

import numpy as np
import numpy.typing as npt
import pytest
from numpy.random import default_rng
from povm_toolbox.library import LocallyBiasedClassicalShadows
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.quantum_info.single_qubit_povm import SingleQubitPOVM
from povm_toolbox.sampler import POVMSampler
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Operator, SparsePauliOp


class TestRandomizedPMs:
    SEED = 13

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        basis_0 = np.asarray([1.0, 0], dtype=complex)
        basis_1 = np.asarray([0, 1.0], dtype=complex)
        basis_plus = 1.0 / np.sqrt(2) * (basis_0 + basis_1)
        basis_minus = 1.0 / np.sqrt(2) * (basis_0 - basis_1)
        basis_plus_i = 1.0 / np.sqrt(2) * (basis_0 + 1.0j * basis_1)
        basis_minus_i = 1.0 / np.sqrt(2) * (basis_0 - 1.0j * basis_1)

        self.Z0 = np.outer(basis_0, basis_0.conj())
        self.Z1 = np.outer(basis_1, basis_1.conj())
        self.X0 = np.outer(basis_plus, basis_plus.conj())
        self.X1 = np.outer(basis_minus, basis_minus.conj())
        self.Y0 = np.outer(basis_plus_i, basis_plus_i.conj())
        self.Y1 = np.outer(basis_minus_i, basis_minus_i.conj())

    @pytest.mark.parametrize(
        ["bias", "expected_outcomes"],
        [
            (
                np.asarray([0.2, 0.3, 0.5]),
                {
                    "ZI": (0.9374999999999998, 0.3505108598934214),
                    "ZY": (-1.2499999999999993, 0.593988704139364),
                },
            ),
            (
                np.asarray([[0.5, 0.1, 0.4], [0.3, 0.4, 0.3]]),
                {
                    "ZI": (0.7291666666666665, 0.24749529339140175),
                    "ZY": (-0.78125, 0.43626216977818505),
                },
            ),
            (
                [[0.5, 0.1, 0.4], [0.3, 0.4, 0.3]],
                {
                    "ZI": (0.7291666666666665, 0.24749529339140175),
                    "ZY": (-0.78125, 0.43626216977818505),
                },
            ),
        ],
    )
    def test_init(
        self,
        bias: npt.ArrayLike,
        expected_outcomes: dict[str, tuple[float, float]],
    ):
        """Test the implementation of locally-biased classical shadows."""

        qc = QuantumCircuit(2)
        qc.h(0)

        num_qubits = qc.num_qubits

        measurement = LocallyBiasedClassicalShadows(
            num_qubits,
            bias=bias,
            seed=self.SEED,
        )
        sampler = StatevectorSampler(seed=default_rng(self.SEED))
        povm_sampler = POVMSampler(sampler=sampler)

        job = povm_sampler.run([qc], shots=32, povm=measurement)
        pub_result = job.result()[0]

        post_processor = POVMPostProcessor(pub_result)

        for pauli, (expected_exp_val, expected_std) in expected_outcomes.items():
            observable = SparsePauliOp([pauli], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            assert np.isclose(exp_value, expected_exp_val)
            assert np.isclose(std, expected_std)

    def test_qc_build(self):
        """Test if we can build a LB Classical Shadow POVM from the generic class"""

        for num_qubits in range(1, 11):
            q = np.random.uniform(0, 5, size=3 * num_qubits).reshape((num_qubits, 3))
            q /= q.sum(axis=1)[:, np.newaxis]

            cs_implementation = LocallyBiasedClassicalShadows(num_qubits=num_qubits, bias=q)
            assert num_qubits == cs_implementation.num_qubits
            cs_povm = cs_implementation.definition()
            for i in range(num_qubits):
                sqpovm = SingleQubitPOVM(
                    [
                        q[i, 0] * Operator.from_label("0"),
                        q[i, 0] * Operator.from_label("1"),
                        q[i, 1] * Operator.from_label("+"),
                        q[i, 1] * Operator.from_label("-"),
                        q[i, 2] * Operator.from_label("r"),
                        q[i, 2] * Operator.from_label("l"),
                    ]
                )
                assert cs_povm._frames[(i,)].num_outcomes == sqpovm.num_outcomes
                for k in range(sqpovm.num_outcomes):
                    assert np.allclose(cs_povm._frames[(i,)][k], sqpovm[k])

    def test_zero_bias(self, subtests):
        """Test the implementation of LBCS with some biases set to zero."""

        qc = QuantumCircuit(2)
        qc.h(0)

        num_qubits = qc.num_qubits
        measurement = LocallyBiasedClassicalShadows(
            num_qubits,
            bias=np.asarray([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]),
            seed=self.SEED,
        )

        with subtests.test("Test that the POVM is not IC."):
            assert not measurement.definition().informationally_complete

        sampler = StatevectorSampler(seed=default_rng(self.SEED))
        povm_sampler = POVMSampler(sampler=sampler)

        job = povm_sampler.run([qc], shots=32, povm=measurement)
        pub_result = job.result()[0]

        post_processor = POVMPostProcessor(pub_result)

        with subtests.test("Test the dual frame."):
            assert post_processor.dual.is_dual_to(measurement.definition())

        with subtests.test("Test with compatible observable."):
            observable = SparsePauliOp(["ZZ"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            assert np.isclose(exp_value, 0.6249999999999993)
            assert np.isclose(std, 0.4347552147751572)

        with subtests.test("Test with incompatible observable."):
            observable = SparsePauliOp(["XY"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            assert np.isclose(exp_value, 0.0)
            assert np.isclose(std, 0.0)

    def test_repr(self):
        """Test that the ``__repr__`` method works correctly."""
        lbcs_str = "LocallyBiasedClassicalShadows(num_qubits=1, bias=array([[0.2, 0.3, 0.5]]))"
        povm = LocallyBiasedClassicalShadows(
            1,
            bias=np.asarray([0.2, 0.3, 0.5]),
        )
        assert povm.__repr__() == lbcs_str
