# (C) Copyright IBM 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the MutuallyUnbiasedBasesMeasurements class."""

import numpy as np
import pytest
from numpy.random import default_rng
from povm_toolbox.library import MutuallyUnbiasedBasesMeasurements
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.sampler import POVMSampler
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, SGate, UGate
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector


class TestMutuallyUnbiasedBasesMeasurements:
    SEED = 17863

    def test_init(self, subtests):
        """Test the implementation of mutually-unbiased-bases POVMs."""
        qc = QuantumCircuit(2)
        qc.h(0)

        num_qubits = qc.num_qubits

        with subtests.test("Test uniform angles across qubits."):
            measurement = MutuallyUnbiasedBasesMeasurements(
                num_qubits,
                bias=np.ones(3) / 3,
                angles=np.asarray([0.75, -np.pi / 3, 0.2]),
                seed=self.SEED,
            )
            sampler = StatevectorSampler(seed=default_rng(self.SEED))
            povm_sampler = POVMSampler(sampler=sampler)

            job = povm_sampler.run([qc], shots=128, povm=measurement)
            pub_result = job.result()[0]

            post_processor = POVMPostProcessor(pub_result)

            observable = SparsePauliOp(["ZI"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            assert np.isclose(exp_value, 0.6518233926221875)
            assert np.isclose(std, 0.11921601584589436)
            observable = SparsePauliOp(["ZY"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            assert np.isclose(exp_value, -1.0553339936080581)
            assert np.isclose(std, 0.2102928553415571)

        with subtests.test("Test specific angles for each qubit."):
            measurement = MutuallyUnbiasedBasesMeasurements(
                num_qubits,
                bias=np.ones(3) / 3,
                angles=np.asarray([[1.2, 0.0, 0.4], [3.5, -0.4, 0.8]]),
                seed=self.SEED,
            )
            sampler = StatevectorSampler(seed=default_rng(self.SEED))
            povm_sampler = POVMSampler(sampler=sampler)

            job = povm_sampler.run([qc], shots=128, povm=measurement)
            pub_result = job.result()[0]

            post_processor = POVMPostProcessor(pub_result)

            observable = SparsePauliOp(["ZI"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            assert np.isclose(exp_value, 0.7504820624371005)
            assert np.isclose(std, 0.11019654428864213)
            observable = SparsePauliOp(["ZY"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            assert np.isclose(exp_value, -0.8423974419138216)
            assert np.isclose(std, 0.23586993121676594)

        with subtests.test("Test `array_like` bias and angles."):
            measurement = MutuallyUnbiasedBasesMeasurements(
                num_qubits,
                bias=[1 / 3, 1 / 3, 1 / 3],
                angles=[[1.2, 0.0, 0.4], [3.5, -0.4, 0.8]],
                seed=self.SEED,
            )
            sampler = StatevectorSampler(seed=default_rng(self.SEED))
            povm_sampler = POVMSampler(sampler=sampler)

            job = povm_sampler.run([qc], shots=128, povm=measurement)
            pub_result = job.result()[0]

            post_processor = POVMPostProcessor(pub_result)

            observable = SparsePauliOp(["ZI"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            assert np.isclose(exp_value, 0.7504820624371005)
            assert np.isclose(std, 0.11019654428864213)
            observable = SparsePauliOp(["ZY"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            assert np.isclose(exp_value, -0.8423974419138216)
            assert np.isclose(std, 0.23586993121676594)

    def test_init_errors(self, subtests):
        """Test that the ``__init__`` method raises errors correctly."""
        with subtests.test("Test invalid shape for ``bias``.") and pytest.raises(ValueError):
            MutuallyUnbiasedBasesMeasurements(1, bias=np.ones(2) / 2, angles=np.zeros(3))
        with subtests.test("Test invalid shape for ``angles``.") and pytest.raises(ValueError):
            MutuallyUnbiasedBasesMeasurements(1, bias=np.ones(3) / 3, angles=np.zeros(2))

    def test_repr(self):
        """Test that the ``__repr__`` method works correctly."""
        mub_str = "MutuallyUnbiasedBasesMeasurements(num_qubits=1, bias=array([[0.2, 0.3, 0.5]]), angles=array([0., 1., 2.]))"
        povm = MutuallyUnbiasedBasesMeasurements(
            1, bias=np.asarray([0.2, 0.3, 0.5]), angles=np.arange(3, dtype=float)
        )
        assert povm.__repr__() == mub_str

    def test_process_angles(self):
        """Test if the processing of the angles works correctly."""
        rng = default_rng()

        set_angles = 2 * np.pi * (rng.random(12).reshape((4, 3)) - 0.5)

        for angles in set_angles:
            theta, phi, lam = angles

            processed_angles_1 = MutuallyUnbiasedBasesMeasurements._process_angles(theta, phi, lam)

            H = HGate().to_matrix()
            S = SGate().to_matrix()
            U = UGate(theta, phi, lam).to_matrix()

            rotated_Z_msmt = U @ np.asarray([1, 0])
            rotated_X_msmt = U @ H @ np.asarray([1, 0])
            rotated_Y_msmt = U @ S @ H @ np.asarray([1, 0])
            rotated_msmts = [rotated_Z_msmt, rotated_X_msmt, rotated_Y_msmt]

            bloch_vectors = np.real_if_close(
                [
                    [
                        Statevector(msmt).expectation_value(Operator.from_label(op))
                        for op in ["X", "Y", "Z"]
                    ]
                    for msmt in rotated_msmts
                ]
            )

            thetas = np.arctan2(np.linalg.norm(bloch_vectors[:, :2], axis=1), bloch_vectors[:, 2])
            phis = np.arctan2(bloch_vectors[:, 1], bloch_vectors[:, 0])
            processed_angles_2 = np.vstack((thetas, phis)).T.flatten()

            assert np.allclose(processed_angles_1, processed_angles_2)
