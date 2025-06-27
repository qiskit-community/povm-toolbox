# (C) Copyright IBM 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the POVMPostProcessor class."""

import numpy as np
import pytest
from povm_toolbox.library import ClassicalShadows, LocallyBiasedClassicalShadows
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.quantum_info import MultiQubitDual, ProductDual
from povm_toolbox.sampler import POVMSampler
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Operator, SparsePauliOp


class TestPostProcessor:
    """Test the methods and attributes of the :class:`.POVMPostProcessor class`."""

    SEED = 42

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        povm_sampler = POVMSampler(sampler=Sampler(seed=self.SEED))
        self.measurement = ClassicalShadows(num_qubits=2, seed=self.SEED)

        job = povm_sampler.run([qc], shots=32, povm=self.measurement)
        result = job.result()
        self.pub_result = result[0]

    def test_init(self, subtests):
        """Test that ``__init__`` works correctly."""
        with subtests.test("Initialization with default ``dual``."):
            post_processor = POVMPostProcessor(self.pub_result)
            assert isinstance(post_processor, POVMPostProcessor)
            assert post_processor._dual is None

        with subtests.test("Initialization with valid ``dual`` argument."):
            povm = self.measurement.definition()
            dual = ProductDual.build_dual_from_frame(povm)
            post_processor = POVMPostProcessor(self.pub_result, dual=dual)
            assert isinstance(post_processor, POVMPostProcessor)
            assert post_processor._dual is dual

        with subtests.test("Initialization with invalid ``dual`` argument.") and pytest.raises(
            ValueError
        ):
            povm = LocallyBiasedClassicalShadows(
                num_qubits=2, bias=np.asarray([0.8, 0.1, 0.1])
            ).definition()
            dual = ProductDual.build_dual_from_frame(povm)
            post_processor = POVMPostProcessor(self.pub_result, dual=dual)

    def test_dual(self, subtests):
        """Test that the ``dual`` property and setter work correctly."""
        with subtests.test("Test default ``dual``."):
            post_processor = POVMPostProcessor(self.pub_result)
            assert post_processor._dual is None
            assert isinstance(post_processor.dual, ProductDual)
            assert isinstance(post_processor._dual, ProductDual)
        with subtests.test("Test setting ``dual`` after initialization."):
            post_processor = POVMPostProcessor(self.pub_result)
            assert post_processor._dual is None
            povm = self.measurement.definition()
            dual = ProductDual.build_dual_from_frame(povm, alphas=((1, 2, 2, 2, 2, 2), None))
            post_processor.dual = dual
            assert post_processor._dual is dual
            assert post_processor.dual is dual
        with subtests.test("Test setting invalid ``dual`` after initialization.") and pytest.raises(
            ValueError
        ):
            post_processor = POVMPostProcessor(self.pub_result)
            povm = LocallyBiasedClassicalShadows(
                num_qubits=2, bias=np.asarray([0.8, 0.1, 0.1])
            ).definition()
            dual = ProductDual.build_dual_from_frame(povm)
            post_processor.dual = dual

    def test_get_decomposition_weights(self):
        """Test that the ``get_decomposition_weights`` method works correctly."""
        observable = SparsePauliOp(["ZZ", "XX", "YY"], coeffs=[1, 2, 3])
        post_processor = POVMPostProcessor(self.pub_result)
        weights = post_processor.get_decomposition_weights(
            observable, set([(0, 0), (0, 1), (1, 1), (0, 2), (2, 2), (2, 3), (5, 5), (5, 0)])
        )
        assert np.isclose(weights[(0, 0)], 1 * 9)
        assert np.isclose(weights[(0, 1)], -1 * 9)
        assert np.isclose(weights[(1, 1)], 1 * 9)
        assert np.isclose(weights[(0, 2)], 0)
        assert np.isclose(weights[(2, 2)], 2 * 9)
        assert np.isclose(weights[(2, 3)], -2 * 9)
        assert np.isclose(weights[(5, 5)], 3 * 9)
        assert np.isclose(weights[(5, 0)], 0)

    def test_get_expectation_value(self, subtests):
        """Test that the ``get_expectation_value`` method works correctly."""
        post_processor = POVMPostProcessor(self.pub_result)
        with subtests.test("Test with default ``loc`` for un-parametrized circuit."):
            observable = SparsePauliOp(["ZZ", "XX", "YY"], coeffs=[1, 2, 3])
            exp_val, std = post_processor.get_expectation_value(observable)
            assert np.isclose(exp_val, -2.2499999999999987)
            assert np.isclose(std, 2.3563572213988917)
        with subtests.test("Test with specified ``loc`` argument."):
            observable = SparsePauliOp(["IZ", "XX", "ZY"], coeffs=[-0.5, 1, -2])
            exp_val, std = post_processor.get_expectation_value(observable, loc=0)
            assert np.isclose(exp_val, -1.6406249999999998)
            assert np.isclose(std, 1.3442744428582185)

    def test_get_expectation_value_parametrized_circuit(self, subtests):
        """Test that the ``get_expectation_value`` method works correctly with parametrized circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.ry(theta=Parameter("theta"), qubit=0)
        povm_sampler = POVMSampler(sampler=Sampler(seed=self.SEED))
        measurement = ClassicalShadows(num_qubits=2, seed=self.SEED)
        job = povm_sampler.run(
            [(qc, np.array(2 * [[0, np.pi / 3, np.pi]]))], shots=32, povm=measurement
        )
        pub_result = job.result()[0]
        observable = SparsePauliOp(["IZ", "XX", "ZY"], coeffs=[-0.5, 1, -2])
        with subtests.test("Test with default ``loc`` for parametrized circuit."):
            post_processor = POVMPostProcessor(pub_result)
            assert post_processor.counts.shape == (2, 3)
            assert sum(post_processor.counts[0, 0].values()) == 32
            exp_val, std = post_processor.get_expectation_value(observable)
            assert isinstance(exp_val, np.ndarray)
            assert exp_val.shape == (2, 3)
            assert np.allclose(
                exp_val,
                np.asarray(
                    [
                        [-4.171875, 0.703125, -2.578125],
                        [-0.5625, -0.5625, -2.15625],
                    ]
                ),
            )
            assert isinstance(std, np.ndarray)
            assert std.shape == (2, 3)
            assert np.allclose(
                std,
                np.asarray(
                    [
                        [1.59914439, 0.41510017, 0.8915795],
                        [1.46287216, 1.11232782, 1.04977856],
                    ]
                ),
            )
        with subtests.test("Test with combining counts."):
            post_processor = POVMPostProcessor(pub_result, combine_counts=True)
            assert post_processor.counts.shape == (1,)
            assert sum(post_processor.counts[0].values()) == 2 * 3 * 32
            exp_val, std = post_processor.get_expectation_value(observable)
            assert isinstance(exp_val, float)
            assert np.isclose(exp_val, -1.5546875)
            assert np.isclose(std, 0.47945849433203297)

    def test_single_exp_value_and_std(self):
        """Test that the ``_single_exp_value_and_std`` method works correctly."""
        observable = SparsePauliOp(["ZX", "XZ", "YY"], coeffs=[1.2, 2, -3])
        post_processor = POVMPostProcessor(self.pub_result)
        exp_val, std = post_processor._single_exp_value_and_std(observable, loc=0)
        assert np.isclose(exp_val, 6.862499999999998)
        assert np.isclose(std, 1.9438371907630394)

    def test_catch_zero_division(self):
        """Test that the ``_single_exp_value_and_std`` method catches a zero division gracefully."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        povm_sampler = POVMSampler(sampler=Sampler(seed=self.SEED))

        # NOTE: the test case here is not really sensible because setting shots=1 would only be done
        # if one has another set of circuit parameters, but for this simple test it suffices
        job = povm_sampler.run([qc], shots=1, povm=self.measurement)
        result = job.result()

        observable = SparsePauliOp(["ZX", "XZ", "YY"], coeffs=[1.2, 2, -3])
        post_processor = POVMPostProcessor(result[0])

        exp_val, std = post_processor._single_exp_value_and_std(observable, loc=0)

        assert np.isclose(exp_val, 0.0)
        assert np.isnan(std)

    def test_get_state_snapshot(self, subtests):
        """Test that the ``get_state_snapshot`` method works correctly."""
        post_processor = POVMPostProcessor(self.pub_result)

        with subtests.test("Test method works correctly"):
            outcome = self.pub_result.get_samples()[0]
            # check outcome first
            assert outcome == (5, 1)
            expected_snapshot = {
                (0,): Operator([[0.5, 1.5j], [-1.5j, 0.5]]),
                (1,): Operator([[-1, 0.0], [0, 2.0]]),
            }
            snapshot = post_processor.get_state_snapshot(outcome)
            # check snapshot
            assert snapshot == expected_snapshot

        with subtests.test("Test raises errors") and pytest.raises(NotImplementedError):
            post_processor._dual = MultiQubitDual([Operator(np.eye(4))])
            _ = post_processor.get_state_snapshot(outcome)
