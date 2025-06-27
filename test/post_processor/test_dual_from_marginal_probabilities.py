# (C) Copyright IBM 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the `dual_from_marginal_probabilities` function."""

import numpy as np
import pytest
from numpy.random import default_rng
from povm_toolbox.library import (
    RandomizedProjectiveMeasurements,
)
from povm_toolbox.post_processor import (
    POVMPostProcessor,
    dual_from_marginal_probabilities,
)
from povm_toolbox.quantum_info import MultiQubitPOVM, ProductPOVM, SingleQubitPOVM
from povm_toolbox.sampler import POVMSampler
from qiskit import qpy
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import (
    Operator,
    SparsePauliOp,
    Statevector,
    random_density_matrix,
)


class TestDualFromMarginalProbabilities:
    """Test that we can construct optimal dual of a POVM from marginal probabilities."""

    SEED = 29

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.povm = MultiQubitPOVM(
            [
                0.3 * Operator.from_label("0"),
                0.3 * Operator.from_label("1"),
                0.6 * Operator.from_label("+"),
                0.6 * Operator.from_label("-"),
                0.1 * Operator.from_label("r"),
                0.1 * Operator.from_label("l"),
            ]
        )

    def test_not_implemented(self):
        """Test that errors are correctly raised."""
        joint_povm: MultiQubitPOVM | SingleQubitPOVM = self.povm
        state = random_density_matrix(2, seed=self.SEED)
        with pytest.raises(NotImplementedError):
            _ = dual_from_marginal_probabilities(joint_povm, state)

    def test_implemented(self):
        """Test that the method constructs a valid dual."""
        prod_povm: ProductPOVM = ProductPOVM.from_list([self.povm, self.povm])
        state = random_density_matrix(4, seed=self.SEED)
        dual = dual_from_marginal_probabilities(prod_povm, state)
        assert dual.is_dual_to(prod_povm)

    def test_marginal_dual(self, subtests):
        """Test that the method constructs a valid dual."""
        # Load the circuit that was obtained through:
        #   from qiskit.circuit.random import random_circuit
        #   qc = random_circuit(num_qubits=2, depth=1, measure=False, seed=30)
        # for qiskit==1.1.1
        with open("test/post_processor/random_circuit_qubits=2_depth=1_seed=30.qpy", "rb") as file:
            qc = qpy.load(file)[0]
        num_qubits = qc.num_qubits
        bias = np.asarray([0.5, 0.25, 0.25])
        angles = np.asarray(
            [
                [2.01757238, -1.85001671, 2.52155716, 0.45636669, 1.17175533, -0.48263278],
                [0.0, 0.0, 1.57079633, -2.35619449, 1.57079633, -0.78539816],
            ]
        )

        measurement = RandomizedProjectiveMeasurements(
            num_qubits, bias=bias, angles=angles, seed=self.SEED
        )
        sampler = StatevectorSampler(seed=default_rng(self.SEED))
        povm_sampler = POVMSampler(sampler=sampler)
        job = povm_sampler.run([qc], shots=127, povm=measurement)
        pub_result = job.result()[0]

        observable = SparsePauliOp(
            ["XI", "YI", num_qubits * "Y", num_qubits * "Z"], coeffs=[1.3, 1.2, -1, 1.4]
        )

        post_processor = POVMPostProcessor(pub_result)

        with subtests.test("Test canonical dual."):
            exp_value, std = post_processor.get_expectation_value(observable)
            assert np.isclose(exp_value, -1.125222785367572)
            assert np.isclose(std, 0.5305722525114711)

        with subtests.test("Test marginal dual."):
            post_processor.dual = dual_from_marginal_probabilities(
                povm=post_processor.povm, state=Statevector(qc)
            )
            exp_value, std = post_processor.get_expectation_value(observable)
            assert np.isclose(exp_value, -1.4247893013196669)
            assert np.isclose(std, 0.4097543480314448)

        with subtests.test("Test threshold on marginal dual."):
            post_processor.dual = dual_from_marginal_probabilities(
                povm=post_processor.povm, state=Statevector(qc), threshold=1.0
            )
            assert np.isclose(exp_value, -1.4247893013196669)
            assert np.isclose(std, 0.4097543480314448)
