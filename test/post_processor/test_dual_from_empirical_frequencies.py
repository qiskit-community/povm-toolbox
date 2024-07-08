# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the `dual_from_empirical_frequencies` function."""

from copy import deepcopy
from unittest import TestCase

import numpy as np
from povm_toolbox.library import (
    ClassicalShadows,
    RandomizedProjectiveMeasurements,
)
from povm_toolbox.post_processor import (
    POVMPostProcessor,
    dual_from_empirical_frequencies,
)
from povm_toolbox.quantum_info import MultiQubitPOVM
from povm_toolbox.sampler import POVMSampler
from qiskit import qpy
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import (
    DensityMatrix,
    Operator,
    SparsePauliOp,
)


class TestDualFromEmpiricalFrequencies(TestCase):
    """Test that we can construct optimal dual of a POVM from empirical frequencies."""

    SEED = 30

    def setUp(self) -> None:
        super().setUp()
        # Load the circuit that was obtained through:
        #   from qiskit.circuit.random import random_circuit
        #   qc = random_circuit(num_qubits=2, depth=1, measure=False, seed=30)
        # for qiskit==1.1.1
        with open("test/post_processor/random_circuit_qubits=2_depth=1_seed=30.qpy", "rb") as file:
            qc = qpy.load(file)[0]
        num_qubits = qc.num_qubits
        bias = np.array([0.5, 0.25, 0.25])
        angles = np.array(
            [
                [2.01757238, -1.85001671, 2.52155716, 0.45636669, 1.17175533, -0.48263278],
                [0.0, 0.0, 1.57079633, -2.35619449, 1.57079633, -0.78539816],
            ]
        )

        measurement = RandomizedProjectiveMeasurements(
            num_qubits, bias=bias, angles=angles, seed=self.SEED
        )
        povm_sampler = POVMSampler(sampler=StatevectorSampler(seed=self.SEED))
        job = povm_sampler.run([qc], shots=127, povm=measurement)
        pub_result = job.result()[0]
        self.post_processor = POVMPostProcessor(pub_result)

    def test_empirical_dual(self):
        """Test that the method constructs a valid dual."""
        observable = SparsePauliOp(["XI", "YI", 2 * "Y", 2 * "Z"], coeffs=[1.3, 1.2, -1, 1.4])

        with self.subTest("Test canonical dual."):
            exp_value, std = self.post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, -1.920301957227794)
            self.assertAlmostEqual(std, 0.468920055605158)

        with self.subTest("Test empirical dual with default arguments."):
            self.post_processor.dual = dual_from_empirical_frequencies(
                povm_post_processor=self.post_processor
            )
            exp_value, std = self.post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, -4.156136105226559)
            self.assertAlmostEqual(std, 0.10312825405831737)

        with self.subTest("Test empirical dual with lists arguments."):
            self.post_processor.dual = dual_from_empirical_frequencies(
                povm_post_processor=self.post_processor,
                loc=0,
                bias=[6, 6],
                ansatz=[
                    DensityMatrix(np.eye(2) / 2),
                    DensityMatrix(np.array([[2, 0], [1, 0]]) / 3),
                ],
            )
            exp_value, std = self.post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, -4.16623022578294)
            self.assertAlmostEqual(std, 0.1035227891358374)

        with self.subTest(
            "Test empirical dual with `bias` and `ansatz` arguments repeated for all qubits."
        ):
            self.post_processor.dual = dual_from_empirical_frequencies(
                povm_post_processor=self.post_processor,
                loc=0,
                bias=15,
                ansatz=SparsePauliOp(["I"], coeffs=np.array([0.5])),
            )
            exp_value, std = self.post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, -4.0202588245449125)
            self.assertAlmostEqual(std, 0.11724423989411743)

    def test_errors_raised(self):
        """Test that the method raises the appropriate errors when suitable."""
        with self.subTest("Error if ``povm`` is not a ``ProductPOVM`.") and self.assertRaises(
            NotImplementedError
        ):
            post_processor_2 = deepcopy(self.post_processor)
            post_processor_2._povm = MultiQubitPOVM([Operator(np.eye(4))])
            _ = dual_from_empirical_frequencies(post_processor_2)

        with self.subTest("Error if length of `bias` is invalid.") and self.assertRaises(
            ValueError
        ):
            _ = dual_from_empirical_frequencies(self.post_processor, bias=[1.0])

        with self.subTest("Error if length of `ansatz` is invalid.") and self.assertRaises(
            ValueError
        ):
            _ = dual_from_empirical_frequencies(
                self.post_processor, ansatz=[DensityMatrix(np.eye(2) / 2)]
            )

        with self.subTest("Error if default ``loc`` is not applicable.") and self.assertRaises(
            ValueError
        ):
            qc = QuantumCircuit(2)
            qc.ry(theta=Parameter("theta"), qubit=0)
            povm_sampler = POVMSampler(sampler=StatevectorSampler(seed=self.SEED))
            measurement = ClassicalShadows(num_qubits=2, seed=self.SEED)
            job = povm_sampler.run([(qc, np.array([0, np.pi]))], shots=16, povm=measurement)
            pub_result = job.result()[0]
            post_processor_3 = POVMPostProcessor(pub_result)
            _ = dual_from_empirical_frequencies(post_processor_3)
