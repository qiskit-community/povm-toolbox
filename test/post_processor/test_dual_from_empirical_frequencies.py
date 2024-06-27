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

from unittest import TestCase

import numpy as np
from numpy.random import default_rng
from povm_toolbox.library import (
    RandomizedProjectiveMeasurements,
)
from povm_toolbox.post_processor import (
    POVMPostProcessor,
    dual_from_empirical_frequencies,
)
from povm_toolbox.sampler import POVMSampler
from qiskit.circuit.random import random_circuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import (
    DensityMatrix,
    SparsePauliOp,
)


class TestDualFromEmpiricalFrequencies(TestCase):
    """Test that we can construct optimal dual of a POVM from empirical frequencies."""

    def test_empirical_dual(self):
        """Test that the method constructs a valid dual."""
        qc = random_circuit(2, 1, measure=False, seed=12)
        rng = default_rng(96568)
        num_qubits = qc.num_qubits
        bias = np.array([0.5, 0.25, 0.25])
        angles = np.array(
            [
                [2.01757238, -1.85001671, 2.52155716, 0.45636669, 1.17175533, -0.48263278],
                [0.0, 0.0, 1.57079633, -2.35619449, 1.57079633, -0.78539816],
            ]
        )

        measurement = RandomizedProjectiveMeasurements(
            num_qubits, bias=bias, angles=angles, seed_rng=rng
        )
        sampler = StatevectorSampler(seed=rng)
        povm_sampler = POVMSampler(sampler=sampler)
        job = povm_sampler.run([qc], shots=127, povm=measurement)
        pub_result = job.result()[0]

        observable = SparsePauliOp(
            ["XI", "YI", num_qubits * "Y", num_qubits * "Z"], coeffs=[1.3, 1.2, -1, 1.4]
        )

        post_processor = POVMPostProcessor(pub_result)

        with self.subTest("Test canonical dual."):
            exp_value, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, 1.255719986997053)
            self.assertAlmostEqual(std, 0.5312929210892221)

        with self.subTest("Test empirical dual with default arguments."):
            post_processor.dual = dual_from_empirical_frequencies(
                povm_post_processor=post_processor
            )
            exp_value, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, 0.9128662937761666)
            self.assertAlmostEqual(std, 0.40472771240833644)

        with self.subTest("Test empirical dual with lists arguments."):
            post_processor.dual = dual_from_empirical_frequencies(
                povm_post_processor=post_processor,
                loc=0,
                bias=[6, 6],
                ansatz=[DensityMatrix(np.eye(2) / 2), DensityMatrix(np.eye(2) / 2)],
            )
            exp_value, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, 0.9128662937761666)
            self.assertAlmostEqual(std, 0.40472771240833644)

        with self.subTest(
            "Test empirical dual with `bias` and `ansatz` arguments repeated for all qubits."
        ):
            post_processor.dual = dual_from_empirical_frequencies(
                povm_post_processor=post_processor,
                loc=0,
                bias=6,
                ansatz=SparsePauliOp(["I"], coeffs=np.array([0.5])),
            )
            exp_value, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, 0.9128662937761666)
            self.assertAlmostEqual(std, 0.40472771240833644)

    def test_errors_raised(self):
        """Test that the method raises the appropriate errors when suitable."""
        with self.subTest("Error if ``povm`` is not a ``ProductPOVM`."):
            # TODO
            pass

        with self.subTest("Error if ``loc`` is invalid."):
            # TODO
            pass

        with self.subTest("Error if length of `bias` is invalid."):
            # TODO
            pass

        with self.subTest("Error if length of `ansatz` is invalid."):
            # TODO
            pass
