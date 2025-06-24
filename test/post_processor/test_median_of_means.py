# (C) Copyright IBM 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the MedianOfMeans class."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.random import default_rng
from povm_toolbox.library import ClassicalShadows
from povm_toolbox.post_processor import MedianOfMeans
from povm_toolbox.sampler import POVMSampler
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import SparsePauliOp


class TestMedianOfMeans:
    """Test the methods and attributes of the :class:`.MedianOfMeans class`."""

    SEED = 3433

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

    def test_init_errors(self, subtests):
        """Test that the ``__init__`` method raises errors correctly."""
        # sanity check
        with subtests.test("Valid initialization."):
            post_processor = MedianOfMeans(
                self.pub_result, num_batches=5, seed=default_rng(self.SEED)
            )
            assert post_processor.num_batches == 5
            assert np.isclose(post_processor.delta_confidence, 0.1641699972477976)
        with subtests.test("Invalid type for ``seed``.") and pytest.raises(TypeError):
            MedianOfMeans(self.pub_result, seed=1.2)

    @pytest.mark.parametrize(
        [
            "kwargs",
            "expected_num_batches",
            "expected_delta",
            "expected_exp_val",
            "expected_eps",
        ],
        [
            (
                {"seed": 3433},
                8,
                0.03663127777746836,
                1.125,
                2.9154759474226504,
            ),
            (
                {"seed": 3433, "upper_delta_confidence": 0.1},
                6,
                0.09957413673572789,
                -0.75,
                2.6076809620810595,
            ),
            (
                {"seed": 3433, "num_batches": 4},
                4,
                0.2706705664732254,
                0.0,
                2.0615528128088303,
            ),
            (
                {"num_batches": 8},
                8,
                0.03663127777746836,
                None,  # cannot assert the expectational value with a random seed
                2.9154759474226504,
            ),
        ],
    )
    def test_get_expectation_value(
        self,
        kwargs: dict[str, Any],
        expected_num_batches: int,
        expected_delta: float,
        expected_exp_val: float | None,
        expected_eps: float,
    ):
        """Test that the ``get_expectation_value`` method works correctly."""
        post_processor = MedianOfMeans(self.pub_result, **kwargs)
        observable = SparsePauliOp(["ZZ", "XX", "YY"], coeffs=[1, 2, 3])
        exp_val, epsilon_coef = post_processor.get_expectation_value(observable)
        assert post_processor.num_batches == expected_num_batches
        assert np.isclose(post_processor.delta_confidence, expected_delta)
        if expected_exp_val is not None:
            assert np.isclose(exp_val, expected_exp_val)
        assert np.isclose(epsilon_coef, expected_eps)

    def test_delta_confidence(self):
        """Test that the ``delta_confidence`` property and setter work correctly."""
        post_processor = MedianOfMeans(self.pub_result, num_batches=5)
        assert post_processor.num_batches == 5
        assert np.isclose(post_processor.delta_confidence, 0.1641699972477976)
        post_processor.delta_confidence = 0.098
        assert post_processor.num_batches == 7
        assert np.isclose(post_processor.delta_confidence, 0.060394766844637)
