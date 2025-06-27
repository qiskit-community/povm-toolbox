# (C) Copyright IBM 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the MultiQubitPOVM class."""

import numpy as np
import pytest
from povm_toolbox.quantum_info.multi_qubit_povm import MultiQubitFrame
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator


class TestMultiQubitFrame:
    """Test that we can create valid frame and get warnings if invalid."""

    def test_invalid_operators(self, subtests):
        """Test that errors are correctly raised if invalid operators are supplied."""
        with subtests.test("Non Hermitian operators") and pytest.raises(ValueError):
            ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(-1, 1, (6, 2, 2))
            while np.abs(ops[0, 0, 0].imag) < 1e-6:
                ops = np.random.uniform(-1, 1, (6, 2, 2)) + 1.0j * np.random.uniform(
                    -1, 1, (6, 2, 2)
                )
            _ = MultiQubitFrame(list_operators=[Operator(op) for op in ops])

    def test_dimension(self):
        """Test dimension attribute."""
        for dim in range(1, 10):
            frame = MultiQubitFrame(3 * [Operator(1.0 / 3.0 * np.eye(dim))])
            assert dim == frame.dimension
            assert dim == frame._dimension
            assert dim == dim, frame.operators[1].dim

    def test_getitem(self, subtests):
        """Test the ``__getitem__`` method."""
        n = 6
        frame = MultiQubitFrame(
            [Operator(2 * i / (n * (n + 1)) * np.eye(4)) for i in range(1, n + 1)]
        )
        with subtests.test("1d shape"):
            assert isinstance(frame[2], Operator)
            assert isinstance(frame[1:2], list)
            assert isinstance(frame[:2], list)
            assert frame[0] == frame.operators[0]
            assert frame[3:] == [frame.operators[3], frame.operators[4], frame.operators[5]]
            assert frame[::2] == [frame.operators[0], frame.operators[2], frame.operators[4]]
            assert frame[2::-1] == [frame.operators[2], frame.operators[1], frame.operators[0]]
        with subtests.test("2d shape"):
            frame.shape = (2, 3)
            assert isinstance(frame[0, 2], Operator)
            assert frame[0, 0] == frame.operators[0]
            assert frame[(0, 1)] == frame.operators[1]
            assert frame[1, 2] == frame.operators[5]
        with subtests.test("index too short") and pytest.raises(ValueError):
            frame[0]
        with subtests.test("index too long") and pytest.raises(ValueError):
            frame[0, 0, 0]
        with subtests.test("out of bound index") and pytest.raises(ValueError):
            frame[0, 4]

    def test_informationally_complete(self, subtests):
        """Test whether a frame is informationally complete or not."""
        paulis = ["I", "X", "Y", "Z"]
        with subtests.test("IC frame"):
            frame = MultiQubitFrame([Operator.from_label(label) for label in paulis])
            assert frame.informationally_complete

        with subtests.test("non-IC frame"):
            frame = MultiQubitFrame([Operator.from_label(label) for label in paulis[1:]])
            assert not frame.informationally_complete

    def test_repr(self, subtests):
        """Test the ``__repr__`` method."""
        with subtests.test("Single-qubit case."):
            frame = MultiQubitFrame([Operator.from_label("0"), Operator.from_label("1")])
            assert frame.__repr__() == f"MultiQubitFrame<2> at {hex(id(frame))}"
            frame = MultiQubitFrame(
                2 * [0.5 * Operator.from_label("0"), 0.5 * Operator.from_label("1")]
            )
            assert frame.__repr__() == f"MultiQubitFrame<4> at {hex(id(frame))}"
            frame.shape = (2, 2)
            assert frame.__repr__() == f"MultiQubitFrame<2,2> at {hex(id(frame))}"
        with subtests.test("Multi-qubit case."):
            frame = MultiQubitFrame(
                [
                    Operator.from_label("II"),
                    Operator.from_label("IX"),
                    Operator.from_label("XI"),
                    Operator.from_label("XX"),
                ]
            )
            assert frame.__repr__() == f"MultiQubitFrame(num_qubits=2)<4> at {hex(id(frame))}"
            frame.shape = (2, 2)
            assert frame.__repr__() == f"MultiQubitFrame(num_qubits=2)<2,2> at {hex(id(frame))}"

    def test_pauli_operators(self, subtests):
        """Test errors are raised  correctly for the ``pauli_operators`` attribute."""
        frame = MultiQubitFrame([Operator(np.eye(3))])
        with subtests.test("Non-qubit operators") and pytest.raises(QiskitError):
            _ = frame.pauli_operators

    def test_analysis(self, subtests):
        """Test that the ``analysis`` method works correctly."""
        frame = MultiQubitFrame([Operator.from_label(label) for label in ["0", "1", "I", "Z"]])
        frame_shaped = MultiQubitFrame(
            [Operator.from_label(label) for label in ["0", "1", "I", "Z"]], shape=(2, 2)
        )
        operator = Operator([[0.8, 0], [0, 0.2]])
        with subtests.test("Get a single frame coefficient."):
            assert np.allclose(frame.analysis(operator, 0), 0.8)
            assert np.allclose(frame.analysis(operator, 1), 0.2)
            assert np.allclose(frame.analysis(operator, 2), 1.0)
            assert np.allclose(frame.analysis(operator, 3), 0.6)
            assert np.allclose(frame_shaped.analysis(operator, (0, 0)), 0.8)
            assert np.allclose(frame_shaped.analysis(operator, (0, 1)), 0.2)
            assert np.allclose(frame_shaped.analysis(operator, (1, 0)), 1.0)
            assert np.allclose(frame_shaped.analysis(operator, (1, 1)), 0.6)
        with subtests.test("Get a set of frame coefficients."):
            frame_coefficients = frame.analysis(operator, set([0]))
            assert isinstance(frame_coefficients, dict)
            assert np.allclose(frame_coefficients[0], 0.8)
            frame_coefficients = frame.analysis(operator, set([1, 0]))
            assert isinstance(frame_coefficients, dict)
            assert np.allclose(frame_coefficients[0], 0.8)
            assert np.allclose(frame_coefficients[1], 0.2)
            frame_coefficients = frame_shaped.analysis(operator, set([(1, 0), (0, 0)]))
            assert isinstance(frame_coefficients, dict)
            assert np.allclose(frame_coefficients[0, 0], 0.8)
            assert np.allclose(frame_coefficients[(1, 0)], 1.0)
        with subtests.test("Get all frame coefficients."):
            frame_coefficients = frame.analysis(operator)
            assert isinstance(frame_coefficients, np.ndarray)
            assert np.allclose(frame_coefficients, np.asarray([0.8, 0.2, 1.0, 0.6]))
            frame_coefficients = frame_shaped.analysis(operator)
            assert isinstance(frame_coefficients, np.ndarray)
            assert np.allclose(frame_coefficients, np.asarray([0.8, 0.2, 1.0, 0.6]))
        with subtests.test("Invalid value for ``frame_op_idx``.") and pytest.raises(ValueError):
            _ = frame.analysis(operator, (0, 1))
        with subtests.test("Invalid type for ``frame_op_idx``.") and pytest.raises(TypeError):
            _ = frame.analysis(operator, [0])

    def test_shape(self, subtests):
        """Test that the ``shape`` property works correctly."""
        paulis = ["I", "X", "Y", "Z"]
        with subtests.test("Test works correctly"):
            frame = MultiQubitFrame([Operator.from_label(label) for label in paulis], shape=(2, 2))
            assert frame.shape == (2, 2)
            frame.shape = (4, 1)
            assert frame.shape == (4, 1)
        with subtests.test("Test raises errors correctly") and pytest.raises(ValueError):
            frame = MultiQubitFrame([Operator.from_label(label) for label in paulis])
            frame.shape = (2, 3)
        with subtests.test("Index incompatible with shape") and pytest.raises(ValueError):
            frame = MultiQubitFrame([Operator.from_label(label) for label in paulis], shape=(2, 2))
            frame._ravel_index(0)
