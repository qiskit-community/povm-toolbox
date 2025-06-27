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
from povm_toolbox.quantum_info.multi_qubit_povm import MultiQubitPOVM
from povm_toolbox.quantum_info.single_qubit_povm import SingleQubitPOVM
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator


class TestMultiQubitPOVM:
    """Test that we can create valid POVM and get warnings if invalid."""

    def test_invalid_operators(self, subtests):
        """Test that errors are correctly raised if invalid operators are supplied."""
        with subtests.test("Operators with negative eigenvalues") and pytest.raises(ValueError):
            op = np.asarray([[-0.5, 0], [0, 0]])
            _ = MultiQubitPOVM(list_operators=[Operator(op), Operator(np.eye(2) - op)])
        with subtests.test("Operators not summing up to identity") and pytest.raises(ValueError):
            _ = MultiQubitPOVM(
                list_operators=[0.9 * Operator.from_label("0"), Operator.from_label("1")]
            )
        with subtests.test("Non-square operators") and pytest.raises(ValueError):
            _ = MultiQubitPOVM(
                [
                    Operator(np.asarray([[1, 0, 0], [0, 0, 0]])),
                    Operator(np.asarray([[0, 0, 0], [0, 1, 0]])),
                ]
            )

    def test_num_outcomes(self):
        """Test the number of outcomes, with both `num_outcomes` attribute and `__len__` method."""
        for n in range(1, 10):
            for dim in range(1, 10):
                povm = MultiQubitPOVM(n * [Operator(1.0 / n * np.eye(dim))])
                assert n == povm.num_outcomes
                assert n == len(povm)
                assert n == len(povm.operators)

    def test_informationally_complete(self, subtests):
        """Test whether a POVM is informationally complete or not."""
        with subtests.test("SIC-POVM"):
            import cmath

            vecs = np.sqrt(1.0 / 2.0) * np.asarray(
                [
                    [1, 0],
                    [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0)],
                    [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0) * cmath.exp(2.0j * np.pi / 3)],
                    [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0) * cmath.exp(4.0j * np.pi / 3)],
                ]
            )
            sic_povm = MultiQubitPOVM.from_vectors(vecs)
            assert sic_povm.informationally_complete

        with subtests.test("CS-POVM"):
            coef = 1.0 / 3.0
            cs_povm = MultiQubitPOVM(
                [
                    coef * Operator.from_label("0"),
                    coef * Operator.from_label("1"),
                    coef * Operator.from_label("+"),
                    coef * Operator.from_label("-"),
                    coef * Operator.from_label("r"),
                    coef * Operator.from_label("l"),
                ]
            )
            assert cs_povm.informationally_complete

        with subtests.test("Non IC-POVM"):
            coef = 1.0 / 2.0
            povm = MultiQubitPOVM(
                [
                    coef * Operator.from_label("0"),
                    coef * Operator.from_label("1"),
                    coef * Operator.from_label("+"),
                    coef * Operator.from_label("-"),
                ]
            )
            assert not povm.informationally_complete

    def test_repr(self, subtests):
        """Test the ``__repr__`` method."""
        with subtests.test("Single-qubit case."):
            povm = MultiQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
            assert povm.__repr__() == f"MultiQubitPOVM<2> at {hex(id(povm))}"
            povm = MultiQubitPOVM(
                2 * [0.5 * Operator.from_label("0"), 0.5 * Operator.from_label("1")]
            )
            assert povm.__repr__() == f"MultiQubitPOVM<4> at {hex(id(povm))}"
            povm = SingleQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
            assert povm.__repr__() == f"SingleQubitPOVM<2> at {hex(id(povm))}"
        with subtests.test("Multi-qubit case."):
            povm = MultiQubitPOVM(
                [
                    Operator.from_label("00"),
                    Operator.from_label("01"),
                    Operator.from_label("10"),
                    Operator.from_label("11"),
                ]
            )
            assert povm.__repr__() == f"MultiQubitPOVM(num_qubits=2)<4> at {hex(id(povm))}"

    def test_pauli_operators(self, subtests):
        """Test errors are raised  correctly for the ``pauli_operators`` attribute."""
        povm = MultiQubitPOVM([Operator(np.eye(3))])
        with subtests.test("Non-qubit operators") and pytest.raises(QiskitError):
            _ = povm.pauli_operators

    def test_analysis(self, subtests):
        povm = MultiQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
        operator = Operator([[0.8, 0], [0, 0.2]])
        with subtests.test("Get a single frame coefficient."):
            assert povm.analysis(operator, 0) == 0.8
            assert povm.analysis(operator, 1) == 0.2
        with subtests.test("Get a set of frame coefficients."):
            frame_coefficients = povm.analysis(operator, set([0]))
            assert isinstance(frame_coefficients, dict)
            assert frame_coefficients[0] == 0.8
            frame_coefficients = povm.analysis(operator, set([1, 0]))
            assert isinstance(frame_coefficients, dict)
            assert frame_coefficients[0] == 0.8
            assert frame_coefficients[1] == 0.2
        with subtests.test("Get all frame coefficients."):
            frame_coefficients = povm.analysis(operator)
            assert isinstance(frame_coefficients, np.ndarray)
            assert np.allclose(frame_coefficients, np.asarray([0.8, 0.2]))
        with subtests.test("Invalid type for ``frame_op_idx``.") and pytest.raises(ValueError):
            _ = povm.analysis(operator, (0, 1))

    def test_draw_bloch(self):
        with pytest.raises(NotImplementedError):
            povm = MultiQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
            povm.draw_bloch()
