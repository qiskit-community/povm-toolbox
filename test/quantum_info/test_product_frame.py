# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the ProductFrame class."""

from unittest import TestCase

import numpy as np
from povm_toolbox.quantum_info.multi_qubit_frame import MultiQubitFrame
from povm_toolbox.quantum_info.product_frame import ProductFrame
from qiskit.quantum_info import Operator, SparsePauliOp


class TestProductFrame(TestCase):
    """Tests for the ``ProductFrame`` class."""

    def test_analysis(self):
        """Test that the ``analysis`` method works correctly."""
        paulis = ["I", "X", "Y", "Z"]
        frame_1_qubit = MultiQubitFrame([Operator.from_label(label) for label in paulis])

        num_qubit_max = 3

        for num_qubit in range(1, num_qubit_max + 1):
            frame_product = ProductFrame.from_list(frames=num_qubit * [frame_1_qubit])

            product_paulis = []
            for idx in np.ndindex(frame_product.shape):
                product_paulis.append("".join([paulis[i] for i in idx]))

            frame_n_qubit = ProductFrame.from_list(
                [MultiQubitFrame([Operator.from_label(label) for label in product_paulis])]
            )

            for i, pauli_string in enumerate(product_paulis):
                decomposition_weights_n_qubit = frame_n_qubit.analysis(
                    SparsePauliOp(pauli_string[::-1])
                )
                decomposition_weights_product = frame_product.analysis(
                    SparsePauliOp(pauli_string[::-1])
                ).flatten()
                check = np.zeros(len(product_paulis))
                check[i] = 2**num_qubit
                self.assertTrue(np.allclose(decomposition_weights_n_qubit, check))
                self.assertTrue(np.allclose(decomposition_weights_product, check))

            decomposition_weights_n_qubit = frame_n_qubit.analysis(
                SparsePauliOp(
                    [pauli_string[::-1] for pauli_string in product_paulis],
                    np.ones(len(product_paulis)),
                )
            )
            decomposition_weights_product = frame_product.analysis(
                SparsePauliOp(
                    [pauli_string[::-1] for pauli_string in product_paulis],
                    np.ones(len(product_paulis)),
                )
            ).flatten()
            check = np.ones(len(product_paulis)) * 2**num_qubit
            self.assertTrue(np.allclose(decomposition_weights_n_qubit, check))
            self.assertTrue(np.allclose(decomposition_weights_product, check))
