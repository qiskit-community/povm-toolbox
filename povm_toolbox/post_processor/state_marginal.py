# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TODO."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector

from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.quantum_info import ProductPOVM
from povm_toolbox.quantum_info.product_dual import ProductDUAL


class StateMarginal(POVMPostProcessor):
    """A POVM post-processor that leverages the marginal outcome distributions to set the dual frame."""

    def set_marginal_probabilities_dual(
        self,
        state: SparsePauliOp | DensityMatrix | Statevector,
    ) -> None:
        """Set the dual frame based on the marginal distribution of a supplied state.

        This methods constructs a product dual frame where each local dual frame
        is parametrized  with the alpha-parameters set as the marginal outcome
        probabilities of the supplied state.

        Args:
            state: state from which to compute the marginal outcome probabilities.

        Raises:
            NotImplementedError: if ``self.povm`` is not a :class:`povm_toolbox.quantum_info.product_povm.ProductPOVM`
                instance.
        """
        if not isinstance(self.povm, ProductPOVM):
            raise NotImplementedError(
                "This method is only implemented for `povm_toolbox.quantum_info.product_povm.ProductPOVM`."
            )
        axes = np.arange(len(self.povm.sub_systems), dtype=int)
        joint_prob: np.ndarray = self.povm.get_prob(state)  # type: ignore
        alphas = []
        for qubit_idx in self.povm.sub_systems:
            marg_prob = joint_prob.sum(axis=tuple(np.delete(axes, [qubit_idx])))
            if np.any(np.absolute(marg_prob) < 1e-5):
                marg_prob += 1e-5
            alphas.append(tuple(marg_prob))
        self._dual = ProductDUAL.build_dual_from_frame(self.povm, alphas=tuple(alphas))
