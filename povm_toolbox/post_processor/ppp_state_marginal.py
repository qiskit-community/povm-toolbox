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

from povm_toolbox.post_processor.dual_optimizer import DUALOptimizer
from povm_toolbox.quantum_info import ProductPOVM
from povm_toolbox.quantum_info.product_dual import ProductDUAL


class PPPStateMarginal(DUALOptimizer):
    """A common POVM result post-processor."""

    def set_marginal_probabilities_dual(
        self,
        state: SparsePauliOp | DensityMatrix | Statevector,
    ) -> None:
        """TODO."""
        if not isinstance(self.povm, ProductPOVM):
            raise NotImplementedError
        axes = np.arange(len(self.povm.sub_systems), dtype=int)
        joint_prob: np.ndarray = self.povm.get_prob(state)  # type: ignore
        alphas = []
        for qubit_idx in self.povm.sub_systems:
            marg_prob = joint_prob.sum(axis=tuple(np.delete(axes, [qubit_idx])))
            if np.any(np.absolute(marg_prob) < 1e-5):
                marg_prob += 1e-5
            alphas.append(tuple(marg_prob))
        self._dual = ProductDUAL.build_dual_from_frame(self.povm, alphas=tuple(alphas))
