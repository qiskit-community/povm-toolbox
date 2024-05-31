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


class PPPEmpiricalFrequencies(DUALOptimizer):
    """A common POVM result post-processor."""

    def set_empirical_frequencies_dual(
        self,
        loc,
        bias: float | list[float] | None = None,
        ansatz: list[SparsePauliOp | DensityMatrix | Statevector] | None = None,
    ) -> None:
        """TODO."""
        if not isinstance(self.povm, ProductPOVM):
            raise NotImplementedError(
                "This method is only implemented for `povm_toolbox.quantum_info.product_povm.ProductPOVM`."
            )

        counts = self.counts[loc]
        marginals = [np.zeros(subsystem_shape) for subsystem_shape in self.povm.shape]

        # Computing marginals
        shots = sum(counts.values())
        for outcome, count in counts.items():
            for i, k_i in enumerate(outcome):
                marginals[i][k_i] += count

        alphas = []
        # Computing alphas for each subsystem
        for i, sub_system in enumerate(self.povm.sub_systems):
            sub_povm = self.povm[sub_system]
            dim = sub_povm.dimension
            ansatz_state = DensityMatrix(np.eye(dim) / dim) if ansatz is None else ansatz[i]
            sub_bias = (
                sub_povm.n_outcomes
                if bias is None
                else (bias[i] if isinstance(bias, list) else bias)
            )
            sub_alphas = marginals[i] + sub_bias * sub_povm.get_prob(ansatz_state)  # type: ignore
            alphas.append(tuple(sub_alphas / (shots + sub_bias)))

        # Building ProductDUAL from marginals
        self._dual = ProductDUAL.build_dual_from_frame(self.povm, alphas=tuple(alphas))
