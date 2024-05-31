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

from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector

from povm_toolbox.post_processor.dual_optimizer import DUALOptimizer
from povm_toolbox.quantum_info import MultiQubitPOVM, ProductPOVM
from povm_toolbox.quantum_info.multi_qubit_dual import MultiQubitDUAL


class PPPStateOptimal(DUALOptimizer):
    """A common POVM result post-processor."""

    def set_state_optimal_dual(
        self,
        state: SparsePauliOp | DensityMatrix | Statevector,
    ) -> None:
        """TODO."""
        if isinstance(self.povm, MultiQubitPOVM):
            alphas = tuple(self.povm.get_prob(state))  # type: ignore
            self._dual = MultiQubitDUAL.build_dual_from_frame(self.povm, alphas=alphas)
            return
        if isinstance(self.povm, ProductPOVM):
            raise NotImplementedError
        raise TypeError
