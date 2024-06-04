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

from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.quantum_info import MultiQubitPOVM
from povm_toolbox.quantum_info.multi_qubit_dual import MultiQubitDUAL


class PPPStateOptimal(POVMPostProcessor):
    """A POVM post-processor that leverages state knowledge to set the dual frame."""

    def set_state_optimal_dual(
        self,
        state: SparsePauliOp | DensityMatrix | Statevector,
    ) -> None:
        """Set the dual frame based on the outcome distribution of a supplied state.

        This methods constructs a joint dual frame where the alpha-parameters are
        set as the outcome probabilities of the supplied state. It can be shown
        that this is the dual frame minimizing the variance of the estimator (irrespective
        of the observable to estimate) if outcomes are sampled from this state.

        Args:
            state: state from which to compute the outcome probabilities.

        Raises:
            NotImplementedError: if ``self.povm`` is not a :class:`povm_toolbox.quantum_info.product_povm.MultiQubitPOVM`
                instance.
        """
        if not isinstance(self.povm, MultiQubitPOVM):
            # TODO : implement for ProductPOVM
            raise NotImplementedError(
                "This method is only implemented for `povm_toolbox.quantum_info.product_povm.MultiQubitPOVM`."
            )
        alphas = tuple(self.povm.get_prob(state))  # type: ignore
        self._dual = MultiQubitDUAL.build_dual_from_frame(self.povm, alphas=alphas)
