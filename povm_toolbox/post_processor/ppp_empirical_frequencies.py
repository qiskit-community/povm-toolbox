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


class PPPEmpiricalFrequencies(POVMPostProcessor):
    """A POVM post-processor that leverages the outcome frequencies to set the dual frame."""

    def set_empirical_frequencies_dual(
        self,
        loc: int | tuple[int, ...] | None = None,
        bias: float | list[float] | None = None,
        ansatz: list[SparsePauliOp | DensityMatrix | Statevector] | None = None,
    ) -> None:
        """Set the dual frame based on the frequencies of the sampled outcomes.

        Given outcomes sampled from a product POVM, each local dual frame is parametrized
        with the alpha-parameters set as the marginal outcome frequencies. For stability,
        the (local) empirical frequencies can be biased towards the (marginal) outcome
        probabilities of an ansatz state.

        Args:
            loc: index of the results to use. This is relevant if multiple sets of
                parameter values were supplied to the sampler in the same PUB. If None,
                it is assumed that the supplied circuit was not parametrized or that a
                unique set of parameter values was supplied. In this case, ``loc`` is
                trivially set to 0.
            bias: the strength of the bias towards the outcome distribution of the
                ``ansatz`` state.  If it is a ``float``, the same bias is  applied
                to each (local) sub-system. If it is a list of ``float``, a specific
                bias is applied to each sub-system. If None, the bias for each sub-
                system is set to be the number of outcomes of the POVM acting on this
                sub-system.
            ansatz: list of quantum states for each local sub-system, from which the
                local outcome probability distributions are computed for each sub-
                system. The empirical marginal frequencies are biased towards these
                distributions. If None, the fully mixed state is used for each-subsystem.

        Raises:
            ValueError: if `loc` is None and that the POVM post-processor stores more
                than one counter (i.e., multiple sets of parameter values were
                supplied to the sampler in a single pub).
            NotImplementedError: if ``self.povm`` is not a :class:`povm_toolbox.quantum_info.product_povm.ProductPOVM`
                instance.
        """
        if not isinstance(self.povm, ProductPOVM):
            raise NotImplementedError(
                "This method is only implemented for `povm_toolbox.quantum_info.product_povm.ProductPOVM`."
            )

        if loc is None:
            if self.counts.shape == (1,):
                loc = (0,)
            else:
                raise ValueError(
                    "`loc` has to be specified if the POVM post-processor stores"
                    " more than one counter (i.e., if multiple sets of parameter"
                    " values were supplied to the sampler in a single pub). The"
                    f" array of counters is of shape {self.counts.shape}."
                )

        counts = self.counts[loc]
        marginals = [np.zeros(subsystem_shape) for subsystem_shape in self.povm.shape]

        # Computing marginals
        shots = sum(counts.values())
        for outcome, count in counts.items():
            for i, k_i in enumerate(outcome):
                marginals[i][k_i] += count / shots

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
            sub_alphas = shots * marginals[i] + sub_bias * sub_povm.get_prob(ansatz_state)  # type: ignore
            alphas.append(tuple(sub_alphas / (shots + sub_bias)))

        # Building ProductDUAL from marginals
        self._dual = ProductDUAL.build_dual_from_frame(self.povm, alphas=tuple(alphas))