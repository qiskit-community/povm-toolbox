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

from typing import Any

import numpy as np
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector

from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.quantum_info import MultiQubitPOVM, ProductPOVM
from povm_toolbox.quantum_info.base_dual import BaseDUAL
from povm_toolbox.quantum_info.multi_qubit_dual import MultiQubitDUAL
from povm_toolbox.quantum_info.product_dual import ProductDUAL
from povm_toolbox.sampler.povm_sampler_result import POVMPubResult


class DUALOptimizer(POVMPostProcessor):
    """A common POVM result post-processor."""

    def __init__(self, povm_sample: POVMPubResult, dual: BaseDUAL | None = None) -> None:
        """TODO."""
        super().__init__(povm_sample, dual)
        self._gammas: dict[tuple[int, ...], float] = {}

    def optimize(self, **options) -> None:
        """TODO."""
        if (state := options.get("state", None)) is not None:
            if options.get("use_marginal", False):
                self.set_marginal_probabilities_dual(state)
            else:
                self.set_state_optimal_dual(state)
        elif (alphas := options.get("alphas", None)) is not None:
            dual_class = self.povm.default_dual_class
            self._dual = dual_class.build_dual_from_frame(self.povm, alphas=alphas)
        elif options.get("empirical_frequencies", False):
            self.set_empirical_frequencies_dual(0)

        if not self.dual.is_dual_to(self.povm):
            raise ValueError

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

    @property
    def gammas(self) -> dict[tuple[int, ...], float]:
        """TODO."""
        return self._gammas

    @gammas.setter
    def gammas(self, value: dict[tuple[int, ...], float]):
        """TODO."""
        self._gammas = value

    def _scalar_gauge(self, k):
        """TODO."""
        if isinstance(self.povm, ProductPOVM):
            gauge = self._gammas.get(k, 0.0)
            for l_idx, value in self._gammas.items():
                tmp_gauge = value
                for i, sub_system in enumerate(self.povm.sub_systems):
                    tmp_gauge *= np.real(
                        np.trace(self.dual[sub_system][k[i]] @ self.povm[sub_system][l_idx[i]])
                    )
                gauge -= tmp_gauge
            return gauge

    def get_decomposition_weights(
        self,
        observable: SparsePauliOp,
        outcome_idx: set[Any],
    ) -> dict[Any, float]:
        """TODO."""
        omegas = dict(self.dual.get_omegas(observable, outcome_idx))  # type: ignore
        if len(self.gammas) > 0:
            for outcome in omegas:
                omegas[outcome] += self._scalar_gauge(outcome)
        return omegas

    # def greedy_optimize_scalar_gauge(
    #     self, obs, outcomes: set[tuple[int, ...]] | None = None, loc=0
    # ):
    #     """TODO."""
    #     if not (isinstance(self.povm, ProductPOVM) and isinstance(self.dual, ProductDUAL)):
    #         raise NotImplementedError

    #     gauge_matrices = [
    #         np.empty((self.povm[sub_system].n_outcomes, self.povm[sub_system].n_outcomes))
    #         for sub_system in self.povm.sub_systems
    #     ]
    #     for i, sub_system in enumerate(self.povm.sub_systems):
    #         for k_i, l_i in np.ndindex(gauge_matrices[i].shape):
    #             gauge_matrices[i][k_i, l_i] = np.real(
    #                 np.trace(self.dual[sub_system][k_i] @ self.povm[sub_system][l_i])  # type: ignore
    #             )

    #     def F(gauge_matrices, outcome_k: tuple[int, ...], outcome_l: tuple[int, ...]):
    #         F_kl = float(outcome_k == outcome_l)
    #         trace_kl = 1.0
    #         for gauge_matrix_i, k_i, l_i in zip(gauge_matrices, outcome_k, outcome_l):
    #             trace_kl *= gauge_matrix_i[k_i, l_i]
    #         F_kl -= trace_kl
    #         return F_kl

    #     counts = self.counts[loc]
    #     weights = self.get_decomposition_weights(obs, set(counts.keys()))

    #     def get_pw2(gammas, counts, weights, gauge_matrices):
    #         pw2 = 0.0
    #         for outcome in counts:
    #             w = weights[outcome]
    #             for l_idx, gamma_l in gammas.items():
    #                 w += F(gauge_matrices, outcome, l_idx) * gamma_l
    #             pw2 += counts[outcome] * w**2
    #         return pw2

    #     if outcomes is None:
    #         n = 5
    #         outcomes_all = np.empty(shape=len(counts), dtype=object)
    #         pw2 = np.empty(shape=len(counts), dtype=float)
    #         for i, outcome in enumerate(counts):
    #             outcomes_all[i] = outcome
    #             pw2[i] = counts[outcome] * weights[outcome] ** 2
    #         ind = np.argpartition(pw2, -n)[-n:]
    #         outcomes_array = outcomes_all[ind]
    #     else:
    #         outcomes_all = np.empty(shape=len(outcomes), dtype=object)
    #         for i, outcome in enumerate(outcomes):
    #             outcomes_all[i] = outcome
    #         outcomes_array = outcomes_all

    #     n = len(outcomes_array)

    #     old_gammas = np.zeros(n)

    #     for i, outcome in enumerate(outcomes_array):
    #         old_gammas[i] = self.gammas.get(outcome, 0.0)
    #         self.gammas[outcome] = 0.0

    #     weights = self.get_decomposition_weights(obs, set(counts.keys()))

    #     b = np.zeros(n)
    #     hess_matrix = np.zeros((n, n))

    #     for outcome_k in counts:
    #         F_k = np.array(
    #             [F(gauge_matrices, outcome_k, outcome_i) for outcome_i in outcomes_array]
    #         )
    #         for i, F_ki in enumerate(F_k):
    #             b[i] += 2 * counts[outcome_k] * weights[outcome_k] * F_ki
    #             for j, F_kj in enumerate(F_k):
    #                 hess_matrix[i, j] += 2 * counts[outcome_k] * F_ki * F_kj

    #     def jac(x, counts, omegas, gauge_matrices, outcomes_ind, hess_matrix, b):
    #         return hess_matrix @ x + b

    #     def hess(x, counts, omegas, gauge_matrices, outcomes_ind, hess_matrix, b):
    #         return hess_matrix

    #     def fun(x, counts, omegas, gauge_matrices, outcomes_ind, hess_matrix, b):
    #         gammas = {outcome: x_i for outcome, x_i in zip(outcomes_ind, x)}
    #         return get_pw2(gammas, counts, omegas, gauge_matrices)

    #     def callback(x):
    #         print(fun(x, counts, weights, gauge_matrices, outcomes_array, hess_matrix, b))

    #     print(outcomes_array)

    #     res = minimize(
    #         fun,
    #         old_gammas,
    #         args=(counts, weights, gauge_matrices, outcomes_array, hess_matrix, b),
    #         method="trust-exact",
    #         jac=jac,
    #         hess=hess,
    #         callback=callback,
    #         options={"gtol": 1e-4, "disp": True},
    #     )
    #     print(f'   {"Optimized":<10}{res.fun}')

    #     for outcome, x_i in zip(outcomes_array, res.x):
    #         self.gammas[outcome] = self.gammas.get(outcome, 0.0) + x_i

    #     return {outcome: x_i for outcome, x_i in zip(outcomes_array, res.x)}

    # def greedy_exact_scalar_gauge(
    #     self, obs, outcomes: set[tuple[int, ...]] | None = None, loc=0
    # ):
    #     """TODO."""
    #     if not (isinstance(self.povm, ProductPOVM) and isinstance(self.dual, ProductDUAL)):
    #         raise NotImplementedError

    #     for outcome in outcomes:
    #         self.gammas[outcome] = 0.0

    #     gauge_matrices = [
    #         np.empty((self.povm[sub_system].n_outcomes, self.povm[sub_system].n_outcomes))
    #         for sub_system in self.povm.sub_systems
    #     ]
    #     for i, sub_system in enumerate(self.povm.sub_systems):
    #         for k_i, l_i in np.ndindex(gauge_matrices[i].shape):
    #             gauge_matrices[i][k_i, l_i] = np.real(
    #                 np.trace(self.dual[sub_system][k_i] @ self.povm[sub_system][l_i])  # type: ignore
    #             )

    #     def F(gauge_matrices, outcome_k: tuple[int, ...], outcome_l: tuple[int, ...]):
    #         F_kl = float(outcome_k == outcome_l)
    #         trace_kl = 1.0
    #         for gauge_matrix_i, k_i, l_i in zip(gauge_matrices, outcome_k, outcome_l):
    #             trace_kl *= gauge_matrix_i[k_i, l_i]
    #         F_kl -= trace_kl
    #         return F_kl
    #     outcomes_all = np.empty(shape=len(outcomes), dtype=object)
    #     for i, outcome in enumerate(outcomes):
    #         outcomes_all[i] = outcome
    #     outcomes_array = outcomes_all

    #     counts = self.counts[loc]
    #     weights = self.get_decomposition_weights(obs, set(counts.keys()))

    #     n = len(outcomes_array)

    #     b = np.zeros(n)
    #     hess_matrix = np.zeros((n, n))

    #     for outcome_k in counts:
    #         F_k = np.array(
    #             [F(gauge_matrices, outcome_k, outcome_i) for outcome_i in outcomes_array]
    #         )
    #         for i, F_ki in enumerate(F_k):
    #             b[i] += 2 * counts[outcome_k] * weights[outcome_k] * F_ki
    #             for j, F_kj in enumerate(F_k):
    #                 hess_matrix[i, j] += 2 * counts[outcome_k] * F_ki * F_kj

    #     x = solve(hess_matrix, b)

    #     for outcome, x_i in zip(outcomes_array, x):
    #         self.gammas[outcome] = self.gammas.get(outcome, 0.0) + x_i

    #     return {outcome: x_i for outcome, x_i in zip(outcomes_array, x)}
