# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""dual_from_empirical_frequencies."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import numpy as np
from qiskit.quantum_info import DensityMatrix

from povm_toolbox.post_processor.povm_post_processor import POVMPostProcessor
from povm_toolbox.quantum_info import ProductDual, ProductPOVM
from povm_toolbox.quantum_info.base import BaseDual


class SparseArray:
    """_summary_."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """_summary_.

        Args:
            shape: _description_
        """
        self._array: dict[tuple[int, ...], float] = dict()
        self._shape = shape

    def __getitem__(self, key: tuple[int, ...]) -> float:
        """_summary_.

        Args:
            key: _description_

        Returns:
            _description_
        """
        return self._array.get(key, 0.0)

    def __setitem__(self, key: tuple[int, ...], value: float) -> None:
        """_summary_.

        Args:
            key: _description_
            value: _description_

        Raises:
            KeyError: _description_
            KeyError: _description_
        """
        if len(key) != len(self._shape):
            raise KeyError(
                f"Index length ({len(key)}) does not match the dimension ({len(self._shape)}) of the array"
            )
        for i, n_max in zip(key, self._shape):
            if i < 0 or i >= n_max:
                raise KeyError(f"Index {key} has some elements out of range.")
        self._array[key] = value

    def __repr__(self) -> str:
        """_summary_.

        Returns:
            _description_
        """
        return self._array.__repr__()


class MutualInformationOptimizer:
    """_summary_."""

    def __init__(
        self,
        outcome_counts: dict[tuple[int, ...], int],
        outcome_dims: np.ndarray,
        subsystem_sizes: list[int],
    ) -> None:
        """_summary_.

        Args:
            outcome_counts: _description_
            outcome_dims: _description_
            subsystem_sizes: _description_
        """
        self.counts = outcome_counts
        self._outcome_dims = outcome_dims
        self._subsystem_sizes = subsystem_sizes

    def mutual_info(self, subset: set) -> float:
        """_summary_.

        Args:
            subset: _description_

        Returns:
            _description_
        """
        shots = sum(self.counts.values())

        # Ensure subset is a set for fast lookup
        subset = set(subset)
        subset_complement = set(range(len(self._outcome_dims))) - subset

        # Precompute included and excluded dimensions
        included_dims: list[int] = []
        excluded_dims: list[int] = []
        for idx, dim in enumerate(self._outcome_dims):
            (included_dims if idx in subset else excluded_dims).append(dim)

        # Initialize marginals
        marginals = [SparseArray(tuple(included_dims)), SparseArray(tuple(excluded_dims))]

        # Precompute marginals and cache outcome arrays
        outcome_arrays = {}
        for outcome, count in self.counts.items():
            outcome_arr = np.asarray(outcome)
            outcome_arrays[outcome] = outcome_arr

            included_outcome = tuple(outcome_arr[idx] for idx in subset)
            excluded_outcome = tuple(outcome_arr[idx] for idx in subset_complement)

            marginals[0][included_outcome] += count / shots
            marginals[1][excluded_outcome] += count / shots

        # Compute mutual information
        mut_info = 0.0
        for outcome, count in self.counts.items():
            outcome_arr = outcome_arrays[outcome]
            included_outcome = tuple(outcome_arr[idx] for idx in subset)
            excluded_outcome = tuple(outcome_arr[idx] for idx in subset_complement)

            joint_prob = count / shots
            marginal_prob = marginals[0][included_outcome] * marginals[1][excluded_outcome]
            mut_info += joint_prob * np.log(joint_prob / marginal_prob)

        return mut_info

    def total_mi(self, partition: list[set]) -> float:
        """_summary_.

        Args:
            partition: _description_

        Returns:
            _description_
        """
        total_mi = 0.0
        for subset in partition:
            total_mi += self.mutual_info(subset)
        return total_mi

    def subsystem_size(self, subset: set) -> int:
        """_summary_.

        Args:
            subset: _description_

        Returns:
            _description_
        """
        k = 0
        for i in subset:
            k += self._subsystem_sizes[i]
        return k

    def greedy_combine(self, partition: list[set], k_max: int) -> list[set] | None:
        """_summary_.

        Args:
            partition: _description_
            k_max: _description_

        Returns:
            _description_
        """
        argmin = None
        min_value = self.total_mi(partition)
        for i in range(len(partition)):
            for j in range(i + 1, len(partition)):
                if self.subsystem_size(partition[i] | partition[j]) <= k_max:
                    test_partition = deepcopy(partition)
                    test_partition.remove(partition[i])
                    test_partition.remove(partition[j])
                    test_partition.append(partition[i] | partition[j])
                    mut_info = self.total_mi(test_partition)
                    if mut_info < min_value:
                        min_value = mut_info
                        argmin = test_partition
        return argmin

    def greedy_search(self, k_max: int) -> list[set]:
        """_summary_.

        Args:
            k_max: _description_

        Returns:
            _description_
        """
        partition = [{i} for i in range(len(self._outcome_dims))]
        argmin: list[set] | None = partition
        while argmin is not None:
            argmin = self.greedy_combine(partition, k_max)
            partition = argmin if argmin is not None else partition
        return partition


def dual_from_k_local_empirical_frequencies(
    povm_post_processor: POVMPostProcessor,
    *,
    loc: int | tuple[int, ...] | None = None,
    bias: int | None = None,
    k_max: int = 1,
) -> BaseDual:
    """TODO.

    (Return the k-local Dual frame of ``povm`` based on the frequencies of the sampled outcomes.)
    Given outcomes sampled from a :class:`.ProductPOVM`, each local Dual frame is parametrized with
    the alpha-parameters set as the marginal outcome frequencies. For stability, the (local)
    empirical frequencies can be biased towards the (marginal) outcome probabilities of an
    ``ansatz`` state.

    Args:
        povm_post_processor: the :class:`.POVMPostProcessor` object from which to extract the
            :attr:`.POVMPostProcessor.povm` and the empirical frequencies to build the Dual frame.
        loc: index of the results to use. This is relevant if multiple sets of parameter values were
            supplied to the sampler in the same Pub. If ``None``, it is assumed that the supplied
            circuit was not parametrized or that a unique set of parameter values was supplied. In
            this case, ``loc`` is trivially set to 0.
        bias: TODO.
        k_max: TODO.

    Raises:
        NotImplementedError: if :attr:`.POVMPostProcessor.povm` is not a :class:`.ProductPOVM`
            instance.
        ValueError: if ``loc`` is ``None`` and :attr:`.POVMPostProcessor.counts` stores more than a
            single counter (i.e., multiple sets of parameter values were supplied to the sampler in
            a single Pub).

    Returns:
        TODO (The Dual frame based on the empirical outcome frequencies from the post-processed result.)
    """
    povm = povm_post_processor.povm
    if not isinstance(povm, ProductPOVM):
        raise NotImplementedError("This method is only implemented for `ProductPOVM` objects.")

    if loc is None:
        if povm_post_processor.counts.shape == (1,):
            loc = (0,)
        else:
            raise ValueError(
                "`loc` has to be specified if the POVM post-processor stores"
                " more than one counter (i.e., if multiple sets of parameter"
                " values were supplied to the sampler in a single pub). The"
                f" array of counters is of shape {povm_post_processor.counts.shape}."
            )
    counts = povm_post_processor.counts[loc]

    povm_shape = np.asarray(povm.shape)
    povm_dims = [list(povm._frames.values())[i].num_subsystems for i in range(len(povm._frames))]
    optimizer = MutualInformationOptimizer(counts, povm_shape, subsystem_sizes=povm_dims)
    partition = optimizer.greedy_search(k_max)
    partition_as_list = [list(subset) for subset in partition]
    povm_grouped = povm.group(partition_as_list)

    marginals = [
        np.zeros(np.prod(povm_shape[np.asarray(sub_system)])) for sub_system in partition_as_list
    ]

    # Computing marginals
    shots = sum(counts.values())
    for outcome, count in counts.items():
        outcome_arr = np.asarray(outcome)
        for i, subset in enumerate(partition_as_list):
            # mask = np.asarray([idx in subset for idx in range(len(povm_shape))], dtype=np.bool)
            # sub_outcome = np.ravel_multi_index(outcome_arr[mask], povm_shape[mask])
            outcome_and_shape = [
                (outcome_arr[idx], povm_shape[idx])
                for idx in range(len(povm_shape))
                if idx in subset
            ]
            outcome_subarr, povm_subshape = list(zip(*outcome_and_shape))
            sub_outcome = np.ravel_multi_index(outcome_subarr, povm_subshape)
            marginals[i][sub_outcome] += count / shots

    alphas = []
    # Computing alphas for each subsystem
    for i, sub_system in enumerate(povm_grouped.sub_systems):
        sub_povm = povm_grouped[sub_system]
        dim = sub_povm.dimension
        ansatz_state = DensityMatrix(np.eye(dim) / dim)
        sub_bias = bias or sub_povm.num_outcomes

        sub_alphas = shots * marginals[i] + sub_bias * cast(
            np.ndarray, sub_povm.get_prob(ansatz_state)
        )

        alphas.append(tuple(sub_alphas / (shots + sub_bias)))

    # Building ProductDual from frequencies
    return ProductDual.build_dual_from_frame(povm_grouped, alphas=tuple(alphas))
