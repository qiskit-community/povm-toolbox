"""TODO."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp

from ..utilities import get_p_from_paulis
from .base_povm import BasePOVM
from .single_qubit_povm import SingleQubitPOVM


class ProductPOVM(BasePOVM):
    """Class to represent a set of product POVM operators."""

    def __init__(self, povm_list: list[SingleQubitPOVM]):
        """Initialize from explicit list of POVMs."""
        # TODO: change list[SingleQubitPOVM] to list[MultiQubitPOVM]
        self._dimension = 1
        self._n_outcomes = 1
        self._n_operators = 0
        for povm in povm_list:
            self._dimension *= povm.dimension
            self._n_outcomes *= povm.n_outcomes
            self._n_operators += povm.n_outcomes

        self._povm_list = povm_list

        self._check_validity()

    @property
    def dimension(self) -> int:
        """Give the dimension of the Hilbert space on which the effects act."""
        return self._dimension

    @property
    def n_outcomes(self) -> int:
        """Give the number of outcomes of the POVM."""
        # TODO: check behaviour when a MultiQubitPOVM in the list is cleaned
        return self._n_outcomes

    @property
    def n_operators(self) -> int:
        """Give the number of single-qubit operators forming the POVM."""
        return self._n_operators

    def _check_validity(self) -> bool:
        """Check if POVM axioms are fulfilled.

        Returns:
            TODO.
        """
        return all(povm._check_validity() for povm in self._povm_list)

    @staticmethod
    def clean_povm_operators(prod_povm: ProductPOVM) -> ProductPOVM:
        """Merge effects thats are proportionnal to each other and reorder effects in a standard way.

        Returns:
            TODO.
        """
        n_outcomes = 1
        n_operators = 0
        povm_list = []
        for povm in prod_povm._povm_list:
            povm_list.append(SingleQubitPOVM.clean_povm_operators(povm))
            n_outcomes *= povm_list[-1].n_outcomes
            n_operators += povm_list[-1].n_outcomes
        return ProductPOVM(povm_list)

    def __getitem__(self, index: slice | tuple[slice, slice]) -> Operator | list[Operator]:
        """Return a povm operator or a list of povm operators.

        Args:
            index: TODO.

        Returns:
            TODO.
        """
        if isinstance(index, tuple):
            try:
                idx_povm, idx_outcome = index
            except ValueError as exc:
                raise IndexError(
                    f"too many indices for array: 2 were expected, but {len(index)} were indexed"
                ) from exc
            if isinstance(idx_povm, int):
                return self._povm_list[idx_povm][idx_outcome]
            else:
                return [povm[idx_outcome] for povm in self._povm_list[idx_povm]]
        else:
            return self[index, :]

    def __len__(self) -> int:
        """Return the number of outcomes of the POVM.

        Returns:
            TODO.
        """
        return self.n_outcomes

    def get_prob(self, rho: DensityMatrix) -> np.ndarray:
        """Return the outcome probabilities given a state rho.

        Args:
            rho: TODO.

        Returns:
            TODO.
        """
        return get_p_from_paulis(SparsePauliOp.from_operator(rho), self._povm_list).ravel()

    def get_omegas(self, obs: np.ndarray):
        """Return the decomposition weights of obserservable `obs` into the POVM effects.

        Args:
            obs: TODO.

        Returns:
            TODO.
        """
        # TODO
        return np.empty(self.n_outcomes)


#    def plot_bloch_sphere(self, dual=False, colors=None):
#        list_fig = []
#        for sqpovm in self.list_povm:
#            list_fig.append(sqpovm.plot_bloch_sphere(dual, colors))
#        return list_fig
