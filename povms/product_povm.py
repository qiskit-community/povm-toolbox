"""TODO."""

from __future__ import annotations

import numpy as np

from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Operator

from .base_povm import BasePOVM
from .single_qubit_povm import SingleQubitPOVM
from .utilities import get_p_from_paulis


class ProductPOVM(BasePOVM):
    """Class to represent a set of product POVM operators."""

    def __init__(self, povm_list: list[SingleQubitPOVM]):
        """Initialize from explicit list of POVMs."""
        self._dimension = 1
        self._n_outcomes = 1
        self._n_operators = 0
        for povm in povm_list:
            self._dimension *= povm.dimension
            self._n_outcomes *= povm.n_outcomes
            self._n_operators += povm.n_outcomes

        self._povm_list = povm_list

    @property
    def dimension(self) -> int:
        """Give the dimension of the Hilbert space on which the effects act."""
        self._dimension = 1
        for povm in self._povm_list:
            self._dimension *= povm.dimension
        return self._dimension

    @property
    def n_outcomes(self) -> int:
        """Give the number of outcomes of the POVM."""
        self._n_outcomes = 1
        for povm in self._povm_list:
            self._n_outcomes *= povm.n_outcomes
        return self._n_outcomes

    @property
    def n_operators(self) -> int:
        """Give the number of single-qubit operators forming the POVM."""
        self._n_operators = 0
        for povm in self._povm_list:
            self._n_operators += povm.n_outcomes
        return self._n_operators

    def _check_validity(self) -> bool:
        """Check if POVM axioms are fulfilled.

        Returns:
            TODO.
        """
        for povm in self._povm_list:
            if not povm._check_validity():
                return False
        return True
    
    def _clean_povm(self) -> bool:
        """Merge effects thats are proportionnal to each other and reorder effects in a standard way.

        Returns:
            TODO.
        """
        self._n_outcomes = 1
        self._n_operators = 0
        for povm in self._povm_list:
            povm._clean_povm()
            self._n_outcomes *= povm.n_outcomes
            self._n_operators += povm.n_outcomes
        return self._check_validity()

    def __getitem__(self, index : slice | tuple[slice,slice]) ->  Operator | list[Operator]:
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
                raise IndexError(f'too many indices for array: 2 were expected, but {len(index)} were indexed') from exc
            if isinstance(idx_povm, int):
                return self._povm_list[idx_povm][idx_outcome]
            else :
                return [povm[idx_outcome] for povm in self._povm_list[idx_povm]]
        else:
            return self[index,:]
        
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
        return np.empty((self.n_outcomes))

#    def plot_bloch_sphere(self, dual=False, colors=None):
#        list_fig = []
#        for sqpovm in self.list_povm:
#            list_fig.append(sqpovm.plot_bloch_sphere(dual, colors))
#        return list_fig
