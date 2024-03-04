"""TODO."""

from typing import List

from qiskit.quantum_info import DensityMatrix, SparsePauliOp

from base_povm import Povm
from utilities import get_p_from_paulis


class ProductPOVM(Povm):
    """Class to represent a set of product POVM operators."""

    def __init__(self, povm_list: List[Povm]):
        """Initialize from explicit list of POVMs."""
        self.dimension = 1
        self.n_outcomes = 1
        self.n_operators = 0
        for povm in povm_list:
            self.dimension *= povm.dimension
            self.n_outcomes *= povm.n_outcomes
            self.n_operators += povm.n_outcomes

        self.povm_list = povm_list

    def __getitem__(self, index : tuple[slice]):
        if isinstance(index, tuple):
            try:
                idx_povm, idx_outcome = index
            except ValueError:
                raise IndexError(f'too many indices for array: 2 were expected, but {len(index)} were indexed')
            if isinstance(idx_povm, int):
                return self.povm_list[idx_povm][idx_outcome]
            else :
                return [povm[idx_outcome] for povm in self.povm_list[idx_povm]]
        else:
            return self[index,:]

    def get_prob(self, rho: DensityMatrix):
        return get_p_from_paulis(SparsePauliOp.from_operator(rho), self.povm_list).ravel()

#    def plot_bloch_sphere(self, dual=False, colors=None):
#        list_fig = []
#        for sqpovm in self.list_povm:
#            list_fig.append(sqpovm.plot_bloch_sphere(dual, colors))
#        return list_fig
