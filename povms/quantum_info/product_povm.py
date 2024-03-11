"""TODO."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit.quantum_info import DensityMatrix, SparsePauliOp

from ..utilities import get_p_from_paulis
from .base_povm import BasePOVM
from .single_qubit_povm import SingleQubitPOVM


class ProductPOVM(BasePOVM):
    """Class to represent a set of product POVM operators."""

    def __init__(self, povms: dict[tuple[int, ...], BasePOVM]):
        """Initialize a ``ProductPOVM`` instance.

        Args:
            povms: a dictionary mapping from a tuple of subsystem indices to a ``BasePOVM`` object.

        Raises:
            ValueError: if any key in ``povms`` is not a tuple consisting of unique integers. In
                other words, every POVM must act on a distinct set of subsystem indices which do not
                overlap with each other.
            ValueError: if any key in ``povms`` re-uses a previously used subsystem index. In other
                words, all POVMs must act on mutually exclusive subsystem indices.
            ValueError: if any key in ``povms`` does not specify the number of subsystem indices,
                which matches the number of systems acted upon by that POVM
                (:meth:`BasePOVM.n_subsystems`).
        """
        subsystem_indices = set()
        self._dimension = 1
        self._n_outcomes = 1
        self._n_operators = 0
        for idx, povm in povms.items():
            idx_set = set(idx)
            if len(idx) != len(idx_set):
                raise ValueError(
                    "The subsystem indices acted upon by any POVM must be mutually exclusive. "
                    f"The index '{idx}' does not fulfill this criterion."
                )
            if any(i in subsystem_indices for i in idx):
                raise ValueError(
                    "The subsystem indices acted upon by all the POVMs must be mutually exclusive. "
                    f"However, one of the indices in '{idx}' was already encountered before."
                )
            if len(idx_set) != povm.n_subsystems:
                raise ValueError(
                    "The number of subsystem indices for a POVM must match the number of subsystems"
                    " which it acts upon. This is not satisfied for the POVM specified to act on "
                    f"subsystems '{idx}' but having support on '{povm.n_subsystems}' subsystems."
                )
            subsystem_indices.update(idx_set)
            self._dimension *= povm.dimension
            self._n_outcomes *= povm.n_outcomes
            self._n_operators += povm.n_outcomes

        self._povms = povms

        self._check_validity()

    @classmethod
    def from_list(cls, povms: Sequence[BasePOVM]) -> ProductPOVM:
        """Construct a ``ProductPOVM`` from a list of ``BasePOVM`` objects.

        This is a convenience method to simplify the construction of a ``ProductPOVM`` for the cases
        in which the POVM objects act on a sequential order of subsystems. In other words, this
        method converts the sequence of POVMs to a dictionary of POVMs in accordance with the input
        to :meth:`ProcutPOVM.__init__` by using the positions along the sequence as subsystem
        indices.

        Below are some examples:
        ```python
        sqp = SingleQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
        product = ProcuctPOVM.from_list([sqp, sqp])
        # is equivalent to
        product = ProductPOVM({(0,): sqp, (1,): sqp})

        mqp = MultiQubitPOVM(
            [
                Operator.from_label("00"),
                Operator.from_label("01"),
                Operator.from_label("10"),
                Operator.from_label("11"),
            ]
        )
        product = ProductPOVM.from_list([mqp, mqp])
        # is equivalent to
        product = ProductPOVM({(0, 1): mqp, (2, 3): mqp})

        product = ProductPOVM.from_list([sqp, sqp, mqp])
        # is equivalent to
        product = ProductPOVM({(0,): sqp, (1,): sqp, (2, 3): mqp})

        product = ProductPOVM.from_list([sqp, mqp, sqp])
        # is equivalent to
        product = ProductPOVM({(0,): sqp, (1, 2): mqp, (3,): sqp})
        ```

        Args:
            povms: a sequence of ``BasePOVM`` objects.

        Returns:
            A new ``ProductPOVM`` instance.
        """
        povm_dict = {}
        idx = 0
        for povm in povms:
            prev_idx = idx
            idx += povm.n_subsystems
            povm_dict[tuple(range(prev_idx, idx))] = povm
        return cls(povm_dict)

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
        return all(povm._check_validity() for povm in self._povms.values())

    @staticmethod
    def clean_povm_operators(prod_povm: ProductPOVM) -> ProductPOVM:
        """Merge effects thats are proportionnal to each other and reorder effects in a standard way.

        Returns:
            TODO.
        """
        n_outcomes = 1
        n_operators = 0
        povm_list = []
        for povm in prod_povm._povms.values():
            povm_list.append(SingleQubitPOVM.clean_povm_operators(povm))  # type: ignore[arg-type]
            n_outcomes *= povm_list[-1].n_outcomes
            n_operators += povm_list[-1].n_outcomes
        return ProductPOVM.from_list(povm_list)

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
        return get_p_from_paulis(
            SparsePauliOp.from_operator(rho),
            list(self._povms.values()),  # type: ignore[arg-type]
        ).ravel()

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
