"""TODO."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit.quantum_info import DensityMatrix, SparsePauliOp

from .base_povm import BasePOVM
from .multi_qubit_povm import MultiQubitPOVM


class ProductPOVM(BasePOVM):
    r"""Class to represent a set of product POVM operators.

    A product POVM :math:`M` is made of local POVMs :math:`M1, M2, ...` acting
    on respective subsystems. Each global effect can be written as the tensor
    product of local effects,
    :math:`M_{k_1, k_2, ...} = M1_{k_1} \otimes M2_{k2} \otimes ...`.
    """

    def __init__(self, povms: dict[tuple[int, ...], MultiQubitPOVM]):
        """Initialize a ``ProductPOVM`` instance.

        Args:
            povms: a dictionary mapping from a tuple of subsystem indices to a ``MultiQubitPOVM``
                object.

        Raises:
            ValueError: if any key in ``povms`` is not a tuple consisting of unique integers. In
                other words, every POVM must act on a distinct set of subsystem indices which do not
                overlap with each other.
            ValueError: if any key in ``povms`` re-uses a previously used subsystem index. In other
                words, all POVMs must act on mutually exclusive subsystem indices.
            ValueError: if any key in ``povms`` does not specify the number of subsystem indices,
                which matches the number of systems acted upon by that POVM
                (:meth:`MultiQubitPOVM.n_subsystems`).
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
    def from_list(cls, povms: Sequence[MultiQubitPOVM]) -> ProductPOVM:
        """Construct a ``ProductPOVM`` from a list of ``MultiQubitPOVM`` objects.

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
            povms: a sequence of ``MultiQubitPOVM`` objects.

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

    def _check_validity(self) -> None:
        """Check if POVM axioms are fulfilled.

        Raises:
            TODO.
        """
        for povm in self._povms.values():
            povm._check_validity()

    def __len__(self) -> int:
        """Return the number of outcomes of the POVM."""
        return self.n_outcomes

    def get_prob(self, rho: DensityMatrix) -> np.ndarray:
        """Return the outcome probabilities given a state rho.

        Args:
            rho: the input state over which to compute the outcome probabilities.

        Returns:
            An array of probabilities. Its shape is a high-dimensional array with one dimension
            per local POVM stored inside this ``ProductPOVM``. The length of each dimension is
            given by the number of outcomes of the POVM encoded along that axis.

        Raises:
            ValueError: when the provided state ``rho`` does not act on the same number of qubits as
                this ``ProductPOVM``.
        """
        # Convert the provided state to a Pauli operator.
        rho = SparsePauliOp.from_operator(rho)

        # Assert matching operator and POVM sizes.
        if rho.num_qubits != self.n_subsystems:
            raise ValueError(
                f"se of the operator {rho.n_qubits} does not match the size of the povm {len(self)}."
            )

        # Extract the number of outcomes for each local POVM.
        n_outcomes = [povm.n_outcomes for povm in self._povms.values()]

        # Create the output probability array as a high-dimensional matrix. This matrix will have
        # its number of dimensions equal to the number of POVMs stored inside the ProductPOVM. The
        # length of each dimension is given by the number of outcomes of the POVM encoded along it.
        p_init: np.ndarray = np.zeros(n_outcomes, dtype=complex)

        # First, we iterate over all the positions of `p_init`. This corresponds to the different
        # probabilities for the different outcomes whose probability we want to compute.
        #   - `m` is the multi-dimensional index into the high-dimensional `p_init` array.
        for m, _ in np.ndenumerate(p_init):
            # Second, we iterate over our input state, `rho`.
            for label, rho_coeff in rho.label_iter():
                summand = rho_coeff
                # Third, we iterate over the POVMs stored inside the ProductPOVM.
                #   - `j` is the index of the POVM inside the `ProductPOVM`. This encodes the axis
                #     of the high-dimensional array `p_init` along which this local POVM is encoded.
                #   - `idx` are the qubit indices on which this local POVM acts.
                #   - `povm` is the actual local POVM object.
                for j, (idx, povm) in enumerate(self._povms.items()):
                    # Extract the local Pauli term on the qubit indices of this local POVM.
                    sublabel = "".join(label[i] for i in idx)
                    # Try to obtain the coefficient of the local POVM for this local Pauli term.
                    try:
                        coeff = povm.pauli_operators[m[j]][sublabel]
                    except KeyError:
                        # If it does not exist, the current summand becomes 0 because it would be
                        # multiplied by 0.
                        summand = 0.0
                        # In this case we can break the iteration over the remaining local POVMs.
                        break
                    else:
                        # If the label does exist, we multiply the coefficient into our summand.
                        # The factor 2*N_qubit comes from Tr[(P_1...P_N)^2] = 2*N.
                        summand *= coeff * 2 * povm.n_subsystems

                # Once we have finished computing our summand, we add it into `p_init`.
                p_init[m] += summand

        # Return the flattened `p_init` array.
        return np.real_if_close(p_init)

    def get_omegas(self, obs: np.ndarray):
        """Return the decomposition weights of obserservable `obs` into the POVM effects.

        Args:
            obs: TODO.

        Returns:
            TODO.
        """
        # TODO
        return np.empty(self.n_outcomes)
