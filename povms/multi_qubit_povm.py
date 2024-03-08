"""TODO."""

from __future__ import annotations

import copy
from typing import TypeVar

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator

from .base_povm import BasePOVM

# Create a generic variable that can be 'MultiQubitPOVM', or any subclass.
T = TypeVar("T", bound="MultiQubitPOVM")


class MultiQubitPOVM(BasePOVM):
    """Class that collects all information that any MultiQubit POVM should specifiy."""

    def __init__(self, povm_ops: list[Operator]) -> None:
        """Initialize from explicit POVM operators.

        Args:
            povm_operators: list that contains the explicit POVM operators.

        Raises:
            ValueError: TODO.
        """
        self._n_outcomes: int = len(povm_ops)
        self._dimension: int = povm_ops[0].dim[0]
        for op in povm_ops:
            if not (self._dimension == op.dim[0] and self._dimension == op.dim[1]):
                raise ValueError(
                    f"POVM operators need to be square ({op.dim[0]},{op.dim[1]}) and all of the same dimension."
                )
        self.povm_operators: list[Operator] = povm_ops
        self.array_ops = None

        self.dual_operators = None
        self.frame_superop = None
        self.informationlly_complete = None
        # TODO: should all of the public attributes above become part of the BasePOVM interface?

        self._check_validity()

    @property
    def dimension(self) -> int:
        """Give the dimension of the Hilbert space on which the effects act."""
        return self._dimension

    @property
    def n_outcomes(self) -> int:
        """Give the number of outcomes of the POVM."""
        return self._n_outcomes

    def _check_validity(self) -> bool:
        """Check if POVM axioms are fulfilled.

        Returns:
            TODO.

        Raises:
            ValueError: TODO.
        """
        summed_op: np.ndarray = np.zeros((self.dimension, self.dimension), dtype=complex)

        for k, op in enumerate(self.povm_operators):
            if not np.allclose(op, op.adjoint(), atol=1e-5):
                raise ValueError(f"POVM operator {k} is not hermitian.")

            for eigval in np.linalg.eigvalsh(op.data):
                if eigval.real < -1e-6 or np.abs(eigval.imag) > 1e-5:
                    raise ValueError(f"Negative eigenvalue {eigval} in POVM operator {k}.")

            summed_op += op.data

        if not np.allclose(summed_op, np.identity(self.dimension, dtype=complex), atol=1e-5):
            raise ValueError(f"POVM operators not summing up to the identity : \n{summed_op}")

        return True

    @classmethod
    def clean_povm_operators(cls: type[T], povm: T) -> T:
        """Merge effects thats are proportionnal to each other and reorder effects in a standard way.

        Returns:
            TODO.
        """
        povm_ops = copy.deepcopy(povm.povm_operators)
        k1 = 0

        while k1 < len(povm_ops):
            k2 = k1 + 1
            while k2 < len(povm_ops):
                if np.allclose(
                    povm_ops[k1] / np.trace(povm_ops[k1]),
                    povm_ops[k2] / np.trace(povm_ops[k2]),
                ):
                    povm_ops[k1] = Operator(povm_ops[k1] + povm_ops[k2])
                    povm_ops.pop(k2)

                    k2 -= 1
                k2 += 1
            k1 += 1

        sorting_values = np.array(
            [
                (
                    np.real(np.trace(op.data)),
                    np.max(np.linalg.eigvalsh(op.data)),
                    np.real(op.data[0, 0]),
                )
                for op in povm_ops
            ],
            dtype=[("tr", "float"), ("ev", "float"), ("m00", "float")],
        )
        idx_sort = np.argsort(sorting_values, order=("tr", "ev", "m00"))[::-1]
        return cls([povm_ops[idx] for idx in idx_sort])

    def __getitem__(self, index: slice) -> Operator | list[Operator]:
        """Return a povm operator or a list of povm operators."""
        return self.povm_operators[index]

    def __len__(self) -> int:
        """TODO."""
        return self.n_outcomes

    def get_prob(self, rho: DensityMatrix) -> np.ndarray:
        """TODO.

        Args:
            rho: TODO.

        Returns:
            TODO.
        """
        return np.array(
            [np.real(np.trace(rho.data @ povm_op.data)) for povm_op in self.povm_operators]
        )

    def get_omegas(self, obs: np.ndarray):
        """Return the decomposition weights of obserservable `obs` into the POVM effects.

        Args:
            obs: TODO.

        Returns:
            TODO.
        """
        # TODO
        return np.empty(self.n_outcomes)

    @classmethod
    def from_vectors(cls, povm_vectors: np.ndarray):
        """Initialize a POVM from the bloch vectors |psi> (not normalized!) such that Pi = |psi><psi|.

        Args:
            povm_vectors: TODO.

        Returns:
            TODO.
        """
        povm_operators = []
        for vec in povm_vectors:
            povm_operators.append(Operator(np.outer(vec, vec.conj())))
        return cls(povm_operators)

    # @classmethod
    # def from_dilation_unitary(cls, U, dim):
    #     """Initialize a POVM from dilation unitary"""
    #     return cls.from_vectors(U[:,0:dim].conj())
    #
    # @classmethod
    # def from_param(cls, param_raw: np.ndarray, dim: int):
    #     """Initialize a POVM from the list of parameters"""
    #
    #     assert (
    #         (len(param_raw)+dim**2)%(2*dim-1) == 0
    #     ), f"size of the parameters ({len(param_raw)}) does not match expectation."
    #
    #     n_out = (len(param_raw)+dim**2)//(2*dim-1)
    #
    #     param = []
    #     param.append(param_raw[0:(n_out-1)])
    #     count = n_out-1
    #     for i in range(1,dim):
    #         l = 2*(n_out-i)-1
    #         param.append(param_raw[count:count+l])
    #         count += l
    #
    #     u = np.zeros((n_out,n_out), dtype=complex)
    #
    #     k=0
    #     u[:,k] = n_sphere(param[k])
    #     u_gs = gs(u) #Gram-Schmidt
    #
    #     for k in range(1,dim):
    #         x=n_sphere(param[k])
    #         # construct k'th vector of u
    #         for i in range(len(x)//2):
    #             u[:,k] += (x[2*i] + x[2*i+1]*1j) * u_gs[:,k+i]
    #         u_gs = gs(u)
    #
    #     for i in range(len(param)):
    #         u_gs[:,i] *= np.sign(u[0,i])*np.sign(u_gs[0,i])
    #
    #     return cls.from_dilation_unitary(u_gs, dim)
    #
    #
    # #def __getitem__(self, index:slice) -> np.ndarray:
    # #    """Return a numpy array of shape (n_outcomes, d, d) that includes all povm operators."""
    # #    if isinstance(index, int) :
    # #        return self.povm_operators[index].data
    # #    elif isinstance(index, slice) :
    # #        return np.array([op.data for op in self.povm_operators[index]])
    # #    else:
    # #        raise TypeError("Invalid Argument Type")
    #
    #     def get_ops(self, idx:slice=...) -> np.ndarray:
    #     """Return a numpy array of shape (n_outcomes, d, d) that includes all povm operators."""
    #
    #     if self.array_ops is None:
    #         self.array_ops = np.zeros((self.n_outcomes, self.dimension, self.dimension), dtype=complex)
    #         for k, op in enumerate(self.povm_operators):
    #             self.array_ops[k] = op.data
    #
    #     return self.array_ops[idx]
    #
