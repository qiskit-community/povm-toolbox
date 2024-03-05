"""TODO."""

import numpy as np
from typing import List

from numba import jit

from qiskit.quantum_info import SparsePauliOp

from .single_qubit_povm import SingleQubitPOVM



@jit(nopython=True)
def jit_decompose_operator(
    op_labels: np.ndarray,
    op_coeffs: np.ndarray,
    pauli_decomp: np.ndarray,
    omega_init: np.ndarray,
) -> np.ndarray:
    r"""Decompose an operator in Pauli representation into the linear combination of a basis frame.
    
    Note: could be dual or actual povm operators.

    Args:
        op_labels: np.array that contains the labels of the Pauli strings "IXYI" converted to integers
            with the following conversion {"I": 0, "X": 1, "Y": 2, "Z": 3}, e.g.[[0, 1, 2, 1], ...]
        op_coeffs: np.array of the coefficients of the individual pauli strings
        pauli_decomp: np.array of the Pauli decomposition of the duals of the single-qubit povms
            (should be shape(N_qubits, n_outcomes, 4))
        omega_init: initial value of omega. should be zeros. This is an argument just to get the
            np.zeros out of the numba function. should be shape(n_outcomes, ..., n_outcomes).

    Returns:
        coefficients omega_m as a sparse array of shape(n_outcomes, ..., n_outcomes) (N_qubits times)
    """
    n_qubits = pauli_decomp.shape[0]

    for m, _ in np.ndenumerate(omega_init):
        for j, label in enumerate(op_labels):
            # flip_label = np.flip(label)
            summand = op_coeffs[j]
            for i in range(n_qubits):
                summand *= pauli_decomp[i, m[i], label[i]] * 2  # factor 2 from Tr(P^2) = 2
            omega_init[m] += summand
    return omega_init


def get_p_from_paulis(rho: SparsePauliOp, povm: List[SingleQubitPOVM]) -> np.ndarray:
    r"""Get the measurement probabilities 'p' of a state over a given POVM."""
    n_qubits = rho.num_qubits

    n_outcomes = povm[0].n_outcomes

    assert (
        n_qubits == len(povm)
    ), f"size of the operator {n_qubits} does not match the size of the povm {len(povm)}."

    conversion = {"I": 0, "X": 1, "Y": 2, "Z": 3}

    # convert all parameters into the form required to call the numba decorated function
    p_init = np.zeros([n_outcomes] * n_qubits)
    labels = [string for string, _ in rho.label_iter()]
    op_labels = np.array([[conversion[term] for term in label] for label in labels])

    op_coeffs = np.real_if_close(rho.coeffs)

    duals_pauli_decomp = np.zeros((n_qubits, n_outcomes, 4))
    for i in range(n_qubits):
        for j in range(n_outcomes):
            duals_pauli_decomp[i, j] = np.real_if_close(povm[i].povm_pauli_decomp[j])

    p = jit_decompose_operator(op_labels, op_coeffs, duals_pauli_decomp, p_init)

    return p


# Gram-Schmidt
def gs(X: np.ndarray) -> np.ndarray:
    """Return the orthonormal basis resulting from Gram-Schmidt process of X.

    Args:
        X: TODO.

    Returns:
        TODO.
    """
    Q, _ = np.linalg.qr(X)
    return Q


def n_sphere(param: np.ndarray) -> np.ndarray:
    """Return a unit vector on the n-sphere.

    Args:
        param: TODO.

    Returns:
        TODO.
    """
    n = len(param)
    x = np.ones(n + 1)
    for i in range(n - 1):
        x[i] *= np.cos(np.pi * param[i])
        x[i + 1 :] *= np.sin(np.pi * param[i])
    x[-2] *= np.cos(2 * np.pi * param[-1])
    x[-1] *= np.sin(2 * np.pi * param[-1])

    return x


def povms_union():
    """TODO."""
    return None
