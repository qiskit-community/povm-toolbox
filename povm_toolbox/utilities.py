# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A collection of common utilities.

.. currentmodule:: povm_toolbox.utilities

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   matrix_to_double_ket
   double_ket_to_matrix
   jit_get_omega_samples
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numba import njit, prange


def matrix_to_double_ket(op_matrix: np.ndarray) -> np.ndarray:
    """Return the double-ket representation of an operator.

    Args:
        op_matrix: an operator in matrix representation.

    Returns:
        The double-ket representation of the operator ``op``.
    """
    return op_matrix.ravel(
        order="F"
    )  # order='F' option to stack the columns instead of the (by default) rows


def double_ket_to_matrix(op_ket: np.ndarray) -> np.ndarray:
    """Return the matrix representation of an operator.

    Args:
        op_ket: an operator in the double-ket representation.

    Returns:
        The matrix representation of the operator ``op``.
    """
    dim = int(np.sqrt(len(op_ket)))
    return op_ket.reshape((dim, dim), order="F")


# Gram-Schmidt
def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """Transform ``vectors`` into an orthonormal basis (ONB) through the Gram-Schmidt process.

    Args:
        vectors: set of vectors to transform into an ONB.

    Returns:
        The resulting orthonormal basis.
    """
    Q, _ = np.linalg.qr(vectors)
    return cast(np.ndarray, Q)


# Unit vector on n-sphere
def n_sphere(angles: np.ndarray) -> np.ndarray:
    """Return a unit vector on the :math:`n`-sphere.

    Args:
        angles: set of normalized angles defining the unit vector.

    Returns:
        The resulting unit vector.
    """
    # dimension of the sphere
    n = len(angles)
    # initialize the unit vector
    unit_vector = np.ones(n + 1)
    for i in range(n - 1):
        unit_vector[i] *= np.cos(np.pi * angles[i])
        unit_vector[i + 1 :] *= np.sin(np.pi * angles[i])
    unit_vector[-2] *= np.cos(2 * np.pi * angles[-1])
    unit_vector[-1] *= np.sin(2 * np.pi * angles[-1])

    return cast(np.ndarray, unit_vector)


@njit(parallel=True, fastmath=True)  # pragma: no cover
def jit_get_omega_samples(
    op_labels: np.ndarray,
    op_coeffs: np.ndarray,
    pauli_decomp: np.ndarray,
    povm_samples: np.ndarray,
    omega_samples: np.ndarray,
) -> np.ndarray:
    r"""Decompose an operator in Pauli representation into the linear combination of a basis frame.

    The frame can be dual or actual povm operators.

    Args:
        op_labels: np.array that contains the labels of the Pauli strings "IXYI" converted to integers
            with the following conversion {"I": 0, "X": 1, "Y": 2, "Z": 3}, e.g.[[0, 1, 2, 1], ...]
        op_coeffs: np.array of the coefficients of the individual pauli strings
        pauli_decomp: np.array of the Pauli decomposition of the duals of the single-qubit povms
            (should be shape(N_qubits, n_outcomes, 4))
        povm_samples: np.array of the measured POVM samples, should be shape(n_samples, n_qubits),
            e.g. [[1, 3, 2, 0], [4, 2, 3, 1], ...]
        omega_samples: initial value of omega. should be zeros. This is an argument just to get the
            np.zeros out of the numba function. should be shape(n_samples).

    Returns:
        coefficients omega_m as a sparse array of shape(n_outcomes)
    """
    n_qubits = int(pauli_decomp.shape[0])
    n_samples = int(povm_samples.shape[0])
    n_oplabels = int(op_labels.shape[0])
    for sample_ind in prange(n_samples):
        m = povm_samples[sample_ind]
        samp = 0
        for j in prange(n_oplabels):
            label = op_labels[j]
            summand = op_coeffs[j]
            for i in range(n_qubits):
                summand *= pauli_decomp[i, m[i], label[i]] * 2  # factor 2 from Tr(P^2) = 2
            samp += summand
        omega_samples[sample_ind] = samp
    return omega_samples
