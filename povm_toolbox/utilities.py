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
"""

from __future__ import annotations

import numpy as np


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
    return Q


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

    return unit_vector
