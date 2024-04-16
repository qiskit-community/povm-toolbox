# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TODO."""

from __future__ import annotations

import numpy as np


def matrix_to_double_ket(op_matrix):
    """Return the double-ket represenation of an operator.

    Args:
        op: an operator in matrix representation.

    Returns:
        The double-ket represenattion of the operator ``op``.
    """
    return op_matrix.ravel(
        order="F"
    )  # order='F' option to stack the columns instead of the (by default) rows


def double_ket_to_matrix(op_ket):
    """Return the matrix represenation of an operator.

    Args:
        op: an operator in the double-ket represention.

    Returns:
        The matrix represenattion of the operator ``op``.
    """
    dim = int(np.sqrt(len(op_ket)))
    return op_ket.reshape((dim, dim), order="F")


# Gram-Schmidt
def gs(X: np.ndarray) -> np.ndarray:
    """Return the orthonormal basis resulting from Gram-Schmidt process of X.

    Args:
        X: a basis, i.e., a collection of ``n`` linearly independant ``n``-dimeensional vectors.

    Returns:
        An orthonormal basis.
    """
    Q, _ = np.linalg.qr(X)
    return Q


def n_sphere(param: np.ndarray) -> np.ndarray:
    """Return a unit vector on the `n`-sphere.

    Args:
        param: set of ``n`` parameters to define the unit vector. These correspond to
            normalized angles.

    Returns:
        An ``n+1``-dimensional unit vector on the `n`-sphere.
    """
    n = len(param) + 1
    x = np.ones(n)
    for i in range(n - 2):
        x[i] *= np.cos(np.pi * param[i])
        x[i + 1 :] *= np.sin(np.pi * param[i])
    x[-2] *= np.cos(2 * np.pi * param[-1])
    x[-1] *= np.sin(2 * np.pi * param[-1])

    return x
