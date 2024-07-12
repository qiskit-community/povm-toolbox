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
def gs(X):
    """TODO."""
    Q, _ = np.linalg.qr(X)
    return Q


# Unit vector on n-sphere
def n_sphere(param: np.ndarray):
    """TODO."""
    n = len(param)
    x = np.ones(n + 1)
    for i in range(n - 1):
        x[i] *= np.cos(np.pi * param[i])
        x[i + 1 :] *= np.sin(np.pi * param[i])
    x[-2] *= np.cos(2 * np.pi * param[-1])
    x[-1] *= np.sin(2 * np.pi * param[-1])

    return x
