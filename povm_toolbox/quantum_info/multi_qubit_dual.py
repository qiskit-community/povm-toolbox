# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""MultiQubitDual."""

from __future__ import annotations

import sys

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override  # pragma: no cover

import numpy as np
from qiskit.quantum_info import Operator
from scipy.linalg import orth

from povm_toolbox.utilities import double_ket_to_matrix

from .base import BaseDual, BaseFrame
from .multi_qubit_frame import MultiQubitFrame


class MultiQubitDual(MultiQubitFrame, BaseDual):
    """Class that collects all information that any Dual over multiple qubits should specify.

    This is a representation of a dual frame. Its elements are specified as a list of
    :class:`~qiskit.quantum_info.Operator`.
    """

    @override
    def is_dual_to(self, frame: BaseFrame) -> bool:
        if isinstance(frame, MultiQubitFrame):
            check_matrix = frame @ np.conj(self).T
            if frame.informationally_complete:
                rank = frame.dimension**2
            else:
                # compute the projectors onto the support of the frame:
                orthogonal_projectors = orth(frame)
                rank = orthogonal_projectors.shape[1]
                check_matrix = (
                    np.conj(orthogonal_projectors).T @ check_matrix @ orthogonal_projectors
                )
            # check if ``self`` is dual to ``frame`` on its support:
            return np.allclose(check_matrix, np.eye(rank), atol=1e-6)
        raise NotImplementedError

    @override
    @classmethod
    def build_dual_from_frame(
        cls, frame: BaseFrame, alphas: tuple[float, ...] | None = None
    ) -> MultiQubitDual:
        if isinstance(frame, MultiQubitFrame):
            # Set default values for alphas if none is provided.
            if alphas is None:
                alphas = tuple(np.real(np.trace(frame_op.data)) for frame_op in frame.operators)
            # Check that the number of alpha-parameters match the number of operators
            # forming the ``frame``.
            elif len(alphas) != frame.num_operators:
                raise ValueError(
                    f"The number of alpha-parameters should be equal to the number of"
                    f" operators in the frame ({frame.num_operators}). Here, {len(alphas)}"
                    " parameters were provided."
                )

            # Check if some of the primal frame operators are null. This could happen for instance
            # for locally-biased classical shadows if someone is only interested to measure in the
            # `Z` and `X` bases and therefore sets the `bias` of the `Y` basis to zero. We keep the
            # null operators to preserve the indexing of the operators which might follow a
            # convention, as it is the case for classical shadows. If a primal frame operator is
            # null, its corresponding dual frame operator will also be null.
            frame_array = np.asarray(frame)
            dual_operators_array = np.zeros(frame_array.shape, dtype=complex)
            mask = np.asarray(
                [not np.allclose(np.zeros(len(frame_op)), frame_op) for frame_op in frame_array.T],
                dtype=bool,
            )
            # Temporarily remove the null operators to determine the non-null dual frame operators.
            frame_array = frame_array[:, mask]

            # Set the weighting matrix according to the alpha-parameters.
            diag_trace = np.diag(1.0 / (np.asarray(alphas)[mask]))
            # Compute the weighted frame super-operator.
            superop = frame_array @ diag_trace @ np.conj(frame_array).T

            # Solve the linear system to find the dual operators. If ``frame`` is IC, then
            # ``superop`` will be full rank and invertible. If ``frame`` is not IC, then ``superop``
            # will not be full rank and we use the Moore-Penrose inverse to determine the dual
            # operators on the support of ``frame``.
            dual_operators_array[:, mask], _, _, _ = np.linalg.lstsq(
                superop,
                frame_array @ diag_trace,
            )
            # Convert dual operators from double-ket to operator representation.
            dual_operators = [Operator(double_ket_to_matrix(op)) for op in dual_operators_array.T]

            return cls(dual_operators)

        # We could build a ``MultiQubitDual`` instance (i.e. joint dual frame) that
        # is a dual frame to a ``ProductFrame``, but we have not implemented this yet.
        raise NotImplementedError(f"Not implemented for {type(frame)}")
