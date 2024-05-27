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
from qiskit.quantum_info import Operator, SparsePauliOp

from povm_toolbox.utilities import double_ket_to_matrix

from .base_dual import BaseDUAL
from .base_frame import BaseFrame
from .multi_qubit_frame import MultiQubitFrame


class MultiQubitDUAL(MultiQubitFrame, BaseDUAL):
    """Class that collects all information that any MultiQubit DUAL should specify.

    This is a representation of a dual frame. Its elements are specified as a list
    of :class:`~qiskit.quantum_info.Operator`.
    """

    def get_omegas(
        self, obs: SparsePauliOp | Operator, outcome_idx: int | set[int] | None = None
    ) -> float | dict[int, float] | np.ndarray:
        r"""Return the decomposition weights of the provided observable into the POVM effects to which ``self`` is a dual.

        Here the POVM itself is not explicitly needed, its dual is sufficient. Given
        an observable :math:`O` which is in the span of a given POVM, one can write the
        observable :math:`O` as the weighted sum of the POVM effects, :math:`O = \sum_k w_k M_k`
        for real weights :math:`w_k`. There might be infinitely many valid sets of weight.
        This method returns a possible set of weights.

        Args:
            obs: the observable to be decomposed into the POVM effects.
            outcome_idx: label(s) indicating which decomposition weights are queried.

        Returns:
            An array of decomposition weights.
        """
        return self.analysis(obs, outcome_idx)

    def is_dual_to(self, frame: BaseFrame) -> bool:
        """Check if ``self`` is a dual to another frame."""
        if isinstance(frame, MultiQubitFrame):
            return np.allclose(frame @ np.conj(self).T, np.eye(self.dimension**2), atol=1e-6)
        raise NotImplementedError

    @classmethod
    def build_dual_from_frame(
        cls, frame: BaseFrame, alphas: tuple[float, ...] | None = None
    ) -> MultiQubitDUAL:
        """Construct a dual frame to another frame.

        Args:
            frame: The primal frame from which we will build the dual frame.

        Returns:
            A multi-qubit dual frame to the supplied ``frame``.
        """
        if isinstance(frame, MultiQubitFrame):
            # Set default values for alphas if none is provided.
            if alphas is None:
                alphas = tuple(np.real(np.trace(frame_op.data)) for frame_op in frame.operators)
            # Check that the number of alpha-parameters match the number of operators
            # forming the ``frame``.
            elif len(alphas) != frame.n_operators:
                raise ValueError(
                    f"The number of alpha-parameters should be equal to the number of"
                    f" operators in the frame ({frame.n_operators}). Here, {len(alphas)}"
                    " parameters were provided."
                )

            # Set the weighting matrix according to the alpha-parameters
            diag_trace = np.diag([1.0 / alpha for alpha in alphas])
            # Compute the weighed frame super-operator.
            superop = frame @ diag_trace @ np.conj(frame).T

            # Solve the linear system to find the dual operators.
            dual_operators_array = np.linalg.solve(
                superop,
                frame @ diag_trace,
            )
            # Convert dual operators from double-ket to operator representation.
            dual_operators = [Operator(double_ket_to_matrix(op)) for op in dual_operators_array.T]

            return cls(dual_operators)

        # We could build a ``MultiQubitDUAL`` instance (i.e. joint dual frame) that
        # is a dual frame to a ``ProductFrame``, but we have not implemented this yet.
        raise NotImplementedError(f"Not implemented for {type(frame)}")
