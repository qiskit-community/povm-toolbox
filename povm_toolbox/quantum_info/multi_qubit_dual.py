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
from qiskit.quantum_info import Operator

from povm_toolbox.utilities import double_ket_to_matrix

from .base_dual import BaseDUAL
from .base_frame import BaseFrame
from .multi_qubit_frame import MultiQubitFrame


class MultiQubitDUAL(MultiQubitFrame, BaseDUAL):
    """Class that collects all information that any MultiQubit POVM should specify.

    This is a representation of a positive operator-valued measure (POVM). The effects are
    specified as a list of :class:`~qiskit.quantum_info.Operator`.
    """

    def get_omegas(
        self, obs: Operator, outcome_idx: int | set[int] | None = None
    ) -> float | dict[int, float] | np.ndarray:
        r"""Return the decomposition weights of observable ``obs`` into the POVM effects.

        Given an observable :math:`O` which is in the span of the POVM, one can write the
        observable :math:`O` as the weighted sum of the POVM effects, :math:`O = \sum_k w_k M_k`
        for real weights :math:`w_k`. There might be infinitely many valid sets of weight.
        This method returns a possible set of weights.

        Args:
            obs: the observable to be decomposed into the POVM effects.

        Returns:
            An array of decomposition weights.
        """
        return self.analysis(obs, outcome_idx)

    def is_dual_to(self, frame=BaseFrame) -> bool:
        """Check if `self` is a dual to another frame."""
        if isinstance(frame, MultiQubitFrame):
            return np.allclose(frame @ np.conj(self).T, np.eye(self.dimension**2), atol=1e-6)
        raise NotImplementedError

    @classmethod
    def build_dual_from_frame(cls, frame=BaseFrame) -> MultiQubitDUAL:
        """Construct a dual frame to another frame."""
        if isinstance(frame, MultiQubitFrame):
            diag_trace = np.diag([1.0 / np.trace(frame_op.data) for frame_op in frame.operators])
            superop = frame @ diag_trace @ np.conj(frame).T

            dual_operators_array = np.linalg.solve(
                superop,
                frame @ diag_trace,
            )
            dual_operators = [Operator(double_ket_to_matrix(op)) for op in dual_operators_array.T]

            # TODO : move this test to unittest in the future and just return cls(dual_operators)
            dual_frame = cls(dual_operators)
            if not dual_frame.is_dual_to(frame):
                raise ValueError
            return dual_frame

        raise NotImplementedError
