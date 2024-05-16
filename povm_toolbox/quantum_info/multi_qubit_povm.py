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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp, Statevector

from .base_povm import BasePOVM
from .multi_qubit_dual import MultiQubitDUAL
from .multi_qubit_frame import MultiQubitFrame


class MultiQubitPOVM(MultiQubitFrame, BasePOVM):
    """Class that collects all information that any MultiQubit POVM should specify.

    This is a representation of a positive operator-valued measure (POVM). The effects are
    specified as a list of :class:`~qiskit.quantum_info.Operator`.
    """

    default_dual_class = MultiQubitDUAL

    def _check_validity(self) -> None:
        r"""Check if POVM axioms are fulfilled.

        Raises:
            ValueError: if any of the POVM operators is not hermitian.
            ValueError: if any of the POVM operators has a negative eigenvalue.
            ValueError: if all POVM operators do not sum to the identity.
        """
        summed_op: np.ndarray = np.zeros((self.dimension, self.dimension), dtype=complex)

        for k, op in enumerate(self.operators):
            if not np.allclose(op, op.adjoint(), atol=1e-5):
                raise ValueError(f"POVM operator {k} is not hermitian.")

            for eigval in np.linalg.eigvalsh(op.data):
                if eigval.real < -1e-6 or np.abs(eigval.imag) > 1e-5:
                    raise ValueError(f"Negative eigenvalue {eigval} in POVM operator {k}.")

            summed_op += op.data

        if not np.allclose(summed_op, np.identity(self.dimension, dtype=complex), atol=1e-5):
            raise ValueError(f"POVM operators not summing up to the identity : \n{summed_op}")

    def get_prob(
        self,
        rho: SparsePauliOp | DensityMatrix | Statevector,
        outcome_idx: int | set[int] | None = None,
    ) -> float | dict[int, float] | np.ndarray:
        r"""Return the outcome probabilities given a state ``rho``.

        Each outcome :math:`k` is associated with an effect :math:`M_k` of the POVM. The probability of obtaining
        the outcome :math:`k` when measuring a state ``rho`` is given by :math:`p_k = Tr[M_k \rho]`.

        Args:
            rho: the input state over which to compute the outcome probabilities.
            outcome_idx: label(s) indicating which outcome probabilities are queried.

        Returns:
            An array of probabilities. The length of the array is given by the number of outcomes of the POVM.
        """
        if not isinstance(rho, SparsePauliOp):
            rho = Operator(rho)
        return self.analysis(rho, outcome_idx)

    def draw_bloch(
        self,
        title: str = "",
        fig: Figure | None = None,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        font_size: float | None = None,
    ) -> Figure:
        """TODO.

        Args:
            title: A string that represents the plot title.
            fig: User supplied Matplotlib Figure instance for plotting Bloch sphere.
            ax: User supplied Matplotlib axes to render the bloch sphere.
            figsize: Figure size in inches. Has no effect if passing ``ax``.
            font_size: Size of font used for Bloch sphere labels.
        """
        raise NotImplementedError
