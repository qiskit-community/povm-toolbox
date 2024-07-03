# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""MultiQubitPOVM."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import BasePOVM
from .multi_qubit_dual import MultiQubitDual
from .multi_qubit_frame import MultiQubitFrame


class MultiQubitPOVM(MultiQubitFrame, BasePOVM):
    """Class that collects all information that any POVM over multiple qubits should specify.

    This is a representation of a positive operator-valued measure (POVM). The effects are
    specified as a list of :class:`~qiskit.quantum_info.Operator`.

    Below is a simple example showing how you define some 2-qubit POVM:

    >>> from qiskit.quantum_info import Operator
    >>> from povm_toolbox.quantum_info import MultiQubitPOVM
    >>> povm = MultiQubitPOVM(
    ...     [
    ...         Operator.from_label("00"),
    ...         Operator.from_label("01"),
    ...         Operator.from_label("10"),
    ...         Operator.from_label("11"),
    ...     ]
    ... )
    >>> print(povm)
    MultiQubitPOVM(num_qubits=2)<4> at 0x...
    """

    default_dual_class = MultiQubitDual

    def _check_validity(self) -> None:
        """Check if POVM axioms are fulfilled.

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

    def draw_bloch(
        self,
        *,
        title: str = "",
        figure: Figure | None = None,
        axes: Axes | list[Axes] | None = None,
        figsize: tuple[float, float] | None = None,
        font_size: float | None = None,
        colorbar: bool = False,
    ) -> Figure:
        """Plot the Bloch vector of each effect of the POVM.

        .. warning::
           This method is not actually implemented for a generic multi-qubit POVM. However, it is
           available for single-qubit POVMs (see :meth:`.SingleQubitPOVM.draw_bloch`) as well as
           products of such single-qubit POVMs (see :meth:`.ProductPOVM.draw_bloch`).

        Args:
            title: A string that represents the plot title.
            figure: User supplied Matplotlib Figure instance for plotting Bloch sphere.
            axes: User supplied Matplotlib axes to render the bloch sphere.
            figsize: Figure size in inches. Has no effect if passing ``ax``.
            font_size: Size of font used for Bloch sphere labels.
            colorbar: If ``True``, normalize the vectors on the Bloch sphere and
                add a colormap to keep track of the norm of the vectors. It can
                help to visualize the vector if they have a small norm.

        Returns:
            The resulting figure.
        """
        raise NotImplementedError
