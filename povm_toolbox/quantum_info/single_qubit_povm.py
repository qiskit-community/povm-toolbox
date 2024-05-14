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

from .multi_qubit_povm import MultiQubitPOVM


class SingleQubitPOVM(MultiQubitPOVM):
    """Class to represent a set of IC single-qubit POVM operators."""

    def _check_validity(self) -> None:
        """TODO.

        Raises:
            ValueError: TODO.
        """
        if not self.dimension == 2:
            raise ValueError(
                f"Dimension of Single Qubit POVM operator space should be 2, not {self.dimension}."
            )
        super()._check_validity()

    def get_bloch_vectors(self):
        r = np.empty((self.n_outcomes, 3))
        for i, pauli_op in enumerate(self.pauli_operators):
            r[i, 0] = 2 * np.real_if_close(pauli_op.get("X", 0))
            r[i, 1] = 2 * np.real_if_close(pauli_op.get("Y", 0))
            r[i, 2] = 2 * np.real_if_close(pauli_op.get("Z", 0))
        return r

    def draw_bloch(self, title="", fig=None, ax=None, figsize=None, font_size=None):
        from qiskit.visualization.bloch import Bloch
        from qiskit.visualization.utils import matplotlib_close_if_inline

        if figsize is None:
            figsize = (5, 5)
        B = Bloch(fig=fig, axes=ax, font_size=font_size)
        vectors = self.get_bloch_vectors()
        B.add_vectors(vectors)
        B.render(title=title)
        if fig is None:
            fig = B.fig
            fig.set_size_inches(figsize[0], figsize[1])
            matplotlib_close_if_inline(fig)
            return fig
        return None
