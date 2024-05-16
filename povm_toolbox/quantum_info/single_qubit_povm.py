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
                "Dimension of Single Qubit POVM operator space should be 2,"
                f" not {self.dimension}."
            )
        super()._check_validity()

    def get_bloch_vectors(self) -> np.ndarray:
        """TODO."""
        r = np.empty((self.n_outcomes, 3))
        for i, pauli_op in enumerate(self.pauli_operators):
            # Check that the povm effect is rank-1:
            if np.linalg.matrix_rank(self.operators[i]) > 1:
                raise ValueError(
                    "Bloch vector is only well-defined for single-qubit rank-1"
                    f" POVMs. However, the effect number {i} of this POVM has"
                    f" rank {np.linalg.matrix_rank(self.operators[i])}."
                )
            r[i, 0] = 2 * np.real_if_close(pauli_op.get("X", 0))
            r[i, 1] = 2 * np.real_if_close(pauli_op.get("Y", 0))
            r[i, 2] = 2 * np.real_if_close(pauli_op.get("Z", 0))
        return r

    def draw_bloch(
        self,
        title: str = "",
        fig: Figure | None = None,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        font_size: float | None = None,
        colorbar: bool = False,
    ) -> Figure:
        """TODO.

        Args:
            title: A string that represents the plot title.
            fig: User supplied Matplotlib Figure instance for plotting Bloch sphere.
            ax: User supplied Matplotlib axes to render the bloch sphere.
            figsize: Figure size in inches. Has no effect if passing ``ax``.
            font_size: Size of font used for Bloch sphere labels.
            colorbar: If ``True``, normalize the vectors on the Bloch sphere and
                add a colormap to keep track of the norm of the vectors. It can
                help to visualize the vector if they have a small norm.
        """
        from qiskit.visualization.bloch import Bloch
        from qiskit.visualization.utils import matplotlib_close_if_inline

        if figsize is None:
            figsize = (5, 4) if colorbar else (5, 5)

        # Initialize Bloch sphere
        B = Bloch(fig=fig, axes=ax, font_size=font_size)

        # Compute Bloch vector
        vectors = self.get_bloch_vectors()

        if colorbar:
            # Keep track of vector norms through colorbar
            import matplotlib as mpl

            cmap = mpl.colormaps["viridis"]
            B.vector_color = [cmap(np.linalg.norm(vec)) for vec in vectors]
            # Normalize
            for i in range(len(vectors)):
                vectors[i] /= np.linalg.norm(vectors[i])

        B.add_vectors(vectors)
        B.render(title=title)

        if fig is None:
            fig = B.fig
            ax = B.axes
            fig.set_size_inches(figsize[0], figsize[1])
            matplotlib_close_if_inline(fig)

        if colorbar:
            fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ax=ax, label="weight")

        return fig
