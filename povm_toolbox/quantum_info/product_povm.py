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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector
from qiskit.visualization.utils import matplotlib_close_if_inline

from .base_povm import BasePOVM
from .multi_qubit_povm import MultiQubitPOVM
from .product_dual import ProductDUAL
from .product_frame import ProductFrame


class ProductPOVM(ProductFrame[MultiQubitPOVM], BasePOVM):
    r"""Class to represent a set of product POVM operators.

    A product POVM :math:`M` is made of local POVMs :math:`M1, M2, ...` acting
    on respective subsystems. Each global effect can be written as the tensor
    product of local effects,
    :math:`M_{k_1, k_2, ...} = M1_{k_1} \otimes M2_{k2} \otimes ...`.
    """

    default_dual_class = ProductDUAL

    def _check_validity(self) -> None:
        """Check if POVM axioms are fulfilled.

        Raises:
            TODO.
        """
        for povm in self._povms.values():
            if not isinstance(povm, MultiQubitPOVM):
                raise TypeError
            povm._check_validity()

    def get_prob(
        self,
        rho: SparsePauliOp | DensityMatrix | Statevector,
        outcome_idx: tuple[int, ...] | set[tuple[int, ...]] | None = None,
    ) -> float | dict[tuple[int, ...], float] | np.ndarray:
        """Return the outcome probabilities given a state rho.

        Args:
            rho: the input state over which to compute the outcome probabilities.
            outcome_idx: the outcomes for which one queries the probability. Each outcome is labeled
                by a tuple of integers (one index per local POVM). One can query a single outcome or a
                set of outcomes. If ``None``, all outcomes are queried.

        Returns:
            Probabilities of obtaining the outcome(s) specified by ``outcome_idx`` over the state ``rho``.
            If a specific outcome was queried, a ``float`` is returned. If a specific set of outcomes was
            queried, a dictionary mapping outcomes to probabilities is returned. If all outcomes were
            queried, a high-dimensional array with one dimension per local POVM stored inside this
            :class:`.`ProductPOVM` is returned. The length of each dimension is given by the number of outcomes
            of the POVM encoded along that axis.
        """
        if not isinstance(rho, SparsePauliOp):
            rho = SparsePauliOp.from_operator(rho)
        return self.analysis(rho, outcome_idx)

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
        """Plot a Bloch sphere for each single-qubit POVM forming the product POVM.

        Args:
            title: a string that represents the plot title.
            figure: User supplied Matplotlib Figure instance for plotting Bloch sphere.
            axes: User supplied Matplotlib axes to render the bloch sphere.
            figsize: size of each individual Bloch sphere figure, in inches.
            font_size: Font size for the Bloch ball figures.
            colorbar: If ``True``, normalize the vectors on the Bloch sphere and
                add a colormap to keep track of the norm of the vectors. It can
                help to visualize the vector if they have a small norm.
        """
        # Number of subplots (one per qubit)
        num = self.num_subsystems

        # Check that all local POVMs are single-qubit POVMs
        if any([len(idx) > 1 for idx in self.sub_systems]):
            raise NotImplementedError

        # Determine the number of rows and columns for the figure
        n_cols = int(np.sqrt(num) * 4 / 3)
        n_rows = int(np.sqrt(num) * 3 / 4) or 1
        while n_cols * n_rows < num:
            n_cols += 1 if n_cols * n_rows < num else 0
            n_rows += 1 if n_cols * n_rows < num else 0
            while (n_cols - 1) * n_rows >= num:
                n_cols -= 1

        # Set default values
        if figsize is None:
            figsize = (5, 4) if colorbar else (5, 5)
        width, height = figsize
        width *= n_cols
        height *= n_rows
        title_font_size = font_size if font_size is not None else 16

        # Plot figure
        fig = figure if figure is not None else plt.figure(figsize=(width, height))
        for i, idx in enumerate(self.sub_systems):
            ax = (
                axes[i]
                if isinstance(axes, list)
                else fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
            )
            self[idx].draw_bloch(
                title="qubit " + ", ".join(map(str, idx)),
                figure=fig,
                axes=ax,
                figsize=figsize,
                font_size=font_size,
                colorbar=colorbar,
            )
        fig.suptitle(title, fontsize=title_font_size, y=1.0)
        matplotlib_close_if_inline(fig)

        return fig
