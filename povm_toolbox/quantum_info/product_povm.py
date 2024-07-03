# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""ProductPOVM."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qiskit.visualization.utils import matplotlib_close_if_inline

from .base import BasePOVM
from .multi_qubit_povm import MultiQubitPOVM
from .product_dual import ProductDual
from .product_frame import ProductFrame


class ProductPOVM(ProductFrame[MultiQubitPOVM], BasePOVM):
    r"""Class to represent a set of product POVM operators.

    A product POVM :math:`M` is made of local POVMs :math:`M1, M2, ...` acting on respective
    subsystems. Each global effect can be written as the tensor product of local effects,
    :math:`M_{k_1, k_2, ...} = M1_{k_1} \otimes M2_{k_2} \otimes \cdots`.

    Below is an example of how to construct an instance of this class.

    >>> from qiskit.quantum_info import Operator
    >>> from povm_toolbox.quantum_info import SingleQubitPOVM, MultiQubitPOVM, ProductPOVM
    >>> sqp = SingleQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
    >>> mqp = MultiQubitPOVM(
    ...     [
    ...         Operator.from_label("00"),
    ...         Operator.from_label("01"),
    ...         Operator.from_label("10"),
    ...         Operator.from_label("11"),
    ...     ]
    ... )
    >>> product = ProductPOVM({(0,): sqp, (1, 3): mqp, (2,): sqp})

    .. note::
        For most cases, you may find that :meth:`ProductPOVM.from_list` works just fine and is
        easier to use.
    """

    default_dual_class = ProductDual

    def _check_validity(self) -> None:
        """Check if frame axioms are fulfilled for all local frames.

        In addition to the checks performed by the super-class, the following errors may be raised.

        Raises:
            TypeError: if any internal frame is not a :class:`.MultiQubitPOVM` instance.
        """
        for povm in self._frames.values():
            if not isinstance(povm, MultiQubitPOVM):
                raise TypeError(
                    "Expected the internal frame to be of type `MultiQubitPOVM` but found an object"
                    f" of type `{type(povm)}` instead."
                )
            povm._check_validity()

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

        Returns:
            The resulting figure.

        Raises:
            NotImplementedError: if this product POVM contains a :class:`.MultiQubitPOVM` acting on
                more than a single qubit.
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
