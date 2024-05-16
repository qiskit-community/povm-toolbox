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
from matplotlib.figure import Figure
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector

from .base_povm import BasePOVM
from .multi_qubit_povm import MultiQubitPOVM
from .product_dual import ProductDUAL
from .product_frame import ProductFrame
from .single_qubit_povm import SingleQubitPOVM


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
        title: str = "",
        figsize: tuple[float, float] | None = None,
        *,
        font_size: float | None = None,
        title_font_size: float | None = None,
        title_pad: float = 1,
        colorbar: bool = False,
    ) -> Figure:
        """Plot a Bloch sphere for each single-qubit POVM forming the product POVM.

        Args:
            title: a string that represents the plot title
            figsize: size of each individual Bloch sphere figure, in inches.
            font_size: Font size for the Bloch ball figures.
            title_font_size: Font size for the title.
            title_pad: Padding for the title (suptitle `y` position is `y=1+title_pad/100`).
            colorbar: TODO.
        """
        import matplotlib.pyplot as plt
        from qiskit.visualization.utils import matplotlib_close_if_inline

        num = self.n_subsystems

        if any([len(idx) > 1 for idx in self.sub_systems]):
            raise NotImplementedError

        w = int(np.sqrt(num) * 4 / 3)
        h = int(np.sqrt(num) * 3 / 4) or 1
        while w * h < num:
            w += 1 if w * h < num else 0
            h += 1 if w * h < num else 0
            while (w - 1) * h >= num:
                w -= 1
        if figsize is None:
            figsize = (5, 4) if colorbar else (5, 5)
        width, height = figsize
        width *= w
        height *= h
        default_title_font_size = font_size if font_size is not None else 16
        title_font_size = (
            title_font_size if title_font_size is not None else default_title_font_size
        )
        fig = plt.figure(figsize=(width, height))
        for i, idx in enumerate(self.sub_systems):
            ax = fig.add_subplot(h, w, i + 1, projection="3d")

            if not isinstance(sqpovm := self[idx], SingleQubitPOVM):
                raise NotImplementedError
            sqpovm.draw_bloch(
                title="qubit " + str(idx[0]),
                fig=fig,
                ax=ax,
                figsize=figsize,
                font_size=font_size,
                colorbar=colorbar,
            )
        fig.suptitle(title, fontsize=title_font_size, y=1.0 + title_pad / 100)
        matplotlib_close_if_inline(fig)
        return fig
