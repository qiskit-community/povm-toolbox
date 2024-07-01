# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""SingleQubitPOVM."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qiskit.visualization.bloch import Bloch
from qiskit.visualization.utils import matplotlib_close_if_inline

from .multi_qubit_povm import MultiQubitPOVM


class SingleQubitPOVM(MultiQubitPOVM):
    """A convenience class to represent a single-qubit :class:`.MultiQubitPOVM` instance.

    Below is a simple example showing how you define a symmetric and informationally-complete POVM
    (SIC-POVM):

    >>> import cmath
    >>> import numpy as np
    >>> from povm_toolbox.quantum_info import SingleQubitPOVM
    >>> vecs = np.sqrt(1.0 / 2.0) * np.array(
    ...     [
    ...         [1, 0],
    ...         [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0)],
    ...         [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0) * cmath.exp(2.0j * np.pi / 3)],
    ...         [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0) * cmath.exp(4.0j * np.pi / 3)],
    ...     ]
    ... )
    >>> sic_povm = SingleQubitPOVM.from_vectors(vecs)
    >>> print(sic_povm)
    SingleQubitPOVM<4> at 0x...
    """

    def _check_validity(self) -> None:
        """Check if POVM axioms are fulfilled.

        In addition to the checks performed by the super-class, the following errors may be raised.

        Raises:
            ValueError: if the dimension does not equal 2 (i.e. the POVM acts on more than 1 qubit).
        """
        if not self.dimension == 2:
            raise ValueError(
                "Dimension of Single Qubit POVM operator space should be 2,"
                f" not {self.dimension}."
            )
        super()._check_validity()

    def get_bloch_vectors(self) -> np.ndarray:
        r"""Compute the Bloch vector of each effect of the POVM.

        For a rank-1 POVM, each effect :math:`M_k` can be written as

        .. math::
            M_k = \gamma_k |\psi_k \rangle \langle \psi_k | = \gamma_k
            \frac{1}{2} \left( \mathbb{I} + \vec{a}_k \cdot \vec{\sigma} \right)

        where :math:`\vec{\sigma}` is the usual Pauli vector and :math:`||\vec{a}_k||^2=1`.
        We then define the Bloch vector of a rank-1 effect as
        :math:`\vec{r}_k = \gamma_k \vec{a}_k`, which uniquely defines the rank-1 effect.

        Example:

        >>> import cmath
        >>> import numpy as np
        >>> from povm_toolbox.quantum_info import SingleQubitPOVM
        >>> vecs = np.sqrt(1.0 / 2.0) * np.array(
        ...     [
        ...         [1, 0],
        ...         [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0)],
        ...         [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0) * cmath.exp(2.0j * np.pi / 3)],
        ...         [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0) * cmath.exp(4.0j * np.pi / 3)],
        ...     ]
        ... )
        >>> sic_povm = SingleQubitPOVM.from_vectors(vecs)
        >>> bloch_vectors = sic_povm.get_bloch_vectors()
        >>> print(bloch_vectors)  # doctest: +FLOAT_CMP
        [[ 0.          0.          0.5       ]
         [ 0.47140452  0.         -0.16666667]
         [-0.23570226  0.40824829 -0.16666667]
         [-0.23570226 -0.40824829 -0.16666667]]

        Returns:
            The Bloch vector of all POVM effects.

        Raises:
            ValueError: if any effect of this POVM has a rank greater than 1.
        """
        r = np.empty((self.num_outcomes, 3))
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
        *,
        title: str = "",
        figure: Figure | None = None,
        axes: Axes | list[Axes] | None = None,
        figsize: tuple[float, float] | None = None,
        font_size: float | None = None,
        colorbar: bool = False,
    ) -> Figure:
        """Plot the Bloch vector of each effect of the POVM.

        .. plot::
           :include-source:

           >>> import cmath
           >>> import numpy as np
           >>> from povm_toolbox.quantum_info import SingleQubitPOVM
           >>> vecs = np.sqrt(1.0 / 2.0) * np.array(
           ...     [
           ...         [1, 0],
           ...         [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0)],
           ...         [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0) * cmath.exp(2.0j * np.pi / 3)],
           ...         [np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0) * cmath.exp(4.0j * np.pi / 3)],
           ...     ]
           ... )
           >>> sic_povm = SingleQubitPOVM.from_vectors(vecs)
           >>> sic_povm.draw_bloch()
           <Figure size 500x500 with 1 Axes>

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
        if figsize is None:
            figsize = (5, 4) if colorbar else (5, 5)

        # Initialize Bloch sphere
        B = Bloch(fig=figure, axes=axes, font_size=font_size)

        # Compute Bloch vector
        vectors = self.get_bloch_vectors()

        if colorbar:
            # Keep track of vector norms through colorbar
            cmap = mpl.colormaps["viridis"]
            B.vector_color = [cmap(np.linalg.norm(vec)) for vec in vectors]
            # Normalize
            for i in range(len(vectors)):
                vectors[i] /= np.linalg.norm(vectors[i])

        B.add_vectors(vectors)
        B.render(title=title)

        if figure is None:
            figure = B.fig
            axes = B.axes
            figure.set_size_inches(figsize[0], figsize[1])
            matplotlib_close_if_inline(figure)

        if colorbar:
            figure.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ax=axes, label="weight")

        return figure
