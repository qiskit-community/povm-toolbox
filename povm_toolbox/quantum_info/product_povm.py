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
from qiskit.quantum_info import DensityMatrix, Operator

from .base_povm import BasePOVM
from .multi_qubit_povm import MultiQubitPOVM
from .product_frame import ProductFrame


class ProductPOVM(ProductFrame[MultiQubitPOVM], BasePOVM):
    r"""Class to represent a set of product POVM operators.

    A product POVM :math:`M` is made of local POVMs :math:`M1, M2, ...` acting
    on respective subsystems. Each global effect can be written as the tensor
    product of local effects,
    :math:`M_{k_1, k_2, ...} = M1_{k_1} \otimes M2_{k2} \otimes ...`.
    """

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
        rho: DensityMatrix,
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
            ``ProductPOVM`` is returned. The length of each dimension is given by the number of outcomes
            of the POVM encoded along that axis.
        """
        return self.analysis(Operator(rho), outcome_idx)
