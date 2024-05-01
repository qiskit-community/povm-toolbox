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

from .base_dual import BaseDUAL
from .base_frame import BaseFrame
from .multi_qubit_dual import MultiQubitDUAL
from .product_frame import ProductFrame


class ProductDUAL(ProductFrame[MultiQubitDUAL], BaseDUAL):
    r"""Class to represent a set of product POVM operators.

    A product POVM :math:`M` is made of local POVMs :math:`M1, M2, ...` acting
    on respective subsystems. Each global effect can be written as the tensor
    product of local effects,
    :math:`M_{k_1, k_2, ...} = M1_{k_1} \otimes M2_{k2} \otimes ...`.
    """

    def get_omegas(
        self,
        obs: Operator,
        outcome_idx: tuple[int, ...] | set[tuple[int, ...]] | None = None,
    ) -> float | dict[tuple[int, ...], float] | np.ndarray:
        r"""Return the decomposition weights of observable ``obs`` into the POVM effects.

        Given an observable :math:`O` which is in the span of the POVM, one can write the
        observable :math:`O` as the weighted sum of the POVM effects, :math:`O = \sum_k w_k M_k`
        for real weights :math:`w_k`. There might be infinitely many valid sets of weight.
        This method returns a possible set of weights.

        Args:
            obs: the observable to be decomposed into the POVM effects.

        Returns:
            Decomposition weight(s) associated to the effect(s) specified by ``outcome_idx``.
            If a specific outcome was queried, a ``float`` is returned. If a specific set of outcomes was
            queried, a dictionary mapping outcome labels to weights is returned. If all outcomes were
            queried, a high-dimensional array with one dimension per local POVM stored inside this
            ``ProductPOVM`` is returned. The length of each dimension is given by the number of outcomes
            of the POVM encoded along that axis.
        """
        # TODO: check that obs is Hermitian ?
        return self.analysis(obs, outcome_idx)

    def is_dual_to(self, frame=BaseFrame) -> bool:
        """Check if `self` is a dual to another frame."""
        if isinstance(frame, ProductFrame) and set(self.sub_systems) == set(frame.sub_systems):
            return all([self[idx].is_dual_to(frame[idx]) for idx in self.sub_systems])
        # TODO: maybe differentiate two distinct cases:
        #   1) the subsystems are not the same, e.g. `self` acts on (0,) and (1,) but `frame` acts
        #      on (0,) and (2,). Then we should raise an ValueError
        #   2) the subsystems are the same but differently allocated, e.g. `self` acts on (0,) and
        #      (1,2) but `frame` on (0,1) and (2,). `self` could still be a valid dual frame but we
        #      have not implemented the check for this. Then we should raise an NotImplementedError.
        raise NotImplementedError

    @classmethod
    def build_dual_from_frame(cls, frame=BaseFrame) -> ProductDUAL:
        """Construct a dual frame to another frame."""
        dual_operators = {}
        if isinstance(frame, ProductFrame):
            for sub_system in frame.sub_systems:
                dual_operators[sub_system] = MultiQubitDUAL.build_dual_from_frame(frame[sub_system])
            # TODO : move this test to unittest in the future and just return cls(dual_operators)
            dual_frame = cls(dual_operators)
            if not dual_frame.is_dual_to(frame):
                raise ValueError
            return dual_frame
        raise NotImplementedError
