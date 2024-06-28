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
from qiskit.quantum_info import Operator, SparsePauliOp

from .base_dual import BaseDual
from .base_frame import BaseFrame
from .multi_qubit_dual import MultiQubitDual
from .product_frame import ProductFrame


class ProductDual(ProductFrame[MultiQubitDual], BaseDual):
    r"""Class to represent a set of product Dual operators.

    A product Dual :math:`D` is made of local Dual :math:`D1, D2, ...` acting
    on respective subsystems. Each global effect can be written as the tensor
    product of local effects,
    :math:`D_{k_1, k_2, ...} = D1_{k_1} \otimes D2_{k2} \otimes ...`.
    """

    def get_omegas(
        self,
        observable: SparsePauliOp | Operator,
        outcome_idx: tuple[int, ...] | set[tuple[int, ...]] | None = None,
    ) -> float | dict[tuple[int, ...], float] | np.ndarray:
        r"""Return the decomposition weights of the provided observable into the POVM effects to which ``self`` is a dual.

        Here the POVM itself is not explicitly needed, its dual is sufficient. Given
        an observable :math:`O` which is in the span of a given POVM, one can write the
        observable :math:`O` as the weighted sum of the POVM effects, :math:`O = \sum_k w_k M_k`
        for real weights :math:`w_k`. There might be infinitely many valid sets of weight.
        This method returns a possible set of weights.

        Args:
            observable: the observable to be decomposed into the POVM effects.
            outcome_idx: the outcomes for which one queries the probability. Each outcome is labeled
                by a tuple of integers (one index per local POVM). One can query a single outcome or a
                set of outcomes. If ``None``, all outcomes are queried.


        Returns:
            Decomposition weight(s) associated to the effect(s) specified by ``outcome_idx``.
            If a specific outcome was queried, a ``float`` is returned. If a specific set of outcomes was
            queried, a dictionary mapping outcome labels to weights is returned. If all outcomes were
            queried, a high-dimensional array with one dimension per local POVM stored inside this
            :class:`.ProductPOVM` instance is returned. The length of each dimension is given by
            the number of outcomes of the POVM encoded along that axis.
        """
        # TODO: check that observable is Hermitian ?
        return self.analysis(observable, outcome_idx)

    def is_dual_to(self, frame: BaseFrame) -> bool:
        """Check if ``self`` is a dual to another frame."""
        if isinstance(frame, ProductFrame) and set(self.sub_systems) == set(frame.sub_systems):
            return all([self[idx].is_dual_to(frame[idx]) for idx in self.sub_systems])
        # TODO: maybe differentiate two distinct cases:
        #   1) the subsystems are not the same, e.g. ``self`` acts on (0,) and (1,) but ``frame`` acts
        #      on (0,) and (2,). Then we should raise an ValueError
        #   2) the subsystems are the same but differently allocated, e.g. ``self`` acts on (0,) and
        #      (1,2) but ``frame`` on (0,1) and (2,). ``self`` could still be a valid dual frame but we
        #      have not implemented the check for this. Then we should raise an NotImplementedError.
        raise NotImplementedError

    @classmethod
    def build_dual_from_frame(
        cls, frame: BaseFrame, alphas: tuple[tuple[float, ...] | None, ...] | None = None
    ) -> ProductDual:
        """Construct a dual frame to another frame.

        Args:
            frame: the primal frame from which we will build the dual frame.
            alphas: parameters of the local frame super-operators used to build
                the local dual frames which form together the product dual frame.
                If None, the parameters are set as the traces of each local operator
                in each of the primal frames.

        Returns:
            A product dual frame to the supplied ``frame``.
        """
        dual_operators = {}
        if isinstance(frame, ProductFrame):
            if alphas is None:
                alphas = len(frame.sub_systems) * (None,)
            elif len(alphas) != len(frame.sub_systems):
                raise ValueError(
                    f"The number of sets of alpha-parameters ({len(alphas)}) does not match"
                    f" the number of sub-systems ({len(frame.sub_systems)})."
                )
            for sub_system, sub_alphas in zip(frame.sub_systems, alphas):
                dual_operators[sub_system] = MultiQubitDual.build_dual_from_frame(
                    frame[sub_system], sub_alphas
                )
            return cls(dual_operators)
        raise NotImplementedError
