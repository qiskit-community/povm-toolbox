# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""ProductDual."""

from __future__ import annotations

import sys

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override  # pragma: no cover

from .base import BaseDual, BaseFrame
from .multi_qubit_dual import MultiQubitDual
from .product_frame import ProductFrame


class ProductDual(ProductFrame[MultiQubitDual], BaseDual):
    r"""Class to represent a set of product Dual operators.

    A product Dual :math:`D` is made of local Duals :math:`D1, D2, ...` acting on respective
    subsystems. Each global effect can be written as the tensor product of local effects,
    :math:`D_{k_1, k_2, ...} = D1_{k_1} \otimes D2_{k__2} \otimes \cdots`.
    """

    @override
    def is_dual_to(self, frame: BaseFrame) -> bool:
        if isinstance(frame, ProductFrame) and set(self.sub_systems) == set(frame.sub_systems):
            return all([self[idx].is_dual_to(frame[idx]) for idx in self.sub_systems])
        # TODO: maybe differentiate two distinct cases:
        #   1) the subsystems are not the same, e.g. ``self`` acts on (0,) and (1,) but ``frame``
        #      acts on (0,) and (2,). Then we should raise an ValueError
        #   2) the subsystems are the same but differently allocated, e.g. ``self`` acts on (0,) and
        #      (1,2) but ``frame`` on (0,1) and (2,). ``self`` could still be a valid dual frame but
        #      we have not implemented the check for this. Then we should raise an
        #      NotImplementedError.
        raise NotImplementedError

    @override
    @classmethod
    def build_dual_from_frame(
        cls, frame: BaseFrame, alphas: tuple[tuple[float, ...] | None, ...] | None = None
    ) -> ProductDual:
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
