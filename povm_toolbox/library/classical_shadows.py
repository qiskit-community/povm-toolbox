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
from numpy.random import Generator

from .locally_biased_classical_shadows import LocallyBiasedClassicalShadows


class ClassicalShadows(LocallyBiasedClassicalShadows):
    """A classical shadows POVM."""

    def __init__(
        self,
        n_qubit: int,
        shot_batch_size: int = 1,
        seed_rng: int | Generator | None = None,
    ):
        """Construct a classical shadow POVM.

        TODO: The same as above, but also hard-coding the biases to be equally distributed.
        """
        bias = 1.0 / 3.0 * np.ones(3)
        super().__init__(
            n_qubit=n_qubit,
            bias=bias,
            shot_batch_size=shot_batch_size,
            seed_rng=seed_rng,
        )
