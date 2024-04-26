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

from .randomized_projective_measurements import RandomizedProjectiveMeasurements


class LocallyBiasedClassicalShadows(RandomizedProjectiveMeasurements):
    """TODO."""

    def __init__(
        self,
        n_qubit: int,
        bias: np.ndarray,
        shot_batch_size: int = 1,
    ):
        """Construct a locally-biased classical shadow POVM.

        TODO: The same as above, but the angles are hard-coded to be X/Y/Z for all qubits.
        """
        angles = np.array([0.0, 0.0, 0.5 * np.pi, 0.0, 0.5 * np.pi, 0.5 * np.pi])
        assert bias.shape[-1] == 3
        super().__init__(
            n_qubit=n_qubit,
            bias=bias,
            angles=angles,
            shot_batch_size=shot_batch_size,
        )
