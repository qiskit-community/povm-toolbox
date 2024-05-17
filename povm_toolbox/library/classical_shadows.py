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
        qubit_specifier: list[int] | None = None,  # TODO: add | Layout
        shot_batch_size: int = 1,
        seed_rng: int | Generator | None = None,
    ):
        """Implement a classical shadows POVM.

        This is a special case of a :class:`LocallyBiasedClassicalShadows`, where
        the bias are taken to be uniform. That is, there is an equal probability
        to perform a measurement in the Z, X and Y bases.

        Args:
            n_qubits: the number of qubits.
            qubit_specifier: list of index specifying on which qubits the POVM acts.
            shot_batch_size: number of shots assigned to each sampled measurement basis.
                If set to 1, a new basis is sampled for each shot.
            seed_rng: optional seed to fix the :class:`numpy.random.Generator` used to
                sample measurement bases. The user can also directly provide a random
                generator. If None, a random seed will be used.
        """
        bias = 1.0 / 3.0 * np.ones(3)
        super().__init__(
            n_qubit=n_qubit,
            bias=bias,
            qubit_specifier=qubit_specifier,
            shot_batch_size=shot_batch_size,
            seed_rng=seed_rng,
        )
