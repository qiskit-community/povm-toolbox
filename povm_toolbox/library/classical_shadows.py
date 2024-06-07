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
        measurement_twirl: bool = False,
        measurement_layout: list[int] | None = None,  # TODO: add | Layout
        shot_batch_size: int = 1,
        seed_rng: int | Generator | None = None,
    ):
        """Implement a classical shadows POVM.

        This is a special case of a :class:`LocallyBiasedClassicalShadows`, where
        the bias are taken to be uniform. That is, there is an equal probability
        to perform a measurement in the Z, X and Y bases.

        Args:
            n_qubits: the number of qubits.
            measurement_twirl : option to randomly twirl the measurements. For each single-qubit
                projective measurement, random twirling is equivalent to randomly flipping the
                measurement. This is equivalent to randomly taking the opposite Bloch vector in
                the Bloch sphere representation.
            measurement_layout: list of indices specifying on which qubits the POVM
                acts. If None, two cases can be distinguished: 1) if a circuit supplied
                to the :meth:`.compose_circuits` has been transpiled, its final
                transpile layout will be used as default value, 2) otherwise, a
                simple one-to-one layout ``list(range(n_qubits))`` is used.
            shot_batch_size: number of shots assigned to each sampled PVM. If set
                to 1, a new PVM is sampled for each shot. Note that the ``shots``
                argument of the method :meth:`.POVMSampler.run` is effectively the
                number of batches (i.e., the number of sampled PVMs). The actual
                total number of shots is then ``shots``  multiplied by ``shot_batch_size``.
            seed_rng: optional seed to fix the :class:`numpy.random.Generator` used to
                sample measurement bases. The user can also directly provide a random
                generator. If None, a random seed will be used.
        """
        bias = 1.0 / 3.0 * np.ones(3)
        super().__init__(
            n_qubit=n_qubit,
            bias=bias,
            measurement_twirl=measurement_twirl,
            measurement_layout=measurement_layout,
            shot_batch_size=shot_batch_size,
            seed_rng=seed_rng,
        )
