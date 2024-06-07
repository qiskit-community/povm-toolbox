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

from .randomized_projective_measurements import RandomizedProjectiveMeasurements


class LocallyBiasedClassicalShadows(RandomizedProjectiveMeasurements):
    """A locally-biased classical shadows POVM."""

    def __init__(
        self,
        n_qubit: int,
        bias: np.ndarray,
        measurement_twirl: bool = False,
        measurement_layout: list[int] | None = None,  # TODO: add | Layout
        shot_batch_size: int = 1,
        seed_rng: int | Generator | None = None,
    ):
        """Implement a locally-biased classical shadow POVM.

        This is a special case of a :class:`RandomizedProjectiveMeasurements`, where the PVMs are
        chosen to be Z-, X- and Y-measurements.

        Args:
            n_qubits: the number of qubits.
            bias: can be either 1D or 2D. If 1D, it should contain float values indicating the bias
                for measuring in each of the PVMs. I.e., its length equals the number of PVMs (3).
                These floats should sum to 1. If 2D, it will have a new set of biases for each
                qubit.
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
            seed_rng: optional seed to fix the :class:`numpy.random.Generator` used to sample PVMs.
                The Z-,X-,Y-measurements are sampled according to the probability distribution(s)
                specified by ``bias``. The user can also directly provide a random generator. If
                None, a random seed will be used.
        """
        angles = np.array([0.0, 0.0, 0.5 * np.pi, 0.0, 0.5 * np.pi, 0.5 * np.pi])
        assert bias.shape[-1] == 3
        super().__init__(
            n_qubit=n_qubit,
            bias=bias,
            angles=angles,
            measurement_twirl=measurement_twirl,
            measurement_layout=measurement_layout,
            shot_batch_size=shot_batch_size,
            seed_rng=seed_rng,
        )
