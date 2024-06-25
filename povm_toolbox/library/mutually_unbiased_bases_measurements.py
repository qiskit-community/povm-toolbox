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
from scipy.spatial.transform import Rotation

from .randomized_projective_measurements import RandomizedProjectiveMeasurements


class MutuallyUnbiasedBasesMeasurements(RandomizedProjectiveMeasurements):
    """A mutually-unbiased-bases (MUB) POVM."""

    def __init__(
        self,
        n_qubit: int,
        bias: np.ndarray,
        angles: np.ndarray,
        measurement_twirl: bool = False,
        measurement_layout: list[int] | None = None,  # TODO: add | Layout
        shot_repetitions: int = 1,
        seed_rng: int | Generator | None = None,
    ):
        """Implement a mutually-unbiased-bases (MUB) POVM.

        This is a special case of a :class:`.RandomizedProjectiveMeasurements` (RPM) POVM. A
        MUB-POVM corresponds to an arbitrary rotated :class:`.LocallyBiasedClassicalShadows`
        (LBCS) POVM. That is, the MUB-POVMs can be seen as lying between RPM- and LBCS-POVMs.
        More precisely, the set of RPM-POVMs includes the set of MUB-POVMs which includes the
        set of LBCS-POVMs.

        Args:
            n_qubits: the number of qubits.
            bias: can be either 1D or 2D. If 1D, it should contain float values indicating the bias
                for measuring in each of the PVMs. I.e., its length equals the number of PVMs (3).
                These floats should sum to 1. If 2D, it will have a new set of biases for each
                qubit.
            angles: can be either 1D or 2D. If 1D, it should be of length 3 and contain float values
                to indicate the three Euler angles to rotate the locally-biased classical shadows (LBCS)
                measurement in the Bloch sphere representation. If 2D, it will have a new set of
                angles for each qubit. The angles are expected in the order ``theta``, ``phi``, ``lam``
                which are the parameters of the :class:`.qiskit.circuit.library.UGate` instance used to
                rotate the LBCS measurement effects. This Note that this differs from the angles expected
                during the initialization of a :class:`.RandomizedProjectiveMeasurements` instance,
                where the angles are expected to be pairs of angles ``(theta, phi)`` for each projective
                measurement forming the overall randomized measurement.
            measurement_twirl : option to randomly twirl the measurements. For each single-qubit
                projective measurement, random twirling is equivalent to randomly flipping the
                measurement. This is equivalent to randomly taking the opposite Bloch vector in
                the Bloch sphere representation.
            measurement_layout: list of indices specifying on which qubits the POVM
                acts. If None, two cases can be distinguished: 1) if a circuit supplied
                to the :meth:`.compose_circuits` has been transpiled, its final
                transpile layout will be used as default value, 2) otherwise, a
                simple one-to-one layout ``list(range(n_qubits))`` is used.
            shot_repetitions: number of times the measurement is repeated for each
                sampled PVM. More precisely, a new PVM is sampled for all ``shots``
                (i.e. the number of times as specified by the user via the ``shots``
                argument of the method :meth:`.POVMSampler.run`). Then, the parameter
                ``shot_repetitions`` states how many times we repeat the measurement
                for each sampled PVM (default is 1). Therefore, the effective total
                number of measurement shots is ``shots`` multiplied by ``shot_repetitions``.
            seed_rng: optional seed to fix the :class:`numpy.random.Generator` used to sample PVMs.
                The Z-,X-,Y-measurements are sampled according to the probability distribution(s)
                specified by ``bias``. The user can also directly provide a random generator. If
                None, a random seed will be used.
        """
        if bias.shape[-1] != 3:
            raise ValueError

        if angles.shape == (3,):
            theta, phi, lam = angles
            processed_angles = self._process_angles(theta, phi, lam)
        elif angles.shape == (n_qubit, 3):
            processed_angles = np.zeros((n_qubit, 6))
            for i, angles_qubit_i in enumerate(angles):
                theta, phi, lam = angles_qubit_i
                processed_angles[i] = self._process_angles(theta, phi, lam)
        else:
            raise ValueError

        super().__init__(
            n_qubit=n_qubit,
            bias=bias,
            angles=processed_angles,
            measurement_twirl=measurement_twirl,
            measurement_layout=measurement_layout,
            shot_repetitions=shot_repetitions,
            seed_rng=seed_rng,
        )

    @staticmethod
    def _process_angles(theta: float, phi: float, lam: float) -> np.ndarray:
        """Transform the three Euler angles into two angles for each of the rotated X,Y,Z measurements.

        One way to obtain the rotated measurements would be to first (optionally) rotate the Z-measurement
        into an X- or Y-measurement when applicable and then apply in all cases the fixed rotation defined
        by :class:`.qiskit.circuit.library.UGate` with parameters ``theta``, ``phi`` and ``lam``. However,
        it means to have two subsequent rotations and therefore two unitary gates are added to the circuits.
        Instead, we can look at the final orientation of the rotated measurements and apply a direct rotation
        from the canonical Z-measurement to the respective rotated measurements. Then, only one parametrized
        rotation gate is needed and two angles for each rotated measurement.

        The rotation defined by :class:`.qiskit.circuit.library.UGate` with parameter ``theta``, ``phi`` and
        ``lam`` is equivalent - in the Bloch sphere representation - to the sequence of intrinsic rotations
        z-y'-z'' for angles ``phi``, ``theta`` and ``lam`` respectively (note the changed order of angles).

        Args:
            theta: rotation around the y' axis.
            phi: rotation around the z axis.
            lam: rotation around the z'' axis.

        Returns:
            Flatten array of theta and phi angles of the respective rotated measurements.
        """
        # In the Bloch sphere representation, a UGate(theta, phi, lam) is
        # equivalent to the following sequence of intrinsic rotations:
        r = Rotation.from_euler("ZYZ", [phi, theta, lam])
        # get the Bloch vectors from the rotated Z-,X-,Y-measurements resp.
        bloch_vectors = r.apply([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        # compute rotation angles from [0,0,1] (i.e. Bloch sphere representation of |0>) to the
        # respective Bloch vectors
        thetas = np.arctan2(np.linalg.norm(bloch_vectors[:, :2], axis=1), bloch_vectors[:, 2])
        phis = np.arctan2(bloch_vectors[:, 1], bloch_vectors[:, 0])

        return (np.vstack((thetas, phis)).T).flatten()
