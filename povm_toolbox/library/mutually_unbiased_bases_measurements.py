# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""MutuallyUnbiasedBasesMeasurements."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from scipy.spatial.transform import Rotation

from .randomized_projective_measurements import RandomizedProjectiveMeasurements


class MutuallyUnbiasedBasesMeasurements(RandomizedProjectiveMeasurements):
    """A mutually-unbiased-bases (MUB) POVM.

    This is a special case of a :class:`.RandomizedProjectiveMeasurements` (RPM) POVM. A
    MUB-POVM corresponds to an arbitrary rotated :class:`.LocallyBiasedClassicalShadows`
    (LBCS) POVM. That is, the MUB-POVMs can be seen as lying between RPM- and LBCS-POVMs.
    More precisely, the set of RPM-POVMs includes the set of MUB-POVMs which includes the
    set of LBCS-POVMs.

    The example below shows how you construct a MUB POVM. It plots a visual representation of the
    POVM's definition to exemplify the biased measurement probabilities and rotated measurement
    bases.

    .. plot::
       :include-source:

       >>> import numpy as np
       >>> from povm_toolbox.library import MutuallyUnbiasedBasesMeasurements
       >>> povm = MutuallyUnbiasedBasesMeasurements(
       ...     2,
       ...     bias=np.array([[0.1, 0.6, 0.3], [0.5, 0.25, 0.25]]),
       ...     angles=np.array([[np.pi/4, np.pi/4, np.pi/4], [np.pi/3, np.pi/6, np.pi/3]]),
       ... )
       >>> print(povm)
       MutuallyUnbiasedBasesMeasurements(num_qubits=2, bias=array([[0.1 , 0.6 , 0.3 ], [0.5 , 0.25, 0.25]]),
           angles=array([[0.78539816, 0.78539816, 0.78539816], [1.04719755, 0.52359878, 1.04719755]]))
       >>> povm.definition().draw_bloch()
       <Figure size 1000x500 with 2 Axes>
    """

    def __init__(
        self,
        num_qubits: int,
        bias: np.ndarray,
        angles: np.ndarray,
        *,
        measurement_layout: list[int] | None = None,  # TODO: add | Layout
        measurement_twirl: bool = False,
        shot_repetitions: int = 1,
        insert_barriers: bool = False,
        seed: int | Generator | None = None,
    ) -> None:
        """Initialize a mutually-unbiased-bases (MUB) POVM.

        Args:
            num_qubits: the number of qubits.
            bias: can be either 1D or 2D. If 1D, it should contain float values indicating the bias
                for measuring in each of the PVMs. I.e., its length equals the number of PVMs (3).
                These floats should sum to 1. If 2D, it will have a new set of biases for each
                qubit.
            angles: can be either 1D or 2D. If 1D, it should be of length 3 and contain float values
                to indicate the three Euler angles to rotate the locally-biased classical shadows
                (LBCS) measurement in the Bloch sphere representation. If 2D, it will have a new set
                of angles for each qubit. The angles are expected in the order ``theta``, ``phi``,
                ``lam`` which are the parameters of the :class:`~.qiskit.circuit.library.UGate`
                instance used to rotate the LBCS measurement effects. Note that this differs from
                the angles expected during the initialization of a
                :class:`.RandomizedProjectiveMeasurements` instance, where the angles are expected
                to be pairs of angles ``(theta, phi)`` for each projective measurement forming the
                overall randomized measurement.
            measurement_layout: optional list of indices specifying on which qubits the POVM acts.
                See :attr:`.measurement_layout` for more details.
            measurement_twirl: whether to randomly twirl the measurements. For each single-qubit
                projective measurement, random twirling is equivalent to randomly flipping the
                measurement. This is equivalent to randomly taking the opposite Bloch vector in the
                Bloch sphere representation.
            shot_repetitions: number of times the measurement is repeated for each sampled PVM. More
                precisely, a new PVM is sampled for all ``shots`` (i.e. the number of times as
                specified by the user via the ``shots`` argument of the method
                :meth:`.POVMSampler.run`). Then, the parameter ``shot_repetitions`` states how many
                times we repeat the measurement for each sampled PVM (default is 1). Therefore, the
                effective total number of measurement shots is ``shots`` multiplied by
                ``shot_repetitions``.
            insert_barriers: whether to insert a barrier between the composed circuits. This is not
                done by default but can prove useful when visualizing the composed circuit.
            seed: optional seed to fix the :class:`numpy.random.Generator` used to sample PVMs.
                The MUB measurements are sampled according to the probability distribution(s)
                specified by ``bias``. The user can also directly provide a random generator. If
                ``None``, a random seed will be used.

        Raises:
            ValueError: if the shape of ``bias`` is not valid.
            ValueError: if the shape of ``angles`` is not valid.
        """
        if bias.shape[-1] != 3:
            raise ValueError(
                "The last dimension of ``bias`` is expected to be of length 3, but has"
                f" length {bias.shape[-1]} instead."
            )

        if angles.shape == (3,):
            theta, phi, lam = angles
            processed_angles = self._process_angles(theta, phi, lam)
        elif angles.shape == (num_qubits, 3):
            processed_angles = np.zeros((num_qubits, 6))
            for i, angles_qubit_i in enumerate(angles):
                theta, phi, lam = angles_qubit_i
                processed_angles[i] = self._process_angles(theta, phi, lam)
        else:
            raise ValueError(
                "``angles`` is expected to have shape (3,) or (``num_qubits``, 3)"
                f" but has shape {angles.shape} instead."
            )

        self.rotation_angles: np.ndarray = angles
        """The angles indicating the rotation to obtain the MUB from an otherwise LBCS POVM."""

        super().__init__(
            num_qubits=num_qubits,
            bias=bias,
            angles=processed_angles,
            measurement_twirl=measurement_twirl,
            measurement_layout=measurement_layout,
            shot_repetitions=shot_repetitions,
            insert_barriers=insert_barriers,
            seed=seed,
        )

    def __repr__(self) -> str:
        """Return the string representation of a MutuallyUnbiasedBasesMeasurements instance."""
        return (
            f"{self.__class__.__name__}(num_qubits={self.num_qubits}, bias={self.bias!r}, "
            f"angles={self.rotation_angles!r})"
        )

    @staticmethod
    def _process_angles(theta: float, phi: float, lam: float) -> np.ndarray:
        """Transform the three Euler angles into two for each of the rotated X,Y,Z measurements.

        One way to obtain the rotated measurements would be to first (optionally) rotate the
        Z-measurement into an X- or Y-measurement when applicable and then apply in all cases the
        fixed rotation defined by :class:`~.qiskit.circuit.library.UGate` with parameters ``theta``,
        ``phi`` and ``lam``. However, it means to have two subsequent rotations and therefore two
        unitary gates are added to the circuits. Instead, we can look at the final orientation of
        the rotated measurements and apply a direct rotation from the canonical Z-measurement to the
        respective rotated measurements. Then, only one parametrized rotation gate is needed and two
        angles for each rotated measurement.

        The rotation defined by :class:`~.qiskit.circuit.library.UGate` with parameter ``theta``,
        ``phi`` and ``lam`` is equivalent - in the Bloch sphere representation - to the sequence of
        intrinsic rotations z-y'-z'' for angles ``phi``, ``theta`` and ``lam`` respectively (note
        the changed order of the angles).

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
