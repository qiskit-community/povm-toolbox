# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""LocallyBiasedClassicalShadows."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from .mutually_unbiased_bases_measurements import MutuallyUnbiasedBasesMeasurements


class LocallyBiasedClassicalShadows(MutuallyUnbiasedBasesMeasurements):
    """A locally-biased classical shadows POVM.

    This is a special case of a :class:`MutuallyUnbiasedBasesMeasurements`, where the PVMs are
    chosen to be Z-, X- and Y-measurements. That is, the ``angles`` of the MUB are zero.

    The example below shows how you construct a locally-biased classical shadows POVM. It plots a
    visual representation of the POVM's definition to exemplify the biased measurement
    probabilities.

    .. plot::
       :include-source:

       >>> import numpy as np
       >>> from povm_toolbox.library import LocallyBiasedClassicalShadows
       >>> povm = LocallyBiasedClassicalShadows(2, bias=np.array([[0.1, 0.6, 0.3], [0.5, 0.25, 0.25]]))
       >>> print(povm)
       LocallyBiasedClassicalShadows(num_qubits=2, bias=array([[0.1 , 0.6 , 0.3 ], [0.5 , 0.25, 0.25]]))
       >>> povm.definition().draw_bloch()
       <Figure size 1000x500 with 2 Axes>
    """

    def __init__(
        self,
        num_qubits: int,
        bias: np.ndarray,
        *,
        measurement_layout: list[int] | None = None,  # TODO: add | Layout
        measurement_twirl: bool = False,
        shot_repetitions: int = 1,
        insert_barriers: bool = False,
        seed: int | Generator | None = None,
    ) -> None:
        """Initialize a locally-biased classical shadows POVM.

        Args:
            num_qubits: the number of qubits.
            bias: can be either 1D or 2D. If 1D, it should contain float values indicating the bias
                for measuring in each of the PVMs. I.e., its length equals the number of PVMs (3).
                These floats should sum to 1. If 2D, it will have a new set of biases for each
                qubit.
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
                The Z-,X-,Y-measurements are sampled according to the probability distribution(s)
                specified by ``bias``. The user can also directly provide a random generator. If
                ``None``, a random seed will be used.
        """
        angles = np.array([0.0, 0.0, 0.0])
        assert bias.shape[-1] == 3
        super().__init__(
            num_qubits=num_qubits,
            bias=bias,
            angles=angles,
            measurement_twirl=measurement_twirl,
            measurement_layout=measurement_layout,
            shot_repetitions=shot_repetitions,
            insert_barriers=insert_barriers,
            seed=seed,
        )

    def __repr__(self) -> str:
        """Return the string representation of a LocallyBiasedClassicalShadows instance."""
        return f"{self.__class__.__name__}(num_qubits={self.num_qubits}, bias={self.bias!r})"
