# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""ClassicalShadows."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from .locally_biased_classical_shadows import LocallyBiasedClassicalShadows


class ClassicalShadows(LocallyBiasedClassicalShadows):
    """A classical shadows POVM.

    This is a special case of the :class:`LocallyBiasedClassicalShadows`, where the bias is taken to
    be uniform. That is, there is an equal probability to perform a measurement in the Z, X and Y
    bases.

    The example below shows how you construct a classical shadows POVM. It plots a visual
    representation of the POVM's definition to exemplify the equal measurement probabilities.

    .. plot::
       :include-source:

       >>> from povm_toolbox.library import ClassicalShadows
       >>> povm = ClassicalShadows(2, measurement_twirl=True, shot_repetitions=10)
       >>> print(povm)
       ClassicalShadows(num_qubits=2)
       >>> povm.definition().draw_bloch()
       <Figure size 1000x500 with 2 Axes>
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        measurement_layout: list[int] | None = None,  # TODO: add | Layout
        measurement_twirl: bool = False,
        shot_repetitions: int = 1,
        insert_barriers: bool = False,
        seed: int | Generator | None = None,
    ) -> None:
        """Initialize a classical shadows POVM.

        Args:
            num_qubits: the number of qubits.
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
                The user can also directly provide a random generator. If ``None``, a random seed
                will be used.
        """
        bias = 1.0 / 3.0 * np.ones(3)
        super().__init__(
            num_qubits=num_qubits,
            bias=bias,
            measurement_twirl=measurement_twirl,
            measurement_layout=measurement_layout,
            shot_repetitions=shot_repetitions,
            insert_barriers=insert_barriers,
            seed=seed,
        )

    def __repr__(self) -> str:
        """Return the string representation of a ClassicalShadows instance."""
        return f"{self.__class__.__name__}(num_qubits={self.num_qubits})"
