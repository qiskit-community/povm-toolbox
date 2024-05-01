# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A library of POVM implementations.

This module provides "ready-to-go" POVM implementations.
Each one serves as a factory to generate a specific, concrete POVM and provides the means to
`sample` from it.

.. currentmodule:: povm_toolbox.library

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   POVMImplementation
   RandomizedProjectiveMeasurements
   LocallyBiasedClassicalShadows
   ClassicalShadows

Submodules
----------

.. autosummary::
   :toctree:

   metadata
"""

from .classical_shadows import ClassicalShadows
from .locally_biased_classical_shadows import LocallyBiasedClassicalShadows
from .povm_implementation import POVMImplementation
from .randomized_projective_measurements import RandomizedProjectiveMeasurements

__all__ = [
    "POVMImplementation",
    "RandomizedProjectiveMeasurements",
    "LocallyBiasedClassicalShadows",
    "ClassicalShadows",
]
