# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A module for working with POVMs on a quantum-informational setting.

.. currentmodule:: povm_toolbox.quantum_info

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BasePOVM
   MultiQubitPOVM
   SingleQubitPOVM
   ProductPOVM
"""

from .base_povm import BasePOVM
from .multi_qubit_povm import MultiQubitPOVM
from .product_povm import ProductPOVM
from .single_qubit_povm import SingleQubitPOVM

__all__ = [
    "BasePOVM",
    "MultiQubitPOVM",
    "SingleQubitPOVM",
    "ProductPOVM",
]
