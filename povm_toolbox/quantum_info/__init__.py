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

.. note::
   In this module, we use the formalism of frame theory. POVMs are considered special cases
   of *frames* and, for this reason, the *effects* of POVMs are often called *frame operators*.

.. currentmodule:: povm_toolbox.quantum_info

POVM Classes
------------

These classes allow you to build up POVM definitions. You can create POVMs defined over any number
of qubits using the :class:`.MultiQubitPOVM` class and construct tensor products of these using the
:class:`.ProductPOVM` class.

The :class:`.SingleQubitPOVM` is a convenient subclass of the :class:`.MultiQubitPOVM` for the case
of acting on a single qubit. It provides some nice methods for inspecting the POVM visually within a
Bloch sphere, :meth:`.SingleQubitPOVM.draw_bloch`.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   SingleQubitPOVM
   MultiQubitPOVM
   ProductPOVM

Dual Classes
------------

These classes are used to represent dual frames of the POVM objects listed above.
To learn more about dual frames, be sure to check out some of the other material like
`this introduction <../explanations/introduction.rst>`_ or
`this guide on dual frame optimization <../how_tos/dual_optimizer.ipynb>`_.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MultiQubitDual
   ProductDual


Abstract Frames
---------------

Both, the POVM and Dual classes above have much functionality in common because they can be viewed
as `frames`. The classes below implement the common functionalities.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MultiQubitFrame
   ProductFrame

Submodules
----------

.. autosummary::
   :toctree:

   base
"""

from .multi_qubit_dual import MultiQubitDual
from .multi_qubit_frame import MultiQubitFrame
from .multi_qubit_povm import MultiQubitPOVM
from .product_dual import ProductDual
from .product_frame import ProductFrame
from .product_povm import ProductPOVM
from .single_qubit_povm import SingleQubitPOVM

__all__ = [
    "MultiQubitPOVM",
    "SingleQubitPOVM",
    "ProductPOVM",
    "MultiQubitDual",
    "ProductDual",
    "MultiQubitFrame",
    "ProductFrame",
]
