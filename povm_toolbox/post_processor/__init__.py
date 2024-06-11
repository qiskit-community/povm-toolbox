# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A module of POVM result post-processing tools.

.. currentmodule:: povm_toolbox.post_processor

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   POVMPostProcessor
   optimal_dual_from_state
   dual_from_marginal_probabilities
   dual_from_empirical_frequencies
"""

from .dual_from_empirical_frequencies import dual_from_empirical_frequencies
from .dual_from_marginal_probabilities import dual_from_marginal_probabilities
from .dual_from_state import optimal_dual_from_state
from .povm_post_processor import POVMPostProcessor

__all__ = [
    "POVMPostProcessor",
    "optimal_dual_from_state",
    "dual_from_marginal_probabilities",
    "dual_from_empirical_frequencies",
]
