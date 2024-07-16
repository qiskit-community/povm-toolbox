# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A module of post-processing tools for POVM results.

.. currentmodule:: povm_toolbox.post_processor

Post-processing of the sampled POVM results (see also the :mod:`~povm_toolbox.sampler` module) is
required for the computation of expectation values of the observables of interest. In particular,
one can optimize the Dual frame of a POVM to improve the obtained expectation values.
For more details, refer to `this how-to guide <../how_tos/dual_optimizer.ipynb>`_.

The PostProcessor
-----------------

The main entry-point to the post-processing of POVM results is provided by the
:class:`.POVMPostProcessor` class and its sub-classes.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   POVMPostProcessor
   MedianOfMeans

Various Dual Frames
-------------------

Additionally, this module provides a number of functions to easily construct specific Dual frames.
The functions :func:`.dual_from_state` and :func:`.dual_from_marginal_probabilities` require a
reference state to be available, while :func:`.dual_from_empirical_frequencies` does not.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   dual_from_state
   dual_from_marginal_probabilities
   dual_from_empirical_frequencies
"""

from .dual_from_empirical_frequencies import dual_from_empirical_frequencies
from .dual_from_marginal_probabilities import dual_from_marginal_probabilities
from .dual_from_state import dual_from_state
from .median_of_means import MedianOfMeans
from .povm_post_processor import POVMPostProcessor

__all__ = [
    "POVMPostProcessor",
    "MedianOfMeans",
    "dual_from_state",
    "dual_from_marginal_probabilities",
    "dual_from_empirical_frequencies",
]
