# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""RPMMetadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .povm_metadata import POVMMetadata


@dataclass(repr=False)
class RPMMetadata(POVMMetadata):
    """A metadata container for randomized projective measurements (RPM) POVM sampling results."""

    pvm_keys: np.ndarray
    """The keys which associate a specific result sample with the corresponding RPM parameters.

    Shape of ``pvm_keys`` is assumed to be ``(*pv.shape, num_batches, num_qubits)``,
    where ``pv`` is the bindings array provided by the user to run with the
    parametrized quantum circuit supplied in the :meth:`.POVMSampler.run` method.
    """
