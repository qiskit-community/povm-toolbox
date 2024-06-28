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

from collections import Counter

import numpy as np
from qiskit.primitives.containers import DataBin, PubResult

from povm_toolbox.library.metadata import POVMMetadata


class POVMPubResult(PubResult):
    """Result class for the :class:`.POVMSamplerJob`."""

    def __init__(
        self,
        data: DataBin,
        metadata: POVMMetadata,
    ) -> None:
        """Initialize the result object.

        Args:
            data: The raw data bin object that contains raw measurement
                bitstrings.
            metadata: The `POVMMetadata` object that stores the POVM used
                and all necessary data to interpret the raw measurement bitstring.
                E.g., for randomized POVMs, each bitstring has to be associated
                with the corresponding pvm_key to produce a meaningful POVM outcome.
        """
        super().__init__(data, metadata)

    @property
    def metadata(self) -> POVMMetadata:
        """Note: this subclass returns a different type than its base."""
        return self._metadata  # type:ignore

    def get_counts(self, loc: int | tuple[int, ...] | None = None) -> np.ndarray | Counter:
        """Get the histogram data of an experiment.

        Args:
            loc: Which entry of the ``BitArray`` to return a dictionary for.
                If a ``BindingsArray`` was originally passed to the `POVMSampler``,
                ``loc`` indicates the set of parameter values for which counts are
                to be obtained.
        """
        return self.metadata.povm_implementation.get_povm_counts_from_raw(
            self.data, self.metadata, loc=loc
        )

    def get_samples(
        self, loc: int | tuple[int, ...] | None = None
    ) -> np.ndarray | list[tuple[int, ...]]:
        """Get the individual POVM outcomes of an experiment.

        Args:
            loc: Which entry of the ``BitArray`` to return the samples for.
                If a ``BindingsArray`` was originally passed to the `POVMSampler``,
                ``loc`` indicates the set of parameter values for which counts are
                to be obtained.
        """
        return self.metadata.povm_implementation.get_povm_outcomes_from_raw(
            self.data, self.metadata, loc=loc
        )
