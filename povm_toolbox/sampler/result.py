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

from typing import Any

from qiskit.primitives.containers import DataBin, PubResult

from povm_toolbox.library.povm_implementation import POVMImplementation, POVMMetadata
from povm_toolbox.quantum_info.base_povm import BasePOVM


class POVMPubResult(PubResult):
    """Base class to gather all relevant result information."""

    def __init__(
        self,
        data: DataBin,
        povm_metadata: POVMMetadata,
        pub_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the result object.

        Args:
            data: The raw data bin object that contains raw measurement
                bitstrings. Each bitstring has to be associated with the
                corresponding pvm_key to produce a meaningful POVM outcvome.
            povm: The POVM that was used to collect the samples.
            pvm_keys: A list of indices indicating which pvm from the
                randomized ``povm`` was used for each shot. The length
                of the list should be the same as the number of shots.
        """
        super().__init__(data, pub_metadata)
        self.povm_metadata = povm_metadata

    @property
    def povm_implementation(self) -> POVMImplementation:
        """Return the ``POVMImplementation`` associated with the result."""
        return self.povm_metadata.povm

    @property
    def povm(self) -> BasePOVM:
        """Return the ``BasePOVM`` associated with the result."""
        return self.povm_metadata.povm.to_povm()

    def get_counts(self, loc: int | tuple[int, ...] | None = None):
        """Get the histogram data of an experiment.

        Args:
            loc: Which entry of the ``BitArray`` to return a dictionary for.
                If a ``BindingsArray`` was originally passed to the `POVMSampler``,
                ``loc`` indicates the set of parameter values for which counts are
                to be obtained.
        """
        return self.povm_implementation.get_counts_from_raw(
            data=self.data, povm_metadata=self.povm_metadata, loc=loc
        )
