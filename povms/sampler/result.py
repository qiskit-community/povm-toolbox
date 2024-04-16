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
from typing import Any

from qiskit.primitives.containers import DataBin, PubResult

from povms.library.povm_implementation import POVMImplementation


class POVMPubResult(PubResult):
    """Base class to gather all relevant result information."""

    def __init__(
        self,
        data: DataBin,
        povm: POVMImplementation,
        pvm_keys: list[tuple[int, ...]],
        metadata: dict[str, Any] | None = None,
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
        super().__init__(data, metadata)
        self.povm = povm
        self.pvm_keys = pvm_keys

    def get_counts(self, loc: int | tuple[int, ...] | None = None):
        """Get the histogram data of an experiment.

        Args:
            loc: Which entry of the ``BitArray`` to return a dictionary for.
                If a ``BindingsArray`` was originally passed to the `POVMSampler``,
                ``loc`` indicates the set of parameter values for which counts are
                to be obtained.
        """
        povm_outcomes = []
        # TODO : improve performance. Currently we loop over all shots and get the
        # outcome label each time. There's probably a way to group equivalent outcomes
        # earlier or do it in a smarter way.

        # TODO : be careful with ``loc``, try to really separate the enduser's parameters
        # locations and the PVM parameter locations !
        if loc is not None:
            raise NotImplementedError("The use of the argument ``loc`` is not yet supported.")

        for i, raw_bitsring in enumerate(self.data.povm_meas.get_bitstrings(loc)):
            povm_outcomes.append(
                self.povm.get_outcome_label(
                    pvm_idx=self.pvm_keys[i], bitstring_outcome=raw_bitsring
                )
            )

        return Counter(povm_outcomes)
