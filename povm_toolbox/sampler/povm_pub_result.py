# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""POVMPubResult."""

from __future__ import annotations

from collections import Counter

import numpy as np
from qiskit.primitives.containers import DataBin, PubResult

from povm_toolbox.library.metadata import POVMMetadata


class POVMPubResult(PubResult):
    """The result of a :class:`.POVMSamplerJob`."""

    def __init__(
        self,
        data: DataBin,
        metadata: POVMMetadata,
    ) -> None:
        """Initialize the result object.

        Args:
            data: The raw data bin object that contains raw measurement bitstrings.
            metadata: The metadata object that stores the POVM used and all necessary data to
                interpret the raw measurement bitstring. For example, for randomized POVMs, each
                bitstring has to be associated with the corresponding
                :class:`~povm_toolbox.library.metadata.RPMMetadata.pvm_keys` to produce a meaningful
                POVM outcome.
        """
        super().__init__(data, metadata)

    @property
    def metadata(self) -> POVMMetadata:
        """The metadata of this result object.

        .. warning::
           The object returned by instances of this subclass have a different type than dictated by
           the :class:`~qiskit.primitives.containers.pub_result.PubResult` interface.
        """
        return self._metadata  # type:ignore

    def get_counts(self, *, loc: int | tuple[int, ...] | None = None) -> np.ndarray | Counter:
        """Get the histogram data of the result.

        This method will leverage :meth:`~.POVMImplementation.get_povm_counts_from_raw` from the
        :class:`.POVMImplementation` instance stored inside the :attr:`metadata` to construct a
        histogram of POVM outcomes.

        Args:
            loc: Which entry of the :class:`~qiskit.primitives.containers.bit_array.BitArray` to
                return a histogram for. If the Pub that was submitted to :meth:`.POVMSampler.run`
                contained circuit parameters, ``loc`` can be used to indicate the set of parameter
                values for which to compute the histogram. If ``loc is None``, the histogram will be
                computed for all parameter values at once.

        Returns:
            Either a single or an array of histograms of the POVM outcomes. The shape depends on the
            value of ``loc`` and the number of parameters that were submitted in the Pub to
            :meth:`.POVMSampler.run`.
        """
        return self.metadata.povm_implementation.get_povm_counts_from_raw(
            self.data, self.metadata, loc=loc
        )

    def get_samples(
        self, *, loc: int | tuple[int, ...] | None = None
    ) -> np.ndarray | list[tuple[int, ...]]:
        """Get the individual POVM outcomes of the result.

        This method will leverage :meth:`~.POVMImplementation.get_povm_outcomes_from_raw` from the
        :class:`.POVMImplementation` instance stored inside the :attr:`metadata` to recover the
        sampled POVM outcomes.

        Args:
            loc: Which entry of the :class:`~qiskit.primitives.containers.bit_array.BitArray` to
                return the samples for. If the Pub that was submitted to :meth:`.POVMSampler.run`
                contained circuit parameters, ``loc`` can be used to indicate the set of parameter
                values for which to obtain the samples. If ``loc is None``, the samples will be
                obtained for all parameter values at once.

        Returns:
            Either a single or an array of POVM outcomes. The shape depends on the value of ``loc``
            and the number of parameters that were submitted in the Pub to :meth:`.POVMSampler.run`.
        """
        return self.metadata.povm_implementation.get_povm_outcomes_from_raw(
            self.data, self.metadata, loc=loc
        )
