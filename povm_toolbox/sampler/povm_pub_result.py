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

import sys
from collections import Counter

if sys.version_info < (3, 10):
    from typing import Any  # pragma: no cover

    # There is no way to support this type properly in python 3.9, which will be end of life in
    # November 2025 anyways.
    EllipsisType = Any  # pragma: no cover
else:
    from types import EllipsisType

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

    def get_counts(
        self, *, loc: int | tuple[int, ...] | EllipsisType | None = None
    ) -> np.ndarray | Counter:
        """Get the counter of outcomes from the result.

        This method will leverage :meth:`~.POVMImplementation.get_povm_counts_from_raw` from the
        :class:`.POVMImplementation` instance stored inside the :attr:`metadata` to construct a
        counter of POVM outcomes.

        Args:
            loc: specifies the location of the counts to return. By default, ``None`` is used, which
                aggregates all counts from a single PUB. If ``loc=...``, all counts from the PUB are
                returned, but separately. If ``loc`` is a tuple of integers, it must define a single
                parameter set. Refer to
                `this how-to guide <../how_tos/combine_outcomes.ipynb>`_ for more information.

        Returns:
            The POVM counts. If ``loc=...``, an ``np.ndarray`` of counters is returned. Otherwise, a
            single counter is returned.
        """
        return self.metadata.povm_implementation.get_povm_counts_from_raw(
            self.data, self.metadata, loc=loc
        )

    def get_samples(
        self, *, loc: int | tuple[int, ...] | EllipsisType | None = None
    ) -> np.ndarray | list[tuple[int, ...]]:
        """Get the individual POVM outcomes of the result.

        This method will leverage :meth:`~.POVMImplementation.get_povm_outcomes_from_raw` from the
        :class:`.POVMImplementation` instance stored inside the :attr:`metadata` to recover the
        sampled POVM outcomes.

        Args:
            loc: specifies the location of the outcomes to return. By default, ``None`` is used, which
                aggregates all outcomes from a single PUB. If ``loc=...``, all outcomes from the PUB
                are returned, but separately. If ``loc`` is a tuple of integers, it must define a
                single parameter set. Refer to
                `this how-to guide <../how_tos/combine_outcomes.ipynb>`_ for more information.

        Returns:
            The list of POVM outcomes. If ``loc=...``, an ``np.ndarray`` of outcome lists is returned.
            Otherwise, a single outcome list is returned.
        """
        return self.metadata.povm_implementation.get_povm_outcomes_from_raw(
            self.data, self.metadata, loc=loc
        )
