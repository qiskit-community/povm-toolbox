"""TODO."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.primitives import BasePrimitiveJob

from povms.library.povm_implementation import POVMImplementation
from povms.sampler.result import POVMSamplerResult


@dataclass
class POVMSamplerJob:
    """Job class for the :class:`POVMSampler`.

    Several pubs can be submitted in a single job. Each of these original pubs
    specifies its POVM and shot budget. The budget is divided between the various
    random measurements composing the POVM. So, for each original pub, several
    `utility pubs' were created. This class retrieves the job of all the `utility
    pubs' that were concatenated together and then groups the results in relation
    to the original pubs.

    Args:
        povm: The list of POVMs that were submitted for each original pub.
        base_job: The raw job from which to extract results. The raw results
            of the job are a flattened list of all the results of the `utility
            pubs' that were concatenated together.
        pvm_keys: The length of the list is equal to the number of original
            pubs. Each element of the list is a list of tuples, where each
            tuple indicates which PVM from the randomized ``povm`` was used
            for a specific `utility pub' result. The flattened list of lists
            would be the same length of the raw job results.
        book_keeping: A list of slices to retrieve the `utility pubs'
            correspopnding to each original pubs. The length of the list
            is equal to the number of original pubs.
    """

    povms: list[POVMImplementation]
    base_job: BasePrimitiveJob
    pvm_keys: list[list[tuple[int, ...]]]
    book_keeping: list[int]

    def result(self) -> list[POVMSamplerResult]:
        """Return the results of the job.

        The results are returned in the form of a list of ``POVMSamplerResult``.
        Each POVM sampler result in the list corresponds to an original pub that
        was submitted.
        """
        result: list[POVMSamplerResult] = []
        raw_result = self.base_job.result()
        slice_start = 0
        for i, slice_length in enumerate(self.book_keeping):
            result.append(
                POVMSamplerResult(
                    self.povms[i],
                    raw_result[slice_start : slice_start + slice_length],
                    self.pvm_keys[i],
                )
            )
            slice_start += slice_length
        return result
