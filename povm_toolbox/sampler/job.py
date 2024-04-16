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

import uuid

from qiskit.primitives import BasePrimitiveJob, PrimitiveResult
from qiskit.providers import JobStatus

from povm_toolbox.library.povm_implementation import POVMMetadata
from povm_toolbox.sampler.result import POVMPubResult


class POVMSamplerJob(BasePrimitiveJob[POVMPubResult, JobStatus]):
    """Job class for the :class:`POVMSampler`."""

    def __init__(
        self,
        base_job: BasePrimitiveJob,
        metadata: list[POVMMetadata],
    ) -> None:
        """Initialize the job.

        Args:
            povm: The list of POVMs that were submitted for each pub.
            base_job: The raw job from which to extract results.
            pvm_keys: The length of the list is equal to the number of pubs that
                were submited. Each element of the list is a list of tuples, where
                each tuple indicates which PVM from the randomized ``povm`` was
                used for a specific shot. The length of each nested list is equal
                the number of shots associated with the corresponding pub.
        """
        super().__init__(job_id=str(uuid.uuid4()))

        self.base_job = base_job
        self.metadata = metadata

    def result(self) -> PrimitiveResult[POVMPubResult]:
        """Return the results of the job.

        Returns:
            A ``PrimitiveResult`` containing a list of ``POVMPubResult``.
        """
        raw_results = self.base_job.result()

        if len(raw_results) != len(self.metadata):
            raise ValueError

        povm_pub_results = []

        for pub_result, povm_metadata in zip(raw_results, self.metadata):
            # TODO : something like this to change the number of shots of the bit_array
            # data = pub_result.data
            # bit_array = data.povm_meas # shape=n_shots, num_shots = 1
            # bit_array2 = BitArray(np.squeeze(bit_array.array, axis=-2), bit_array.num_bits) # shape=(), num_shots = n_shots
            # data.povm_meas = bit_array2
            povm_pub_results.append(
                POVMPubResult(
                    data=pub_result.data,
                    povm_metadata=povm_metadata,
                    pub_metadata=pub_result.metadata,
                )
            )

        return PrimitiveResult(povm_pub_results, metadata=raw_results.metadata)

    def status(self) -> JobStatus:
        """Return the status of the job."""
        raise NotImplementedError("Subclass of BasePrimitiveJob must implement `status` method.")

    def done(self) -> bool:
        """Return whether the job has successfully run."""
        return bool(self.base_job.done())

    def running(self) -> bool:
        """Return whether the job is actively running."""
        return bool(self.base_job.running())

    def cancelled(self) -> bool:
        """Return whether all jobs have been cancelled."""
        return bool(self.base_job.cancelled())

    def in_final_state(self) -> bool:
        """Return whether the job is in a final job state such as ``DONE`` or ``ERROR``."""
        return bool(self.base_job.in_final_state())

    def cancel(self):
        """Attempt to cancel the job."""
        self.base_job.cancel()
