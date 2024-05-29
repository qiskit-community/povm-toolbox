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

import json
import uuid
from collections.abc import Callable

from qiskit.primitives import BasePrimitiveJob, PrimitiveResult
from qiskit.providers import JobStatus
from qiskit_ibm_runtime import QiskitRuntimeService

from povm_toolbox.library.metadata import POVMMetadata

from .povm_sampler_result import POVMPubResult


class POVMSamplerJob(BasePrimitiveJob[POVMPubResult, JobStatus]):
    """Job class for the :class:`.POVMSampler`."""

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
                were submitted. Each element of the list is a list of tuples, where
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

        Raises:
            ValueError: TODO.
        """
        raw_results = self.base_job.result()

        if len(raw_results) != len(self.metadata):
            raise ValueError(
                "The numbers of PUB results and associated POVM metadata"
                f" objects do not correspond ({len(raw_results)} and"
                f" {len(self.metadata)})."
            )

        povm_pub_results = []

        for pub_result, povm_metadata in zip(raw_results, self.metadata):
            povm_pub_results.append(
                POVMPubResult(
                    data=povm_metadata.povm_implementation.reshape_data_bin(pub_result.data),
                    metadata=povm_metadata,
                )
            )

        return PrimitiveResult(povm_pub_results, metadata={"raw_results": raw_results})

    def serializable_metadata(self) -> list[dict]:
        """Return a serializable list of dictionaries storing all the metadata."""
        return [pub_metadata.to_dict() for pub_metadata in self.metadata]

    def save_metadata(self, filename: str | None = None):
        """Save the metadata of the job into a JSON file."""
        if filename is None:
            filename = f"job_metadata_{self.base_job.job_id()}"
        with open(f"{filename}.json", "w") as file:
            json.dump(
                {
                    "base_job_id": self.base_job.job_id(),
                    "metadata": self.serializable_metadata(),
                },
                file,
            )

    @staticmethod
    def load_metadata(filename: str) -> tuple[str, list[POVMMetadata]]:
        """Load the metadata of the job from a JSON file."""
        with open(f"{filename}.json") as file:
            data = json.load(file)
        return tuple(
            [
                data["base_job_id"],
                [POVMMetadata.from_dict(pub_dict) for pub_dict in data["metadata"]],
            ]
        )

    @classmethod
    def recover_job(
        cls,
        job_id: str | None = None,
        metadata_filename: str | None = None,
        base_job: BasePrimitiveJob | None = None,
        job_loader: Callable[[str], BasePrimitiveJob] | None = None,
    ) -> POVMSamplerJob:
        """TODO."""
        # Check that either a job ID or a filename was provided:
        if (job_id is None) and (metadata_filename is None):
            raise ValueError

        # If ``metadata_filename`` is None, construct the default filename
        # from the job ID:
        metadata_filename = metadata_filename or f"job_metadata_{job_id}"
        # Load the saved metadata:
        loaded_job_id, metadata = cls.load_metadata(metadata_filename)

        if base_job is None:
            # If the job ID was not provided, use the one saved in the metadata:
            if job_id is None:
                job_id = loaded_job_id
            # If the job ID was provided, check that it matches with the one saved
            # in the metadata:
            elif job_id != loaded_job_id:
                raise ValueError

            # Use the default Qiskit Runtime Service loader if none were provided:
            if job_loader is None:
                service = QiskitRuntimeService()
                job_loader = service.job

            # Load the ``BasePrimitiveJob`` object:
            base_job = job_loader(job_id)

        # Return the corresponding :class:`.POVMSampler` object:
        return cls(base_job, metadata)

    def status(self) -> JobStatus:
        """Return the status of the job."""
        return self.base_job.status()

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
