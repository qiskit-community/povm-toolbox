# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""POVMSamplerJob."""

from __future__ import annotations

import logging
import pickle
import sys
import time
import uuid

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override  # pragma: no cover

from qiskit.primitives import BasePrimitiveJob, PrimitiveResult
from qiskit.providers import JobStatus
from qiskit_ibm_runtime import QiskitRuntimeService

from povm_toolbox.library.metadata import POVMMetadata

from .povm_pub_result import POVMPubResult

LOGGER = logging.getLogger(__name__)


class POVMSamplerJob(BasePrimitiveJob[POVMPubResult, JobStatus]):
    """The job returned by :meth:`.POVMSampler.run`."""

    def __init__(
        self,
        base_job: BasePrimitiveJob,
        metadata: list[POVMMetadata],
    ) -> None:
        """Initialize the job.

        Args:
            base_job: the raw job from which to extract results.
            metadata: the metadata list associated with the submitted Pubs.
        """
        super().__init__(job_id=str(uuid.uuid4()))

        self.base_job: BasePrimitiveJob = base_job
        """The internally submitted job instance."""

        self.metadata: list[POVMMetadata] = metadata
        """The metadata list associated with the submitted Pubs."""

    def result(self) -> PrimitiveResult[POVMPubResult]:
        """Return the result of the job.

        Returns:
            A :class:`~qiskit.primitives.PrimitiveResult` containing a list of
            :class:`.POVMPubResult`.

        Raises:
            ValueError: If the number of raw results does not match the number of metadata objects
                stored in :attr:`metadata`.
        """
        t1 = time.time()
        LOGGER.info("Obtaining POVM job result")

        raw_results = self.base_job.result()

        if len(raw_results) != len(self.metadata):
            raise ValueError(
                "The numbers of PUB results and associated POVM metadata objects do not match: "
                f"({len(raw_results)} vs. {len(self.metadata)})."
            )

        povm_pub_results = []

        for pub_result, povm_metadata in zip(raw_results, self.metadata):
            povm_pub_results.append(
                POVMPubResult(
                    data=povm_metadata.povm_implementation.reshape_data_bin(pub_result.data),
                    metadata=povm_metadata,
                )
            )

        res = PrimitiveResult(povm_pub_results, metadata={"raw_results": raw_results})

        t2 = time.time()
        LOGGER.info(f"Finished obtaining POVM result. Took {t2 - t1:.6f}s")

        return res

    def save_metadata(self, filename: str | None = None) -> None:
        """Save the :attr:`metadata` into a pickle file.

        Args:
            filename: name of the file where to store the metadata. If ``None``, the default
                 filename is ``f"job_metadata_{self.base_job.job_id()}.pkl"``.
        """
        if filename is None:
            filename = f"job_metadata_{self.base_job.job_id()}.pkl"
        with open(filename, "wb") as file:
            pickle.dump(
                {
                    "base_job_id": self.base_job.job_id(),
                    "metadata": self.metadata,
                },
                file,
            )
        LOGGER.info(f"Job metadata successfully saved in the '{filename}' file.")

    @staticmethod
    def _load_metadata(filename: str) -> tuple[str, list[POVMMetadata]]:
        """Load the metadata of a :class:`.POVMSamplerJob` instance from a pickle file.

        This is a utility method for loading metadata, which is a part of the job recovery process.
        If you want to perform a full job recovery, this can be achieved through the
        :meth:`.POVMSamplerJob.recover_job` method.

        Args:
            filename: name of the file where the metadata is stored.

        Returns:
            The ID of the internal :class:`.qiskit.primitives.BasePrimitiveJob` object and the list
            of :class:`.POVMMetadata` objects associated to the originally submitted pubs.
        """
        with open(filename, "rb") as file:
            data = pickle.load(file)
        return (
            data["base_job_id"],
            data["metadata"],
        )

    @classmethod
    def recover_job(
        cls,
        filename: str,
        base_job: BasePrimitiveJob | None = None,
        *,
        service: QiskitRuntimeService | None = None,
    ) -> POVMSamplerJob:
        """Recover a :class:`.POVMSamplerJob` instance.

        This method can be used to recover a job instance after previously saving its
        :attr:`metadata` via :meth:`save_metadata`.

        Args:
            filename: name of the file where the metadata is stored.
            base_job: the internal :class:`.qiskit.primitives.BasePrimitiveJob` object that was
                stored inside the original :class:`.POVMSamplerJob` object. If ``None``, the
                internal job ID stored in the metadata will be used to recover the internal job from
                the :class:`~qiskit_ibm_runtime.qiskit_runtime_service.QiskitRuntimeService`.
            service: an optional instance of the :class:`.QiskitRuntimeService`. If ``None``, an
                instance will be generated with no arguments, resulting in it extracting the saved
                configuration from disk.

        Raises:
            ValueError : if a ``base_job`` is supplied and its ID does not match with the ID stored
                in the metadata file ``filename``.

        Returns:
            The recovered :class:`.POVMSamplerJob` instance.
        """
        # Load the saved metadata:
        job_id, metadata = cls._load_metadata(filename)

        if base_job is None:
            if service is None:  # pragma: no cover
                # Use Qiskit Runtime Service to recover the ``BasePrimitiveJob``:
                service = QiskitRuntimeService()  # pragma: no cover
            # Load the ``BasePrimitiveJob`` object:
            base_job = service.job(job_id)
        elif base_job.job_id() != job_id:
            raise ValueError(
                f"The ID of the supplied job ({base_job.job_id()}) does not match the ID stored in "
                f"the metadata file ({job_id})."
            )

        # Return the corresponding :class:`.POVMSampler` object:
        return cls(base_job, metadata)

    @override  # type: ignore[misc]
    def status(self) -> JobStatus:
        return self.base_job.status()

    @override  # type: ignore[misc]
    def done(self) -> bool:
        return bool(self.base_job.done())

    @override  # type: ignore[misc]
    def running(self) -> bool:
        return bool(self.base_job.running())

    @override  # type: ignore[misc]
    def cancelled(self) -> bool:
        return bool(self.base_job.cancelled())

    @override  # type: ignore[misc]
    def in_final_state(self) -> bool:
        return bool(self.base_job.in_final_state())

    @override  # type: ignore[misc]
    def cancel(self):
        self.base_job.cancel()
