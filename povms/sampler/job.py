"""TODO."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.primitives import BasePrimitiveJob

from povms.library.povm_implementation import POVMImplementation
from povms.sampler.result import POVMSamplerResult


@dataclass
class POVMSamplerJob:
    """Job class for the :class:`POVMSampler`."""

    povm: POVMImplementation
    base_job: BasePrimitiveJob
    pvm_keys: list[tuple[int, ...]]

    def result(self) -> POVMSamplerResult:
        """Return the result of the job."""
        return POVMSamplerResult(self.povm, self.base_job, self.pvm_keys)
