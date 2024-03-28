"""TODO."""

from __future__ import annotations

from qiskit.primitives import BasePrimitiveJob

from povms.library.povm_implementation import POVMImplementation


class POVMSamplerResult:
    """Base class to gather all relevant result information."""

    def __init__(
        self,
        povm: POVMImplementation,
        raw_job: BasePrimitiveJob,
        pvm_keys: list[tuple[int, ...]],
    ) -> None:
        """Initialize the result object.

        Args:
            job: the job from which to exctract the result.
        """
        self.povm = povm
        self.job = raw_job
        self.pvm_keys = pvm_keys

    def get_counts(self):
        """Get the histogram data of an experiment."""
        counts_dict = {}
        split_result = self.job.result()
        for i, pvm_idx in enumerate(self.pvm_keys):
            pub_result = split_result[i]
            pub_counts = pub_result.data.povm_meas.get_counts()
            # TODO: be aware this attribute name depends on the classical register label
            for pvm_outcome in pub_counts:
                povm_outcome = self.povm.get_outcome_label(pvm_idx, pvm_outcome)
                counts_dict[povm_outcome] = pub_counts[pvm_outcome]
        return counts_dict
