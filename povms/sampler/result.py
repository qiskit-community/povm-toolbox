"""TODO."""

from __future__ import annotations

from qiskit.primitives import PrimitiveResult

from povms.library.povm_implementation import POVMImplementation


class POVMSamplerResult:
    """Base class to gather all relevant result information."""

    def __init__(
        self,
        povm: POVMImplementation,
        result: PrimitiveResult,  # TODO: check type of result objects for V2 primitves, see issue #40
        pvm_keys: list[tuple[int, ...]],
    ) -> None:
        """Initialize the result object.

        Args:
            povm: The POVM that was used to collect the samples.
            result: The raw primitive result object that contains a list of
                the pub results.
            pvm_keys: A list of indices indicating which pvm from the
                randomized ``povm`` was used for each pub result. The length
                of the list should be the same as the length of ``result``.
        """
        self.povm = povm
        self.result = result
        self.pvm_keys = pvm_keys

    def get_counts(self, loc: int | tuple[int, ...] | None = None) -> dict[tuple[int, ...], int]:
        """Get the histogram data of an experiment.

        Args:
            loc: Which entry of the ``BitArray`` to return a dictionary for.
                If a ``BindingsArray`` was originally passed to the `POVMSampler``,
                ``loc`` indicates the set of parameter values for which counts are
                to be obtained.
        """
        counts_dict = {}
        for i, pvm_idx in enumerate(self.pvm_keys):
            pub_result = self.result[i]
            pub_counts = pub_result.data.povm_meas.get_counts(loc)
            # TODO: be aware this attribute name depends on the classical register label
            for pvm_outcome in pub_counts:
                povm_outcome = self.povm.get_outcome_label(pvm_idx, pvm_outcome)
                counts_dict[povm_outcome] = pub_counts[pvm_outcome]
        return counts_dict
