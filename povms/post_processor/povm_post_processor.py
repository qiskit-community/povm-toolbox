"""TODO."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from povms.sampler.result import POVMPubResult


class POVMPostprocessor:
    """Class to represent a POVM post-processor.."""

    def __init__(
        self,
        povm_sample: POVMPubResult,
        alphas: np.ndarray | None = None,
    ) -> None:
        """Initialize the POVM post-processor.

        Args:
            povm_sample: a result from a POVM sampler run.
            alphas: parameters of the frame superoperator of the POVM.
        """
        self.povm = povm_sample.povm.to_povm()
        self.counts = povm_sample.get_counts()
        if alphas is not None:
            self.povm.alphas = alphas

    def get_expectation_value(self, observable: SparsePauliOp) -> float:
        """Return the expectation value of a given observable."""
        exp_val = 0.0
        omegas = dict(self.povm.get_omegas(observable, set(self.counts.keys())))  # type: ignore
        for outcome in self.counts:
            exp_val += self.counts[outcome] * omegas[outcome]
        exp_val /= sum(self.counts.values())
        return exp_val
