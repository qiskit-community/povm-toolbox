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

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from povm_toolbox.sampler.result import POVMPubResult


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
