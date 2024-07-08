# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""POVMSampler."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable

from qiskit.primitives import BaseSamplerV2
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.transpiler import StagedPassManager

from povm_toolbox.library.metadata import POVMMetadata
from povm_toolbox.library.povm_implementation import POVMImplementation

from .povm_sampler_job import POVMSamplerJob
from .povm_sampler_pub import POVMSamplerPub, POVMSamplerPubLike

LOGGER = logging.getLogger(__name__)


class POVMSampler:
    """A :class:`~qiskit.primitives.base.base_sampler.BaseSamplerV2`-compatible interface for sampling POVMs.

    This class does not implement the Sampler primitive interface as a subclass but rather takes an
    existing instance of a Sampler primitive as its input. In this way, it is trivial for the
    end-user to switch between simulated or hardware-executed POVM sampling jobs.

    Here is a simple example to get you started:

    >>> from povm_toolbox.library import ClassicalShadows
    >>> from povm_toolbox.sampler import POVMSampler
    >>> from qiskit.circuit import QuantumCircuit
    >>> from qiskit.primitives import StatevectorSampler
    >>> circ = QuantumCircuit(2)
    >>> _ = circ.h(0)
    >>> _ = circ.cx(0, 1)
    >>> povm = ClassicalShadows(2, seed=42)
    >>> sampler = StatevectorSampler(seed=42)
    >>> povm_sampler = POVMSampler(sampler)
    >>> job = povm_sampler.run([circ], povm=povm, shots=16)
    >>> result = job.result()

    Now, if you want to execute your experiments on real hardware, you can simply replace the
    :class:`~qiskit.primitives.StatevectorSampler` above by a
    :class:`~qiskit_ibm_runtime.SamplerV2`:

    .. code:: python

       from qiskit_ibm_runtime import SamplerV2
       sampler = SamplerV2(...)
    """

    def __init__(
        self,
        sampler: BaseSamplerV2,
    ) -> None:
        """Initialize the POVM Sampler.

        Args:
            sampler: the :class:`~qiskit.primitives.base.base_sampler.BaseSamplerV2` that is used to
                collect the POVM samples.
        """
        self.sampler = sampler
        """The :class:`~qiskit.primitives.base.base_sampler.BaseSamplerV2` that is used to collect
        the POVM samples."""

    def run(
        self,
        pubs: Iterable[POVMSamplerPubLike],
        *,
        shots: int | None = None,
        povm: POVMImplementation | None = None,
        pass_manager: StagedPassManager | None = None,
    ) -> POVMSamplerJob:
        """Run and collect samples from each pub.

        Args:
            pubs: An iterable of :class:`~povm_toolbox.sampler.POVMSamplerPubLike` objects. For
                example, a list of circuits or tuples ``(circuit, parameter_values, shots, povm)``.
            shots: The total number of shots to sample for each pub that does not specify its own
                shots. If ``None``, the default number of shots of the :attr:`.sampler` is used.
            povm: A POVM implementation to sample from for each pub that does not specify it own
                POVM. If ``None``, each pub has to specify its own POVM.
            pass_manager: An optional transpilation pass manager. For each pub, after its circuit
                has been composed with the measurement circuit, the pass manager will be used to
                transpile the composed circuit.

        Returns:
            The POVM sampler job object.
        """
        t1_outer = time.time()
        LOGGER.info("Running POVM jobs")

        coerced_sampler_pubs: list[SamplerPub] = []
        metadata: list[POVMMetadata] = []
        for idx, pub in enumerate(pubs):
            t1_inner = time.time()
            LOGGER.info(f"Preparing pub #{idx}")
            povm_sampler_pub = POVMSamplerPub.coerce(pub=pub, shots=shots, povm=povm)
            sampler_pub, pub_metadata = povm_sampler_pub.to_sampler_pub(pass_manager=pass_manager)
            coerced_sampler_pubs.append(sampler_pub)
            metadata.append(pub_metadata)
            t2_inner = time.time()
            LOGGER.info(f"Finished preparation #{idx}. Took {t2_inner - t1_inner:.6f}s")

        job = self.sampler.run(coerced_sampler_pubs)
        povm_job = POVMSamplerJob(job, metadata)

        t2_outer = time.time()
        LOGGER.info(f"Ran job. Took {t2_outer - t1_outer:.6f}s")

        return povm_job
