"""TODO."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qiskit.primitives import BaseSamplerV2
from qiskit.primitives.containers import SamplerPubLike
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from povms.library.povm_implementation import POVMImplementation
from povms.sampler.job import POVMSamplerJob
from povms.sampler.povm_sampler_pub import POVMSamplerPub, POVMSamplerPubLike


class POVMSampler:
    """POVM Sampler V2 class."""

    def __init__(
        self,
        sampler: BaseSamplerV2,
    ):
        """Initialize the POVM Sampler.

        Args:
            sampler: the ``BaseSampler`` that will be used to collect the POVM samples.
        """
        self.sampler = sampler

    def run(
        self,
        pubs: Iterable[POVMSamplerPubLike],
        *,
        shots: int | None = None,
        povm: POVMImplementation | None = None,
        multi_job: bool = False,
    ) -> Any:
        """Run and collect samples from each pub.

        Args:
            pubs: An iterable of pub-like objects. For example, a list of circuits
                or tuples ``(circuit, parameter_values, shots, povm)``.
            shots: The total number of shots to sample for each pub that does
                not specify its own shots. If ``None``, each pub has to specify its
                own shots.
            povm: A POVM implementation that defines the measurement to perform
                for each pub that does not specify it own POVM. If ``None``, each pub
                has to specify its own POVM.
            multi_job: If ``True``, create a job for each pub. Otherwise, run all pubs
                in one job.

        Returns:
            The job object of POVMSampler's result.
        """
        # TODO: we need to revisit this as part part of issue #37
        pm = generate_preset_pass_manager(optimization_level=1, backend=self.sampler._backend)

        coerced_povms: list[POVMImplementation] = []
        coerced_pubs: list[list[SamplerPubLike]] = []
        pvm_keys: list[list[tuple[int, ...]]] = []
        for pub in pubs:
            povm_sampler_pub = POVMSamplerPub.coerce(pub=pub, shots=shots, povm=povm)
            sub_pubs, keys = povm_sampler_pub.compose_circuits(pass_manager=pm)
            coerced_povms.append(povm_sampler_pub.povm)
            coerced_pubs.append(sub_pubs)
            pvm_keys.append(keys)

        if multi_job:
            job_list = []
            for i, p in enumerate(coerced_pubs):
                job = self.sampler.run(p)
                job_list.append(POVMSamplerJob([coerced_povms[i]], job, [pvm_keys[i]], [len(p)]))
            return job_list

        # Run all the pubs in one job
        # Flatten the list of pubs and keep track of the corresponding slices
        concat_pubs: list[SamplerPubLike] = []
        book_keeping: list[int] = []
        for p in coerced_pubs:
            book_keeping.append(len(p))
            concat_pubs += p
        job = self.sampler.run(concat_pubs)
        return POVMSamplerJob(coerced_povms, job, pvm_keys, book_keeping)
