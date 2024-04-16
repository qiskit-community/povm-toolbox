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

from collections.abc import Iterable

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
    ) -> None:
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
    ) -> POVMSamplerJob:
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

        Returns:
            The POVM sampler job object.
        """
        # TODO: we need to revisit this as part part of issue #37
        pm = generate_preset_pass_manager(optimization_level=1, backend=self.sampler._backend)

        # Run all the pubs in one job
        # Flatten the list of pubs and keep track of the corresponding slices
        coerced_povms: list[POVMImplementation] = []
        coerced_sampler_pubs: list[SamplerPubLike] = []
        pvm_keys: list[list[tuple[int, ...]]] = []
        for pub in pubs:
            povm_sampler_pub = POVMSamplerPub.coerce(pub=pub, shots=shots, povm=povm)
            composed_pub, keys = povm_sampler_pub.compose_circuits(pass_manager=pm)
            coerced_povms.append(povm_sampler_pub.povm)
            coerced_sampler_pubs.append(composed_pub)
            pvm_keys.append(keys)

        job = self.sampler.run(coerced_sampler_pubs)
        return POVMSamplerJob(coerced_povms, job, pvm_keys)
