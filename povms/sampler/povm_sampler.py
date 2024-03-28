"""TODO."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from povms.library.povm_implementation import POVMImplementation
from povms.sampler.job import POVMSamplerJob


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
        povm: POVMImplementation,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        parameter_values: Sequence[float] | Sequence[Sequence[float]] | None = None,
        *,
        shots: int,
    ) -> Any:
        """Run the job of the sampling of the POVM outcomes.

        Args:
            povm: the POVM from which to sample outcomes from.
            cricuits: the circuit or seequence of circuits to be measured.
            parameter_values: values to bind to the parameters of the circuits. The i-th
                circuit ``circuits[i]`` is evaluated with parameters bound as
                ``parameter_values[i]``.

        Returns:
            The job object of the result of the sampler.
        """
        if isinstance(circuits, Sequence):
            raise NotImplementedError
        # TODO: assert circuit qubit routing and stuff

        if parameter_values is not None:
            raise NotImplementedError

        pm = generate_preset_pass_manager(optimization_level=1, backend=self.sampler._backend)
        msmt_qc = povm._build_qc()

        # TODO: assert both circuits are compatible, in particular no measurements at the end of ``circuits``
        # TODO: how to compose classical registers ? CR used for POVM measurements should remain separate
        # TODO: how to deal with transpilation ?
        composed_circuit = circuits.compose(msmt_qc)
        composed_isa_circuit = pm.run(composed_circuit)

        pvm_shots = povm.distribute_shots(shots=shots)
        pubs = [
            (composed_isa_circuit, povm.get_pvm_parameter(pvm_idx), pvm_shots[pvm_idx])
            for pvm_idx in pvm_shots
        ]

        job = self.sampler.run(pubs)

        return POVMSamplerJob(povm, job, list(pvm_shots.keys()))
