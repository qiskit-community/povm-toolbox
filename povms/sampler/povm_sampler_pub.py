"""TODO."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Union

from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers import BindingsArrayLike, SamplerPubLike
from qiskit.transpiler import StagedPassManager

from povms.library.povm_implementation import POVMImplementation

POVMSamplerPubLike = Union[
    QuantumCircuit,
    tuple[QuantumCircuit],
    tuple[QuantumCircuit, BindingsArrayLike],
    tuple[QuantumCircuit, BindingsArrayLike, Union[Integral, None]],
    tuple[
        QuantumCircuit, BindingsArrayLike, Union[Integral, None], Union[POVMImplementation, None]
    ],
]


@dataclass
class POVMSamplerPub:
    """Class to represent a POVM sampler pub."""

    circuit: QuantumCircuit
    parameter_values: BindingsArrayLike
    shots: int
    povm: POVMImplementation

    @classmethod
    def coerce(
        cls,
        pub: POVMSamplerPubLike,
        *,
        shots: int | None = None,
        povm: POVMImplementation | None = None,
    ) -> POVMSamplerPub:
        """Coerce a ``POVMSamplerPubLike`` object into a ``POVMSamplerPub`` instance.

        Args:
            pub: An object to coerce.
            shots: An optional default number of shots to use if not
                already specified by the pub-like object.
            povm: An optional default POVM to use if not already specified
                by the pub-like object.

        Raises:
            NotImplementedError: If paremeter values to be bound to a paremetric
                circuit is passed as an argument in the pub-like object.
            TypeError: If the pub-like object does not have a valid type.
            ValueError: If the pub-like object does not specify a number of shots
                and that no default number of shots is set.
            ValueError: If the pub-like object does not specify a povm and that
                no default povm is set.

        Returns:
            A coerced POVM sampler pub.
        """
        qc: QuantumCircuit
        if isinstance(pub, QuantumCircuit):
            qc = pub
            pub_shots = shots
            pub_povm = povm
        elif isinstance(pub, tuple):
            qc = pub[0]
            if len(pub) >= 2 and pub[1] is not None:
                raise NotImplementedError(
                    "Not yet able to pass parametric circuits and binding values as arguments."
                )
            pub_shots = pub[2] if (len(pub) >= 3 and isinstance(pub[2], int)) else shots
            pub_povm = pub[3] if len(pub) == 4 and pub[3] is not None else povm
        else:
            raise TypeError(f"An invalid POVM Sampler pub-like was given ({type(pub)}). ")

        parameter_values = None

        if pub_shots is None:
            raise ValueError(
                "The number of shots must be specified, either for this particular pub or set a default number."
            )

        if pub_povm is None:
            raise ValueError(
                "The POVM must be specified, either for this particular pub or set a default POVM."
            )

        return cls(qc, parameter_values, pub_shots, pub_povm)

    def compose_circuits(
        self,
        pass_manager: StagedPassManager,
    ) -> tuple[list[SamplerPubLike], list[tuple[int, ...]]]:
        """Compose the pub circuit with measurement circuit(s).

        Compose the internal quantum circuit with the measurement circuits of the
        internal POVM. If a randomized POVM is used, sevral ``SamplerPubLike``
        object are returned, one for each random measurement with its associated
        number of shots.

        Args:
            pass_manager: A staged pass manger to compile composed circuits.

        Raises:
            NotImplementedError: If paremeter values to be bound to a paremetric
                circuit is passed as an argument in the pub-like object.

        Returns:
            A tuple of a list of sampler pubs and a list of keys indicating which random
            measurement is used for each pub.
        """
        # TODO: assert circuit qubit routing and stuff
        # TODO: assert both circuits are compatible, in particular no measurements at the end of ``circuits``
        # TODO: how to compose classical registers ? CR used for POVM measurements should remain separate
        # TODO: how to deal with transpilation ?

        composed_circuit = self.circuit.compose(self.povm.msmt_qc)
        composed_isa_circuit = pass_manager.run(composed_circuit)

        pvm_shots = self.povm.distribute_shots(shots=self.shots)

        if self.parameter_values is not None:
            raise NotImplementedError(
                "Not yet able to pass parametric circuits and binding values as arguments."
            )

        sub_pubs = [
            (
                composed_isa_circuit,
                self.povm.get_pvm_parameter(pvm_idx),
                pvm_shots[pvm_idx],
            )
            for pvm_idx in pvm_shots
        ]

        return (sub_pubs, list(pvm_shots.keys()))
