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

from numbers import Integral
from typing import Union

from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers import BindingsArrayLike, SamplerPubLike
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.shape import ShapedMixin
from qiskit.transpiler import StagedPassManager

from povm_toolbox.library.povm_implementation import POVMImplementation, POVMMetadata

POVMSamplerPubLike = Union[
    QuantumCircuit,
    tuple[QuantumCircuit],
    tuple[QuantumCircuit, BindingsArrayLike],
    tuple[QuantumCircuit, BindingsArrayLike, Union[Integral, None]],
    tuple[
        QuantumCircuit, BindingsArrayLike, Union[Integral, None], Union[POVMImplementation, None]
    ],
]


class POVMSamplerPub(ShapedMixin):
    """Pub (Primitive Unified Bloc) for a POVM Sampler.

    Pub is composed of tuple (circuit, parameter_values, shots, povm_implementation).
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        parameter_values: BindingsArray | None,
        shots: int,
        povm: POVMImplementation,
        *,
        validate: bool = True,
    ):
        """Initialize a sampler pub.

        Args:
            circuit: A quantum circuit.
            parameter_values: A bindings array.
            shots: A specific number of shots to run with. This value takes
                precedence over any value owed by or supplied to a sampler.
            povm: A specific povm to run with. This povm takes precedence
                over any povm supplied to a sampler.
            validate: If ``True``, the input data is validated during initialization.
        """
        super().__init__()
        self._circuit = circuit
        self._parameter_values = parameter_values or BindingsArray()
        self._shots = shots
        self._povm = povm
        self._shape = self._parameter_values.shape
        if validate:
            self.validate()

    @property
    def circuit(self) -> QuantumCircuit:
        """A quantum circuit."""
        return self._circuit

    @property
    def parameter_values(self) -> BindingsArray:
        """A bindings array."""
        return self._parameter_values

    @property
    def shots(self) -> int:
        """An specific number of shots to run with."""
        return self._shots

    @property
    def povm(self) -> POVMImplementation:
        """An specific povm to run with."""
        return self._povm

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
            TypeError: If number of shots is specified but is not an integer.
            TypeError: If the specified number of shots is negative.
            ValueError: If the pub-like object does not specify a number of shots
                and that no default number of shots is set or if the pub-like
                object does not specify a povm and that no default povm is set.
            ValueError: If a tuple is supplied but its length exceed 4.
            NotImplementedError: If paremeter values to be bound to a paremetric
                circuit is passed as an argument in the pub-like object.
            TypeError: If the pub-like object does not have a valid type.

        Returns:
            A coerced POVM sampler pub.
        """
        # Validate shots kwarg if provided
        if shots is not None:
            if not isinstance(shots, int) or isinstance(shots, bool):
                raise TypeError("shots must be an integer")
            if shots < 0:
                raise ValueError("shots must be positive")

        if isinstance(pub, POVMSamplerPub):
            if pub.shots is None and shots is not None:
                return cls(
                    circuit=pub.circuit,
                    parameter_values=pub.parameter_values,
                    shots=shots,
                    povm=pub.povm,
                    validate=False,  # Assume Pub is already validated
                )
            return pub

        if isinstance(pub, QuantumCircuit):
            if shots is None or povm is None:
                raise ValueError(
                    "Only a quantum circuit was submitted and either no default number"
                    " of shots or default povm were set."
                )
            return cls(circuit=pub, parameter_values=None, shots=shots, povm=povm, validate=True)

        if isinstance(pub, tuple):
            if len(pub) not in {1, 2, 3, 4}:
                raise ValueError(
                    f"The length of pub must be 1, 2, 3 or 4, but length {len(pub)} is given."
                )

            qc: QuantumCircuit = pub[0]

            if len(pub) >= 2 and pub[1] is not None:
                raise NotImplementedError(
                    "Not yet able to pass parametric circuits and binding values as arguments."
                )

            parameter_values = None

            pub_shots = pub[2] if len(pub) > 2 and pub[2] is not None else shots
            pub_povm = pub[3] if len(pub) > 3 and pub[3] is not None else povm

        else:
            raise TypeError(
                f"Invalid pub-like object submitted. Type {type(pub)} is not supported."
            )

        return cls(
            circuit=qc,
            parameter_values=parameter_values,
            shots=pub_shots,
            povm=pub_povm,
            validate=True,
        )

    def validate(self):
        """Validate the pub.

        Raises:
            TypeError: If circuit is not a ``QuantumCircuit``.
            ValueError: If the pub-like object does not specify a number of shots
                and that no default number of shots is set.
            TypeError: If number of shots is specified but is not an integer.
            TypeError: If the specified number of shots is negative.
            ValueError: If the number of parameters supplied does not correspond
                to the number of parameters of the circuit.
            ValueError: If the pub-like object does not specify a povm and that
                no default povm is set.
        """
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("circuit must be QuantumCircuit.")

        self.parameter_values.validate()

        if self.shots is None:
            raise ValueError(
                "The number of shots must be specified, either for this particular "
                "pub or set a default POVM for all pubs."
            )
        if not isinstance(self.shots, Integral) or isinstance(self.shots, bool):
            raise TypeError("shots must be an integer")
        if self.shots < 0:
            raise ValueError("shots must be positive")

        # Cross validate circuits and parameter values
        num_parameters = self.parameter_values.num_parameters
        if num_parameters != self.circuit.num_parameters:
            message = (
                f"The number of values ({num_parameters}) does not match "
                f"the number of parameters ({self.circuit.num_parameters}) for the circuit."
            )
            if num_parameters == 0:
                message += (
                    " Note that if you want to run a single pub, you need to wrap it with `[]` like "
                    "`sampler.run([(circuit, param_values)])` instead of "
                    "`sampler.run((circuit, param_values))`."
                )
            raise ValueError(message)

        if self.povm is None:
            raise ValueError(
                "The POVM must be specified, either for this particular pub or "
                "set a default POVM for all pubs."
            )

    def to_sampler_pub(
        self,
        pass_manager: StagedPassManager,
    ) -> tuple[SamplerPubLike | list[SamplerPubLike], POVMMetadata]:
        """TODO."""
        return self.povm.to_sampler_pub(
            self.circuit, self.parameter_values, self.shots, pass_manager
        )
