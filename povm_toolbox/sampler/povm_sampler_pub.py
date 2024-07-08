# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""POVMSamplerPub."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Union

from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers import BindingsArrayLike
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.containers.shape import ShapedMixin
from qiskit.transpiler import StagedPassManager

from povm_toolbox.library.metadata import POVMMetadata
from povm_toolbox.library.povm_implementation import POVMImplementation

POVMSamplerPubLike = Union[
    QuantumCircuit,
    tuple[QuantumCircuit],
    tuple[QuantumCircuit, BindingsArrayLike],
    tuple[QuantumCircuit, BindingsArrayLike, Union[Integral, None]],
    tuple[
        QuantumCircuit, BindingsArrayLike, Union[Integral, None], Union[POVMImplementation, None]
    ],
]
"""The type defining the Pub (Primitive Unified Bloc) structure for :meth:`.POVMSampler.run`."""


class POVMSamplerPub(ShapedMixin):
    """The Pub (Primitive Unified Bloc) input structure for :meth:`.POVMSampler.run`.

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
            circuit: the quantum circuit to sample from.
            parameter_values: an optional bindings array for the parameters in the ``circuit``.
            shots: the specific number of shots to run with. This value takes precedence over any
                value supplied to a sampler.
            povm: the specific POVM to run with. This value takes precedence over any POVM supplied
                to a sampler.
            validate: whether to validate the input data.
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
        """The quantum circuit that is being sampled."""
        return self._circuit

    @property
    def parameter_values(self) -> BindingsArray:
        """The bindings array of circuit parameters."""
        return self._parameter_values

    @property
    def shots(self) -> int:
        """The number of shots being sampled."""
        return self._shots

    @property
    def povm(self) -> POVMImplementation:
        """The POVM with which to sample."""
        return self._povm

    @classmethod
    def coerce(
        cls,
        pub: POVMSamplerPubLike,
        *,
        shots: int | None = None,
        povm: POVMImplementation | None = None,
    ) -> POVMSamplerPub:
        """Coerce a :class:`~povm_toolbox.sampler.POVMSamplerPubLike` object.

        Args:
            pub: An object to coerce.
            shots: An optional default number of shots to use if not already specified by the
                pub-like object.
            povm: An optional default POVM to use if not already specified by the pub-like object.

        Raises:
            TypeError: If a number of shots is specified but it is not a positive integer.
            ValueError: If the pub-like object does not specify a number of shots and no default
                number of shots is set or if the pub-like object does not specify a POVM and no
                default POVM is set.
            ValueError: If a tuple is supplied but its length exceeds 4, rendering the pub an
                invalid :class:`~povm_toolbox.sampler.POVMSamplerPubLike`.
            TypeError: If the pub-like object does not have a valid type.

        Returns:
            A coerced POVM sampler pub.
        """
        if isinstance(pub, POVMSamplerPub):
            if pub.shots is None and shots is not None:
                return cls(
                    circuit=pub.circuit,
                    parameter_values=pub.parameter_values,
                    shots=shots,
                    povm=pub.povm,
                    validate=True,
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

            circuit: QuantumCircuit = pub[0]

            if len(pub) > 1 and pub[1] is not None:
                values = pub[1]
                if not isinstance(values, (BindingsArray, Mapping)):
                    values = {tuple(circuit.parameters): values}
                parameter_values = BindingsArray.coerce(values)
            else:
                parameter_values = None

            pub_shots = pub[2] if len(pub) > 2 and pub[2] is not None else shots
            pub_povm = pub[3] if len(pub) > 3 and pub[3] is not None else povm

        else:
            raise TypeError(
                f"Invalid pub-like object submitted. Type {type(pub)} is not supported."
            )

        return cls(
            circuit=circuit,
            parameter_values=parameter_values,
            shots=pub_shots,
            povm=pub_povm,
            validate=True,
        )

    def validate(self) -> None:
        """Validate the pub.

        Raises:
            TypeError: If :attr:`.circuit` is not a :class:`~qiskit.circuit.QuantumCircuit`.
            ValueError: If the pub-like object does not specify a number of shots and that default
                number of shots is set.
            TypeError: If the number of shots is specified but is not a positive integer.
            ValueError: If the number of parameters supplied does not correspond to the number of
                parameters of the circuit.
            ValueError: If the pub-like object does not specify a POVM and no default POVM is set.
        """
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("circuit must be QuantumCircuit.")

        self.parameter_values.validate()

        if self.shots is None:
            raise ValueError(
                "The number of shots must be specified, either for this particular "
                "pub or set a default POVM for all pubs."
            )
        if not isinstance(self.shots, int):
            raise TypeError("shots must be an integer")
        if self.shots <= 0:
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
        if not isinstance(self.povm, POVMImplementation):
            raise TypeError("`povm` must be a `POVMImplementation` instance.")

    def to_sampler_pub(
        self,
        pass_manager: StagedPassManager | None = None,
    ) -> tuple[SamplerPub, POVMMetadata]:
        """Convert this POVM sampler pub to a standard ``SamplerPub``.

        This calls :meth:`~.POVMImplementation.to_sampler_pub` of :attr:`.povm`.

        Args:
            pass_manager: An optional transpilation pass manager. After the supplied circuit has
                been composed with the measurement circuit, the pass manager will be used to
                transpile the composed circuit.

        Returns:
            A tuple of a sampler pub and a dictionary of metadata which includes the
            :class:`.POVMImplementation` object itself. The metadata should contain all the
            information necessary to extract the POVM outcomes out of raw bitstrings.
        """
        return self.povm.to_sampler_pub(
            circuit=self.circuit,
            circuit_binding=self.parameter_values,
            shots=self.shots,
            pass_manager=pass_manager,
        )
