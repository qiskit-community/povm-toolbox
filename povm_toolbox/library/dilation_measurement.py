# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""DilationMeasurements."""

from __future__ import annotations

import logging
import sys
import time

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override  # pragma: no cover

import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.primitives.containers import DataBin
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.bit_array import BitArray
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.transpiler import StagedPassManager

from povm_toolbox.quantum_info import ProductPOVM, SingleQubitPOVM
from povm_toolbox.utilities import gs, n_sphere

from .metadata import POVMMetadata
from .povm_implementation import POVMImplementation

LOGGER = logging.getLogger(__name__)


class DilationMeasurements(POVMImplementation[POVMMetadata]):
    """A measurement leveraging Naimark's dilation theorem.

    .. note::
        An additional ancilla qubit is required for each qubit in the system to be measured.

    The example below shows how you construct a dilation POVM. It plots a visual representation of the
    POVM's definition to exemplify the different effects' directions.

    .. plot::
       :include-source:

       >>> import numpy as np
       >>> from povm_toolbox.library import DilationMeasurements
       >>> povm = DilationMeasurements(
       ...     1,
       ...     parameters=np.array(
       ...         [0.75, 0.30408673, 0.375, 0.40678524, 0.32509973, 0.25000035, 0.49999321, 0.83333313]
       ...     ),
       ... )
       >>> print(povm)
       DilationMeasurements(num_qubits=1, parameters=array([[0.75      , 0.30408673, 0.375     , 0.40678524, 0.32509973,
               0.25000035, 0.49999321, 0.83333313]]))
       >>> povm.definition().draw_bloch()
       <Figure size 500x500 with 1 Axes>
    """

    def __init__(
        self,
        num_qubits: int,
        parameters: np.ndarray,
        *,
        measurement_layout: list[int] | None = None,  # TODO: add | Layout
        measurement_twirl: bool = False,
    ) -> None:
        super().__init__(num_qubits, measurement_layout=measurement_layout)

        if parameters.shape[-1] != 8:
            raise ValueError()

        if parameters.ndim == 1:
            parameters = np.tile(parameters, (self.num_qubits, 1))
        elif (
            parameters.ndim == 2 and parameters.shape[0] != self.num_qubits
        ) or parameters.ndim > 2:
            raise ValueError()
        self._parameters = parameters

        self.measurement_twirl = measurement_twirl
        """Whether twirling of the PVMs is enabled."""

        # NOTE: this public attribute inherits its docstring from the base class
        self.measurement_circuit = self._build_qc()

    def __repr__(self) -> str:
        """Return the string representation of a RandomizedProjectiveMeasurements instance."""
        return f"{self.__class__.__name__}(num_qubits={self.num_qubits}, parameters={self._parameters!r})"

    @override
    def definition(self) -> ProductPOVM:
        t1 = time.time()
        LOGGER.info("Building POVM definition")
        sq_povms = []
        for i in range(self.num_qubits):
            unitary = self._from_param(self._parameters[i])
            unitary[:, [1, 2]] = unitary[:, [2, 1]]
            unitary[[1, 2]] = unitary[[2, 1]]
            sq_povms.append(SingleQubitPOVM.from_vectors(unitary[:, 0::2].conj()))
        prod = ProductPOVM.from_list(sq_povms)

        t2 = time.time()
        LOGGER.info(f"Finished POVM definition. Took {t2 - t1:.6f}s")

        return prod

    def _build_qc(self) -> QuantumCircuit:
        """Build the quantum circuit that implements the measurement.

        In the case of randomized projective measurements (PMs), we choose for each shot a PM at
        random to perform the measurement. Any PM on single qubits can be described by two
        orthogonal projectors :math:``M_0 = |pi><pi|`` and :math:``M_1 = |pi_orth><pi_orth|``. The
        vector :math:``|pi> = U(theta, phi, 0) |0>`` can be defined by the first two usual Euler
        angles. The third Euler angles defines the global phase, which is irrelevant here. We then
        have :math:``|pi_orth> = U(theta, phi, 0) |1>`` up to another irrelevant global phase. To
        implement this measurement, we use the fact that :math:``p_i = Tr[rho M_i] = Tr[rho
        U|i><i|U_dag] = Tr[U_dag rho U |i><i|]``. In other words, we can first let the state evolve
        under :math:``U_dag`` and then perform a computational basis measurement. Note that we have
        :math:``U(theta, phi, lambda)_dag = U(-theta, -lambda, -phi)``.

        Returns:
            Parametrized quantum circuit that can implement any product of single-qubit projective
            measurements.
        """
        t1 = time.time()
        LOGGER.info("Building POVM circuit")

        qr = QuantumRegister(2 * self.num_qubits, name="povm_qr")
        cr = ClassicalRegister(2 * self.num_qubits, name=self.classical_register_name)
        qc = QuantumCircuit(qr, cr, name="measurement_circuit")
        for i in range(self.num_qubits):
            qc.unitary(self._from_param(self._parameters[i]), [i, i + self.num_qubits])

        qc.measure(qr, cr)

        t2 = time.time()
        LOGGER.info(f"Finished circuit construction. Took {t2 - t1:.6f}s")

        return qc

    def to_sampler_pub(
        self,
        circuit: QuantumCircuit,
        circuit_binding: BindingsArray,
        shots: int,
        *,
        pass_manager: StagedPassManager | None = None,
    ) -> tuple[SamplerPub, POVMMetadata]:
        """Append the measurement circuit(s) to the supplied circuit.

        This method takes a supplied circuit and appends the measurement circuit(s) to it. If the
        measurement circuit is parametrized, its parameters values should be concatenated with the
        parameter values associated with the supplied quantum circuit.

        .. warning::
           The actual number of measurements executed will depend not only on the provided ``shots``
           value but also on the value of :attr:`.shot_repetitions`.

        Args:
            circuit: A quantum circuit.
            circuit_binding: A bindings array.
            shots: A specific number of shots to run with.
            pass_manager: An optional transpilation pass manager. After the supplied circuit has
                been composed with the measurement circuit, the pass manager will be used to
                transpile the composed circuit.

        Returns:
            A tuple of a sampler pub and a dictionary of metadata which includes the
            :class:`.POVMImplementation` object itself. The metadata should contain all the
            information necessary to extract the POVM outcomes out of raw bitstrings.
        """
        t1 = time.time()
        LOGGER.info("Piecing together SamplerPub")

        composed_circuit = self.compose_circuits(circuit)

        if pass_manager is not None:
            composed_circuit = pass_manager.run(composed_circuit)

        pub = SamplerPub(
            circuit=composed_circuit,
            parameter_values=circuit_binding,
            shots=shots,
        )

        metadata = POVMMetadata(povm_implementation=self, composed_circuit=composed_circuit)

        t2 = time.time()
        LOGGER.info(f"Finished building SamplerPub. Took {t2 - t1:.6f}s")

        return (pub, metadata)

    @override
    def reshape_data_bin(self, data: DataBin) -> DataBin:
        return data

    def _povm_outcomes(
        self,
        bit_array: BitArray,
        povm_metadata: POVMMetadata,
        loc: int | tuple[int, ...] | None = None,
    ) -> list[tuple[int, ...]]:
        """TODO."""
        t1 = time.time()
        LOGGER.info("Creating POVM outcomes")

        # loc is assumed to have a length of at most pv.ndim = len(pv.shape)

        povm_outcomes = [
            self._get_outcome_label(bitstring_outcome=raw_bitstring)
            for raw_bitstring in bit_array.get_bitstrings(loc)
        ]

        t2 = time.time()
        LOGGER.info(f"Finished creating POVM outcomes. Took {t2 - t1:.6f}s")

        return povm_outcomes

    def _get_outcome_label(
        self,
        bitstring_outcome: str,
    ) -> tuple[int, ...]:
        """Transform a bitstring outcome to a POVM outcome.

        Args:
            bitstring_outcome: TODO.

        Returns:
            TODO.
        """
        return tuple(
            2 * int(bit_q) + int(bit_a)
            for bit_q, bit_a in zip(
                bitstring_outcome[: self.num_qubits - 1 : -1],
                bitstring_outcome[self.num_qubits - 1 :: -1],
            )
        )

    def _from_param(self, param: np.ndarray):
        """Initialize a POVM from the list of parameters."""
        n_out = 4
        u = np.zeros((n_out, n_out), dtype=complex)

        u[:, 0] = n_sphere(param[:3])
        u_gs = gs(u)  # Gram-Schmidt

        x = n_sphere(param[3:])
        # construct k'th vector of u
        for i in range(len(x) // 2):
            u[:, 1] += (x[2 * i] + x[2 * i + 1] * 1j) * u_gs[:, 1 + i]
        u_gs = gs(u)

        # for i in range(4) :
        #     u_gs[:,i] *= np.sign(u[0,i])*np.sign(u_gs[0,i])

        unitary = u_gs

        return unitary
