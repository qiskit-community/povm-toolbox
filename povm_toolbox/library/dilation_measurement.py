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
from povm_toolbox.utilities import gram_schmidt, n_sphere

from .metadata import POVMMetadata
from .povm_implementation import POVMImplementation

LOGGER = logging.getLogger(__name__)


class DilationMeasurements(POVMImplementation[POVMMetadata]):
    r"""A measurement leveraging Naimark's dilation theorem.

    IC dilation measurements are defined on a space spanning (at least) four states: e.g. :math:`\{
    |0\rangle,|1\rangle,|2\rangle,|3\rangle\}`. To achieve such a measurement using qubits, every
    qubit gets paired with an ancilla qubit. Then, the dilation measurement can be constructed via
    some two-qubit unitary followed by measurements in the computational basis. The binary outcomes
    :math:`\{|00\rangle,|01\rangle,|10\rangle,|11\rangle\}` can then be mapped to the four states
    above.

    There are 8 degrees of freedom in the two-qubit unitary specifying the dilation measurement (for
    each qubit). Different parametrizations of the unitaries are possible. Here, we use the same
    parametrization of the dilation POVM as the one presented in the work of G. García-Pérez, M. A.
    Rossi, B. Sokolov, F. Tacchino, P. K. Barkoutsos, G. Mazzola, I. Tavernelli, and S. Maniscalco,
    “*Learning to measure: adaptive informationally complete generalized measurements for quantum
    algorithms*”, PRX Quantum 2, Publisher: American Physical Society, 040342 (2021). Refer to this
    work for a detailed explanation of the parametrization.

    .. note::
        An additional ancilla qubit is required for each qubit in the system to be measured.
        Depending on the qubit connectivity the coupling of the measured qubit with its ancilla can
        introduce a significant overhead of SWAP gates.

    The example below shows how you construct a dilation POVM. It plots a visual representation of
    the POVM's definition to exemplify the different effects' directions.

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
        insert_barriers: bool = False,
    ) -> None:
        """Initialize a dilation POVM.

        Args:
            num_qubits: the number of qubits.
            parameters: can be either 1D or 2D. If 1D, it should be of length 8 and contain float
                values that specify the parametrization of the dilation POVM. If 2D, it will have
                a new set of parameters for each qubit. The 8 values fix all the degrees of freedom.
            measurement_layout: optional list of indices specifying on which qubits the POVM acts.
                See :attr:`.measurement_layout` for more details.
            insert_barriers: whether to insert a barrier between the composed circuits. This is not
                done by default but can prove useful when visualizing the composed circuit.

        Raises:
            ValueError: if the last dimension of ``parameters`` is not of length 8.
            ValueError: if the shape of ``parameters`` is not valid.
        """
        super().__init__(
            num_qubits, measurement_layout=measurement_layout, insert_barriers=insert_barriers
        )

        if parameters.shape[-1] != 8:
            raise ValueError(
                "The last dimension of ``parameters`` is expected to be of length 8, but has"
                f" length {parameters.shape[-1]} instead."
            )

        if parameters.ndim == 1:
            parameters = np.tile(parameters, (self.num_qubits, 1))
        elif (
            parameters.ndim == 2 and parameters.shape[0] != self.num_qubits
        ) or parameters.ndim != 2:
            raise ValueError(
                "``parameters`` is expected to have shape (8,) or (``num_qubits``, 8)"
                f" but has shape {parameters.shape} instead."
            )
        self._parameters = parameters

        # NOTE: this public attribute inherits its docstring from the base class
        self.measurement_circuit = self._build_qc()

    def __repr__(self) -> str:
        """Return the string representation of a DilationMeasurements instance."""
        return f"{self.__class__.__name__}(num_qubits={self.num_qubits}, parameters={self._parameters!r})"

    @override
    def definition(self) -> ProductPOVM:
        t1 = time.time()
        LOGGER.info("Building POVM definition")
        sq_povms = []
        for i in range(self.num_qubits):
            unitary = self._unitary_from_parameters(self._parameters[i])
            # In qiskit the order of qubits is reversed. Here we re-order it such that the
            # unitary is defined on the Hilbert space `H_q \otimes H_a`` (where "q" stands
            # for the system qubit and "a" for the ancilla qubit). In this way we can extract
            # the correct POVM effects.
            unitary[:, [1, 2]] = unitary[:, [2, 1]]
            unitary[[1, 2]] = unitary[[2, 1]]
            sq_povms.append(SingleQubitPOVM.from_vectors(unitary[:, 0::2].conj()))
        prod = ProductPOVM.from_list(sq_povms)

        t2 = time.time()
        LOGGER.info(f"Finished POVM definition. Took {t2 - t1:.6f}s")

        return prod

    def _build_qc(self) -> QuantumCircuit:
        """Build the quantum circuit that implements the measurement.

        In the case of dilation measurements, the circuit is fixed (not parametrized as
        for randomized measurements for instance). However, an additional ancilla qubit
        is required for each qubit that we want to measure.

        Returns:
            Quantum circuit that implements the dilation POVM defined by :attr:`._parameters`.
        """
        t1 = time.time()
        LOGGER.info("Building POVM circuit")

        qr = QuantumRegister(2 * self.num_qubits, name="povm_qr")
        cr = ClassicalRegister(2 * self.num_qubits, name=self.classical_register_name)
        qc = QuantumCircuit(qr, cr, name="measurement_circuit")
        for i in range(self.num_qubits):
            qc.unitary(self._unitary_from_parameters(self._parameters[i]), [i, i + self.num_qubits])

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

        This method takes a supplied circuit and appends the measurement circuit to it.
        An ancilla qubit is required for each qubit that we want to measure (specified by
        :attr:`.measurement_layout`). If the input circuit has one or more idling qubit(s),
        they will be used as measurement ancilla qubits. If more ancilla qubits are needed,
        those will be added to the input circuit (therefore increasing its size).

        .. warning::
           The number of qubits in the input circuit may be increased due to the ancilla
           qubits required for dilatation measurements.

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

    @override
    def _povm_outcomes(
        self,
        bit_array: BitArray,
        povm_metadata: POVMMetadata,
        *,
        loc: int | tuple[int, ...] | None = None,
    ) -> list[tuple[int, ...]]:
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
            bitstring_outcome: the raw outcome of the measurement. The order of
                qubit is assumed to be reversed. More specifically, the first
                :attr:`.num_qubits` bits will correspond to the ancilla qubit
                (in reverse order) and the last :attr:`.num_qubits` bits will
                correspond to the qubit to measure (in reverse order).

        Returns:
            A tuple of indices indicating the POVM outcomes on each qubit. For each qubit,
            the index goes from :math:``0`` to :math:``3``.
        """
        return tuple(
            2 * int(bit_q) + int(bit_a)
            for bit_q, bit_a in zip(
                bitstring_outcome[: self.num_qubits - 1 : -1],
                bitstring_outcome[self.num_qubits - 1 :: -1],
            )
        )

    def _unitary_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """Construct the unitary defining the dilation POVM from parameters.

        Args:
            parameters: 1D array of length 8. It should contains float values
                that specify the parametrization of the unitary.

        Returns:
            The resulting untiary.
        """
        num_outcomes = 4
        u = np.zeros((num_outcomes, num_outcomes), dtype=complex)

        # construct the first column of the unitary
        u[:, 0] = n_sphere(parameters[:3])
        u_gs = gram_schmidt(u)

        x = n_sphere(parameters[3:])
        # construct the second column of the unitary
        for i in range(len(x) // 2):
            u[:, 1] += (x[2 * i] + x[2 * i + 1] * 1j) * u_gs[:, 1 + i]
        u_gs = gram_schmidt(u)

        unitary = u_gs

        return unitary
