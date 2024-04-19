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

from collections import Counter
from dataclasses import dataclass

import numpy as np
from qiskit.circuit import ClassicalRegister, ParameterVector, QuantumCircuit, QuantumRegister
from qiskit.primitives.containers import DataBin
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.bit_array import BitArray
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.transpiler import StagedPassManager

from povm_toolbox.quantum_info.product_povm import ProductPOVM
from povm_toolbox.quantum_info.single_qubit_povm import SingleQubitPOVM

from .povm_implementation import POVMImplementation, POVMMetadata


@dataclass
class RandomizedPMsMetadata(POVMMetadata):
    """TODO."""

    pvm_keys: list[tuple[int, ...]]


class RandomizedPMs(POVMImplementation[RandomizedPMsMetadata]):
    """Class to represent the implementation of randomized projective measurements."""

    def __init__(
        self,
        n_qubit: int,
        bias: np.ndarray,
        angles: np.ndarray,
        shot_batch_size: int = 1,
    ) -> None:
        """Implement a product POVM through the randomization of single-qubit projective measurement.

        If we extend this interface to support different number of effects for each qubit in the
        future, we may need to move away from np.ndarray input types to sequences of sequences.

        Args:
            n_qubits: the number of qubits.
            bias: can be either 1D or 2D. If 1D, it should contain float values indicating the bias
                for measuring in each of the PVMs. I.e., its length equals the number of PVMs.
                These floats should sum to 1. If 2D, it will have a new set of biases for each
                qubit.
            angles: can be either 1D or 2D. If 1D, it should contain float values to indicate the
                different angles of each effect. I.e. its length equals two times the number of
                PVMs (since we have 2 angles per PVMs). If 2D, it will have a new set of angles
                for each qubit.
            shot_batch_size: number of shots assigned to each sampled PVM. If set to 1, a new PVM
                is sampled for each shot.

        Raises:
            ValueError: TODO.
        """
        super().__init__(n_qubit)

        if 2 * bias.shape[-1] != angles.shape[-1]:
            raise ValueError(f"TODO: error message. {bias.shape[-1]}, {angles.shape[-1]}.")
        self._n_PVMs = bias.shape[-1]

        if bias.ndim == 1:
            bias = np.tile(bias, (self.n_qubit, 1))
        elif (bias.ndim == 2 and bias.shape[0] != self.n_qubit) or bias.ndim > 2:
            raise ValueError("TODO: error message.")
        if np.any(bias < 0.0):
            raise ValueError(
                "There should not be any negative values in the probability distribution parameters."
            )
        if not np.allclose(np.sum(bias, axis=-1), 1.0):
            raise ValueError("The probability distribution parameters should sum up to one.")
        self.bias = bias

        if angles.ndim == 1:
            angles = np.tile(angles, (self.n_qubit, 1))
        elif (angles.ndim == 2 and angles.shape[0] != self.n_qubit) or angles.ndim > 2:
            raise ValueError("TODO: error message.")
        self.angles = angles.reshape((self.n_qubit, self._n_PVMs, 2))

        self.msmt_qc = self._build_qc()

        self.shot_batch_size = shot_batch_size

    def _build_qc(self) -> QuantumCircuit:
        """Build the quantum circuit that implements the measurement.

        In the case of randomized projective measurements (PMs), we choose for each shot a PM at random to perform the
        measurement. Any PM on single qubits can be described by two orthogonal projectors :math:``M_0 = |pi><pi|``
        and :math:``M_1 = |pi_orth><pi_orth|``. The vector :math:``|pi> = U(theta, phi, 0) |0>`` can be defined by the
        first two usual Euler angles. The third Euler angles defines the global phase, which is irrelevant here.
        We then have :math:``|pi_orth> = U(theta, phi, 0) |1>`` up to another irrelvant global phase. To implement
        this measurement, we use the fact that :math:``p_i = Tr[rho M_i] = Tr[rho U|i><i|U_dag] = Tr[U_dag rho U |i><i|]``.
        In other words, we can first let the state evolve under :math:``U_dag`` ands then perform a computational basis
        measurement. Note that we have :math:``U(theta, phi, lambda)_dag = U(-theta, -lambda, -phi)``.

        Returns:
            Paramnetrized quantum circuit that can implement any product of single-qubit projective measurements.
        """
        theta = ParameterVector("theta", length=self.n_qubit)
        phi = ParameterVector("phi", length=self.n_qubit)

        qr = QuantumRegister(self.n_qubit, name="povm_qr")
        cr = ClassicalRegister(self.n_qubit, name=self.classical_register_name)
        qc = QuantumCircuit(qr, cr, name="msmt_qc")
        for i in range(self.n_qubit):
            # We apply ``U_dag``, where ``U`` is the unitary operation to go from the computational basis
            # to the new measurement basis:
            #   qc.u(theta=theta[i], phi=phi[i], lam=0.0, qubit=i).inverse()
            # which is equivalent to :
            #   qc.u(theta=-theta[i], phi=0.0, lam=-phi[i], qubit=i)
            # which can be decomposed into basis gates as:
            qc.rz(-phi[i], qubit=i)
            qc.sx(qubit=i)
            qc.rz(np.pi - theta[i], qubit=i)
            qc.sx(qubit=i)
            qc.rz(3 * np.pi, qubit=i)

        qc.measure(qr, cr)

        return qc

    def to_sampler_pub(
        self,
        circuit: QuantumCircuit,
        circuit_binding: BindingsArray,
        shots: int,
        pass_manager: StagedPassManager,
    ) -> tuple[SamplerPub, RandomizedPMsMetadata]:
        """Append the measurement circuit(s) to the supplied circuit.

        This method takes a supplied circuit and append the measurement circuit
        to it. As the measurement circuit is parametrized, its parameters values
        are concatenated with the parameter values associated with the supplied
        quantum circuit. TODO: explain how the distribution of the shots is done.
        If a randomized POVM is used, the enduser's parameters have to be
        concantenated with the sampled POVM parameters. The POVM parametersare a
        2-D (TODO: update docstrings in next PR, which will change this method anyways)

        Args:
            circuit: A quantum circuit.
            circuit_binding: A bindings array.
            shots: A specific number of shots to run with.

        Returns:
            A tuple of a sampler pub and a dictionnary of metadata which include
            the ``POVMImplementation`` object itself. The metadata should contain
            all the information neceassary to extract the POVM outcomes out of raw
            bitstrings. (TODO: explain what is it exactly)

        Raises:
            ValueError: If the number of shots is not compatible with the batch size.
                It should be a multiple of the batch size.
        """
        if shots % self.shot_batch_size != 0:
            raise ValueError(
                f"The number of shots ({shots}) is not a multiple of "
                f"the batch size ({self.shot_batch_size})."
            )

        num_batches = shots // self.shot_batch_size

        # distribute the shots
        pvm_idx = np.zeros((num_batches, self.n_qubit), dtype=int)
        for i in range(self.n_qubit):
            pvm_idx[:, i] = np.random.choice(
                self._n_PVMs, size=num_batches, replace=True, p=self.bias[i]
            )

        # retrieve the actual parameters from the pvm labels
        pvm_parameters = np.empty((num_batches, 2 * self.n_qubit))
        for i in range(num_batches):
            pvm_parameters[i] = self._get_pvm_parameter(tuple(pvm_idx[i]))
        measurement_binding = BindingsArray.coerce({tuple(self.msmt_qc.parameters): pvm_parameters})
        # TODO : make _get_pvm_parameter return directly a bindingsArray ?

        # We combine the parameter values from the supplied circuit and from the
        # the measurement circuit.
        binding_data = {}

        # We tile the circuit parameter values such that it is duplicated for each measurement shot.
        # E.g., if the supplied circuit has 3 parameters and 5 different set of values are supplied,
        # the corresponding `BindingsArray` has :
        #   .shape = (5,)
        #   .num_parameters = 3
        # Now if the POVM measurement circuit has 2*n_qubit parameters and a set of values is fed for
        # each batch, the corresponding `BindingsArray` has :
        #   .shape = (num_batches,)
        #   .num_parameters = 2*n_qubit
        # Then, the combined `BindingsArray` should have :
        #   .shape = (5, num_batches)
        #   .num_parameters = 3 + 2*n_qubit
        # The data is stored as a dictionary of arrays where each array has a shape such that :
        #   - the last dismension corresponds to the number of parameters stored in this entry
        #     of the dictionnary
        #   - the leading shape corresponds to the different sets of parameter values and is shared
        #     amongst all dictionnary entries (it is the `.shape` of the `BindingsArray`)
        # We loop over the circuit parameter values :
        for circuit_param, circuit_val in circuit_binding.data.items():
            # For each array we insert a dimension on the second to last axis and duplicate `num_batches` times
            # the circuit values over this axis. The resulting np.ndarray shape is (5, num_batches, num_param_of_entry)
            # where num_param_of_entry = circuit_val.shape[-1] is the number of parameters stored in this dictionnary entry.
            # The general shape of the resulting np.ndarray is (*circuit_val.shape[:-1], num_batches, circuit_val.shape[-1]).
            binding_data[circuit_param] = np.tile(circuit_val[..., np.newaxis, :], (num_batches, 1))

        # TODO : Distribution of random PVMs is exactly the same for all set of circuit parameter values,
        # turn this into an option (either same distribution or resample PVMs for each parameter value set)

        # We loop over the measurement parameter values :
        for povm_param, povm_val in measurement_binding.data.items():
            # For each array of POVM parameters we duplicate the array, as many times as there are
            # different sets of circuit parameter values. The tiling is done on the (newly inserted)
            # leading dimension(s) of the array.
            # The shape of the resulting np.ndarray is (*circuit_binding.shape, num_batches, num_povm_parameters).
            binding_data[povm_param] = np.tile(povm_val, (*circuit_binding.shape, 1, 1))

        combined_binding = BindingsArray.coerce(binding_data)

        # TODO: assert circuit qubit routing and stuff
        # TODO: assert both circuits are compatible, in particular no measurements at the end of ``circuits``
        # TODO: how to compose classical registers ? CR used for POVM measurements should remain separate
        # TODO: how to deal with transpilation ?

        composed_circuit = circuit.compose(self.msmt_qc)
        composed_isa_circuit = pass_manager.run(composed_circuit)

        pub = SamplerPub(
            circuit=composed_isa_circuit,
            parameter_values=combined_binding,
            shots=self.shot_batch_size,
        )

        metadata = RandomizedPMsMetadata(povm=self, pvm_keys=list(map(tuple, pvm_idx)))

        return (pub, metadata)

    def reshape_data_bin(self, data: DataBin) -> DataBin:
        """TODO."""
        # We extract the raw ``BitArray``
        raw_bit_array = self._extract_bitarray(data)
        # Next we reshape the array such that the number of shots is correct.
        # For RandomizedPMs, the raw `BitArray` has the following properties :
        #   .shape == (*pub.parameter_values.shape, pub.shots/n)
        #   .num_shots == batch_size
        # -> the internal numpy array has shape (*pub.parameter_values.shape, pub.shots/batch_size, batch_size, num_bits)
        # where `pub` is the corresponding `POVMSamplerPub` supplied to the `run`
        # method. We now reshape the raw `BitArray` such that :
        #   .shape == pub.parameter_values.shape
        #   .num_shots == pub.shots
        # -> internal array with shape (*pub.parameter_values.shape, pub.shots, num_bits)
        # `BitArray.reshape` method does not handle the n=1 case properly,
        # so we have to do it "manually":
        shape = raw_bit_array.array.shape
        new_shape = (*shape[:-3], -1, shape[-1])
        bit_array = BitArray(
            array=raw_bit_array.array.reshape(new_shape), num_bits=raw_bit_array.num_bits
        )
        return data.__class__(**{self.classical_register_name: bit_array})

    def _counter(
        self,
        bit_array: BitArray,
        povm_metadata: RandomizedPMsMetadata,
        loc: int | tuple[int, ...] | None = None,
    ) -> Counter:
        """TODO."""
        try:
            pvm_keys = povm_metadata.pvm_keys
        except KeyError as exc:
            raise KeyError(
                "The metadata of povm sampler result associated with a "
                "RandomizedPMs POVM should specify a list of pvm keys, "
                "but none were found."
            ) from exc

        # TODO : improve performance. Currently we loop over all shots and get the
        # outcome label each time. There's probably a way to group equivalent outcomes
        # earlier or do it in a smarter way.

        povm_outcomes = []
        for i, raw_bitstring in enumerate(bit_array.get_bitstrings(loc)):
            povm_outcomes.append(
                self._get_outcome_label(
                    pvm_idx=pvm_keys[i // self.shot_batch_size], bitstring_outcome=raw_bitstring
                )
            )

        return Counter(povm_outcomes)

    def _get_pvm_parameter(self, pvm_idx: tuple[int, ...]) -> np.ndarray:
        """Return the concrete parameter values associated to a PVM label.

        Args:
            pvm_idx: qubit-wise index indicating which PVM was used to perform the measurement.

        Returns:
            Parameter values for the specified PVM.
        """
        param: np.ndarray = np.zeros((2, self.n_qubit))

        for i in range(self.n_qubit):
            # Axes ordering and reverse indexing because of the alphabetical order of the parameters
            # when submitting pubs to the sampler.
            # TODO: find a cleaner and more general way to deal with this
            param[0, i] = self.angles[i, pvm_idx[i], 1]
            param[1, i] = self.angles[i, pvm_idx[i], 0]

        return param.flatten()

    def _get_outcome_label(
        self, pvm_idx: tuple[int, ...], bitstring_outcome: str
    ) -> tuple[int, ...]:
        """Transform a PVM index and a bitstring outcome to a POVM outcome.

        Args:
            pvm_idx: qubit-wise index indicating which PVM was used to perform the measurement.
            bitstring_outcomes: the outcome of the measurements performed with the PVM label
                by ``pvm_idx``. The order of qubit is assumed to be reversed.

        Returns:
            A tuple of indices indicating the POVM outcomes on each qubit. For each qubit,
            the index goes from :math:``0`` to :math:``2 * self.n_PVM - 1``.
        """
        return tuple(pvm_idx[i] * 2 + int(bit) for i, bit in enumerate(bitstring_outcome[::-1]))

    # TODO: find a better name
    def to_povm(self) -> ProductPOVM:
        """Return the POVM corresponding to this implementation."""
        stabilizers: np.ndarray = np.zeros((self.n_qubit, self._n_PVMs, 2, 2), dtype=complex)

        stabilizers[:, :, 0, 0] = np.cos(self.angles[:, :, 0] / 2.0)
        stabilizers[:, :, 0, 1] = (
            np.cos(self.angles[:, :, 1]) + 1.0j * np.sin(self.angles[:, :, 1])
        ) * np.sin(self.angles[:, :, 0] / 2.0)
        stabilizers[:, :, 1, 0] = -stabilizers[
            :, :, 0, 1
        ].conjugate()  # up to irrelevant global phase e^(i phi)
        stabilizers[:, :, 1, 1] = stabilizers[:, :, 0, 0]  # up to irrelevant global phase e^(i phi)

        stabilizers = np.multiply(stabilizers.T, np.sqrt(self.bias).T).T
        stabilizers = stabilizers.reshape((self.n_qubit, 2 * self._n_PVMs, 2))

        sq_povms = []
        for vecs in stabilizers:
            sq_povms.append(SingleQubitPOVM.from_vectors(vecs))

        return ProductPOVM.from_list(sq_povms)


class LocallyBiased(RandomizedPMs):
    """TODO."""

    def __init__(
        self,
        n_qubit: int,
        bias: np.ndarray,
    ):
        """Construct a locally-biased classical shadow POVM.

        TODO: The same as above, but the angles are hard-coded to be X/Y/Z for all qubits.
        """
        angles = np.array([0.0, 0.0, 0.5 * np.pi, 0.0, 0.5 * np.pi, 0.5 * np.pi])
        assert bias.shape[-1] == 3
        super().__init__(n_qubit=n_qubit, bias=bias, angles=angles)


class ClassicalShadows(LocallyBiased):
    """TODO."""

    def __init__(
        self,
        n_qubit: int,
    ):
        """Construct a classical shadow POVM.

        TODO: The same as above, but also hard-coding the biases to be equally distributed.
        """
        bias = 1.0 / 3.0 * np.ones(3)
        super().__init__(n_qubit=n_qubit, bias=bias)
