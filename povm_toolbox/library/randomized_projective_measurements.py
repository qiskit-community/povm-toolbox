# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""RandomizedProjectiveMeasurements."""

from __future__ import annotations

import logging
import sys
import time

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override  # pragma: no cover

import numpy as np
from numpy.random import Generator, default_rng
from qiskit.circuit import ClassicalRegister, ParameterVector, QuantumCircuit, QuantumRegister
from qiskit.primitives.containers import DataBin, make_data_bin
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.bit_array import BitArray
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.transpiler import StagedPassManager

from povm_toolbox.quantum_info import ProductPOVM, SingleQubitPOVM

from .metadata import RPMMetadata
from .povm_implementation import POVMImplementation

LOGGER = logging.getLogger(__name__)


class RandomizedProjectiveMeasurements(POVMImplementation[RPMMetadata]):
    """A general randomized projective measurements POVM.

    The example below shows how you construct a RPM POVM. It plots a visual representation of the
    POVM's definition to exemplify the different randomization for each qubit.

    .. plot::
       :include-source:

       >>> import numpy as np
       >>> from povm_toolbox.library import RandomizedProjectiveMeasurements
       >>> povm = RandomizedProjectiveMeasurements(
       ...     2,
       ...     bias=np.array([[0.1, 0.6, 0.3], [0.5, 0.25, 0.25]]),
       ...     angles=np.array([
       ...         [np.pi/6, 5*np.pi/6, -np.pi/4, -np.pi/2, -np.pi/2, np.pi/4],
       ...         [np.pi/3, np.pi/3, -np.pi/3, np.pi/3, np.pi/3, -np.pi/3],
       ...     ]),
       ... )
       >>> print(povm)
       RandomizedProjectiveMeasurements(num_qubits=2, bias=array([[0.1 , 0.6 , 0.3 ], [0.5 , 0.25, 0.25]]),
           angles=array([[[ 0.52359878,  2.61799388],
               [-0.78539816, -1.57079633],
               [-1.57079633,  0.78539816]],
              [[ 1.04719755,  1.04719755],
               [-1.04719755,  1.04719755],
               [ 1.04719755, -1.04719755]]]))
       >>> povm.definition().draw_bloch()
       <Figure size 1000x500 with 2 Axes>
    """

    def __init__(
        self,
        num_qubits: int,
        bias: np.ndarray,
        angles: np.ndarray,
        *,
        measurement_layout: list[int] | None = None,  # TODO: add | Layout
        measurement_twirl: bool = False,
        shot_repetitions: int = 1,
        insert_barriers: bool = False,
        seed: int | Generator | None = None,
    ) -> None:
        # NOTE: If we extend this interface to support different number of effects for each qubit in
        # the future, we may need to move away from np.ndarray input types to sequences of sequences.
        """Initialize a randomized projective measurements POVM.

        Args:
            num_qubits: the number of qubits.
            bias: can be either 1D or 2D. If 1D, it should contain float values indicating the bias
                for measuring in each of the PVMs. I.e., its length equals the number of PVMs.
                These floats should sum to 1. If 2D, it will have a new set of biases for each
                qubit.
            angles: can be either 1D or 2D. If 1D, it should be a flatten array containing float
                values to indicate the different angles of each PVM. I.e. its length equals two
                times the number of PVMs (since we have 2 angles per PVM). If 2D, it will have a new
                set of angles for each qubit. The angles are expected to be pairs of angles
                ``(theta, phi)`` for each PVM and correspond to the parameters of the
                :class:`~.qiskit.circuit.library.UGate` instance used to rotate the canonical
                Z-measurement into an arbitrary projective measurement. Note that this differs from
                the angles expected during the initialization of a
                :class:`.MutuallyUnbiasedBasesMeasurements` instance, where a unique triplet of
                angles ``(theta, phi, lam)`` is expected for each qubit.
            measurement_layout: optional list of indices specifying on which qubits the POVM acts.
                See :attr:`.measurement_layout` for more details.
            measurement_twirl: whether to randomly twirl the measurements. For each single-qubit
                projective measurement, random twirling is equivalent to randomly flipping the
                measurement. This is equivalent to randomly taking the opposite Bloch vector in the
                Bloch sphere representation.
            shot_repetitions: number of times the measurement is repeated for each sampled PVM. More
                precisely, a new PVM is sampled for all ``shots`` (i.e. the number of times as
                specified by the user via the ``shots`` argument of the method
                :meth:`.POVMSampler.run`). Then, the parameter ``shot_repetitions`` states how many
                times we repeat the measurement for each sampled PVM (default is 1). Therefore, the
                effective total number of measurement shots is ``shots`` multiplied by
                ``shot_repetitions``.
            insert_barriers: whether to insert a barrier between the composed circuits. This is not
                done by default but can prove useful when visualizing the composed circuit.
            seed: optional seed to fix the :class:`numpy.random.Generator` used to sample PVMs.
                The PVMs are sampled according to the probability distribution(s) specified by
                ``bias``. The user can also directly provide a random generator. If ``None``, a
                random seed will be used.

        Raises:
            ValueError: If the shape of ``bias`` is not compatible with the shape of ``angles``.
            ValueError: If the shape of ``bias`` is not compatible with ``num_qubits``.
            ValueError: If there is a negative value in the probability distribution(s) specified
                by ``bias``.
            ValueError: If the probability distribution(s) specified by ``bias`` don't sum up to 1.
            ValueError: If the shape of ``angles`` is not compatible with ``num_qubits``.
            TypeError: If the type of ``seed`` is not valid.
        """
        super().__init__(
            num_qubits, measurement_layout=measurement_layout, insert_barriers=insert_barriers
        )

        if 2 * bias.shape[-1] != angles.shape[-1]:
            raise ValueError(
                f"The shape of ``bias`` ({bias.shape}) is not compatible with the shape"
                f" of ``angles`` ({angles.shape})."
            )
        self._num_PVMs = bias.shape[-1]

        if bias.ndim == 1:
            bias = np.tile(bias, (self.num_qubits, 1))
        elif (bias.ndim == 2 and bias.shape[0] != self.num_qubits) or bias.ndim > 2:
            raise ValueError(
                f"The shape of ``bias`` ({bias.shape}) is not compatible with"
                f" ``num_qubits`` ({num_qubits})."
            )
        if np.any(bias < 0.0):
            raise ValueError(
                "There should not be any negative values in the probability distribution parameters."
            )
        if not np.allclose(np.sum(bias, axis=-1), 1.0):
            raise ValueError("The probability distribution parameters should sum up to one.")

        self.bias = bias
        """The sampling bias for each PVM per qubit."""

        if angles.ndim == 1:
            angles = np.tile(angles, (self.num_qubits, 1))
        elif (angles.ndim == 2 and angles.shape[0] != self.num_qubits) or angles.ndim > 2:
            raise ValueError(
                f"The shape of ``angles`` ({angles.shape}) is not compatible with"
                f" ``num_qubits`` ({num_qubits})."
            )
        self.angles = angles.reshape((self.num_qubits, self._num_PVMs, 2))
        """The angles defining each PVM. These are stored as pairs of ``(theta, phi)`` and
        correspond to the parameters of the :class:`~.qiskit.circuit.library.UGate` instance used to
        rotate the canonical Z-measurement into an arbitrary projective measurement."""

        self.measurement_twirl = measurement_twirl
        """Whether twirling of the PVMs is enabled."""

        # NOTE: this public attribute inherits its docstring from the base class
        self.measurement_circuit = self._build_qc()

        self.shot_repetitions = shot_repetitions
        """The number of times the measurement is repeated for each sampled PVM. More precisely, a
        new PVM is sampled for all ``shots`` (i.e. the number of times as specified by the user via
        the ``shots`` argument of the method :meth:`.POVMSampler.run`). Then, this attribute states
        how many times we repeat the measurement for each sampled PVM (default is 1). Therefore, the
        effective total number of measurement shots is ``shots`` multiplied by ``shot_repetitions``.
        """

        self._rng: Generator
        if seed is None:
            self._rng = default_rng()
        elif isinstance(seed, int):
            self._rng = default_rng(seed)
        elif isinstance(seed, Generator):
            self._rng = seed
        else:
            raise TypeError(f"The type of `seed` ({type(seed)}) is not valid.")

    def __repr__(self) -> str:
        """Return the string representation of a RandomizedProjectiveMeasurements instance."""
        return (
            f"{self.__class__.__name__}(num_qubits={self.num_qubits}, bias={self.bias!r}, "
            f"angles={self.angles!r})"
        )

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

        self._qc_theta = ParameterVector("theta_measurement", length=self.num_qubits)
        self._qc_phi = ParameterVector("phi_measurement", length=self.num_qubits)

        qr = QuantumRegister(self.num_qubits, name="povm_qr")
        cr = ClassicalRegister(self.num_qubits, name=self.classical_register_name)
        qc = QuantumCircuit(qr, cr, name="measurement_circuit")
        for i in range(self.num_qubits):
            # We apply ``U_dag``, where ``U`` is the unitary operation to go from the computational basis
            # to the new measurement basis:
            #   qc.u(theta=theta[i], phi=phi[i], lam=0.0, qubit=i).inverse()
            # which is equivalent to :
            #   qc.u(theta=-theta[i], phi=0.0, lam=-phi[i], qubit=i)
            # which can be decomposed (up to an irrelevant global phase) into basis gates as:
            qc.rz(-self._qc_phi[i], qubit=i)
            qc.sx(qubit=i)
            qc.rz(np.pi - self._qc_theta[i], qubit=i)
            qc.sx(qubit=i)

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
    ) -> tuple[SamplerPub, RPMMetadata]:
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

        # We combine the parameter values from the supplied circuit and from the
        # the measurement circuit.
        binding_data = {}

        # We tile the circuit parameter values such that it is duplicated for each PVM sampled.

        # E.g., if the supplied circuit has 3 parameters and 5 different set of values are supplied,
        # the corresponding `BindingsArray` has :
        #   .shape = (5,)
        #   .num_parameters = 3
        # Now if the POVM measurement circuit has 2*num_qubits parameters and a set of values is fed for
        # each "shot", the corresponding `BindingsArray` has :
        #   .shape = (shots,)
        #   .num_parameters = 2*num_qubits
        # Then, the combined `BindingsArray` should have :
        #   .shape = (5, shots)
        #   .num_parameters = 3 + 2*num_qubits
        # The data is stored as a dictionary of arrays where each array has a shape such that :
        #   - the last dimension corresponds to the number of parameters stored in this entry
        #     of the dictionary
        #   - the leading shape corresponds to the different sets of parameter values and is shared
        #     amongst all dictionary entries (it is the `.shape` of the `BindingsArray`)
        # We loop over the circuit parameter values :
        for circuit_param, circuit_val in circuit_binding.data.items():
            # For each array we insert a dimension on the second to last axis and duplicate ``shots`` times
            # the circuit values over this axis. The resulting np.ndarray shape is (5, shots, num_param_of_entry)
            # where num_param_of_entry = circuit_val.shape[-1] is the number of parameters stored in this dictionary entry.
            # The general shape of the resulting np.ndarray is (*circuit_val.shape[:-1], shots, circuit_val.shape[-1]).
            binding_data[circuit_param] = np.tile(circuit_val[..., np.newaxis, :], (shots, 1))

        # We create the array that will store the qubit-wise indices of all the sampled PVMs.
        pvm_idx = self._sample_pvm_idxs(circuit_binding, shots)

        # Transform the PVM indices into actual measurement parameters that are
        # then coerced into a :class:``BindingsArray`` object.
        measurement_binding = self._get_pvm_bindings_array(pvm_idx)
        binding_data.update(measurement_binding.data)

        combined_binding = BindingsArray.coerce(binding_data)

        composed_circuit = self.compose_circuits(circuit)

        if pass_manager is not None:
            composed_circuit = pass_manager.run(composed_circuit)

        pub = SamplerPub(
            circuit=composed_circuit,
            parameter_values=combined_binding,
            shots=self.shot_repetitions,
        )

        metadata = RPMMetadata(
            povm_implementation=self, composed_circuit=composed_circuit, pvm_keys=pvm_idx
        )

        t2 = time.time()
        LOGGER.info(f"Finished building SamplerPub. Took {t2 - t1:.6f}s")

        return (pub, metadata)

    @override
    def reshape_data_bin(self, data: DataBin) -> DataBin:
        t1 = time.time()
        LOGGER.info("Reshaping result DataBin")

        # We extract the raw ``BitArray``
        raw_bit_array = self._get_bitarray(data)
        # Next we reshape the array such that the actual number of shots is correct.
        # For RandomizedProjectiveMeasurements (rpm for short), the raw `BitArray`
        # has the following properties :
        #   .shape == (*pub.parameter_values.shape, povm_sampler_pub.shots)
        #   .num_shots == rpm.shot_repetitions
        # -> the internal numpy array has shape (*pub.parameter_values.shape, povm_sampler_pub.shots, rpm.shot_repetitions, num_bits)
        # where `pub` is the corresponding `POVMSamplerPub` supplied to the `run`
        # method. We now reshape the raw `BitArray` such that :
        #   .shape == pub.parameter_values.shape
        #   .num_shots == povm_sampler_pub.shots * rpm.shot_repetitions
        # -> internal array with shape (*pub.parameter_values.shape, povm_sampler_pub.shots*rpm.shot_repetitions, num_bits)
        # `BitArray.reshape` method does not handle the n=1 case properly,
        # so we have to do it "manually":
        shape = raw_bit_array.array.shape
        new_shape = (*shape[:-3], -1, shape[-1])
        bit_array = BitArray(
            array=raw_bit_array.array.reshape(new_shape), num_bits=raw_bit_array.num_bits
        )
        data_bin_cls = make_data_bin(
            [(self.classical_register_name, BitArray)],
            shape=bit_array.shape,
        )
        data_bin = data_bin_cls(**{self.classical_register_name: bit_array})

        t2 = time.time()
        LOGGER.info(f"Finished reshaping DataBin. Took {t2 - t1:.6f}s")

        return data_bin

    def _povm_outcomes(
        self,
        bit_array: BitArray,
        povm_metadata: RPMMetadata,
        *,
        loc: int | tuple[int, ...] | None = None,
    ) -> list[tuple[int, ...]]:
        t1 = time.time()
        LOGGER.info("Creating POVM outcomes")

        # povm_metadata.pvm_keys.shape is (*pv.shape, povm_sampler_pub.shots, num_qubits)
        # and bit_array.num_shots is povm_sampler_pub.shots*self.shot_repetitions
        # loc is assumed to have a length of at most pv.ndim = len(pv.shape)

        try:
            pvm_keys = povm_metadata.pvm_keys if loc is None else povm_metadata.pvm_keys[loc]
        except AttributeError as exc:
            raise AttributeError(
                "The metadata of povm sampler result associated with a "
                "RandomizedPMs POVM should specify a list of pvm keys, "
                "but none were found."
            ) from exc
        if pvm_keys.shape != (bit_array.num_shots // self.shot_repetitions, self.num_qubits):
            raise ValueError(
                "Either the shape of the `BitArray` is not compatible with the"
                " shape of the PVM keys stored in the metadata, or the `loc`"
                " argument is not valid."
            )

        # TODO : improve performance. Currently we loop over all shots and get the
        # outcome label each time. There's probably a way to group equivalent outcomes
        # earlier or do it in a smarter way.

        povm_outcomes = []
        for i, raw_bitstring in enumerate(bit_array.get_bitstrings(loc)):
            povm_outcomes.append(
                self._get_outcome_label(
                    pvm_idx=pvm_keys[i // self.shot_repetitions], bitstring_outcome=raw_bitstring
                )
            )

        t2 = time.time()
        LOGGER.info(f"Finished creating POVM outcomes. Took {t2 - t1:.6f}s")

        return povm_outcomes

    def _sample_pvm_idxs(self, circuit_binding: BindingsArray, shots: int) -> np.ndarray:
        """Sample the qubit-wise indices of PVMs to use for all shots.

        Args:
            circuit_binding: A bindings array.
            shots: A specific number of shots to run with.

        Returns:
            The sampled PVM indices.
        """
        # We create the array that will store the qubit-wise indices of all the sampled PVMs.
        pvm_idx = np.zeros((*circuit_binding.shape, shots, self.num_qubits), dtype=int)
        # We loop over the different qubits :
        for i in range(self.num_qubits):
            # For each qubit, we sample PVMs according to the local bias defined on
            # this particular qubit. We draw a PVM for each "shot" and for
            # each set of circuit parameter values supplied by the user through the
            # :method:``POVMSampler.run`` method.
            pvm_idx[..., i] = self._rng.choice(
                self._num_PVMs,
                size=circuit_binding.size * shots,
                replace=True,
                p=self.bias[i],
            ).reshape(  # Reshape to match the shape of ``pvm_idx``.
                (*circuit_binding.shape, shots)
            )
            # If the twirling option is turned on we double the number of PVMs
            # because each PVM can be twirled. The encoding works as follows :
            #   `pvm_idx % self._num_PVMs` is the index of the PVM used and
            #   `pvm_idx // self._num_PVMs` indicates if the PVM has been twirled.
            # For the example of :class:`ClassicalShadows`, we have:
            #   num_PVMs = 3
            #   pvm_idx == 0 -> untwirled Z-measurement: {|0><0|, |1><1|}
            #   pvm_idx == 1 -> untwirled X-measurement: {|+><+|, |-><-|}
            #   pvm_idx == 2 -> untwirled Y-measurement: {|+i><+i|, |-i><-i|}
            #   pvm_idx == 3 -> twirled Z-measurement: {|1><1|, |0><0|}
            #   pvm_idx == 4 -> twirled X-measurement: {|-><-|, |+><+|}
            #   pvm_idx == 5 -> twirled Y-measurement: {|-i><-i|, |+i><+i|}
            if self.measurement_twirl:
                pvm_idx[..., i] += self._num_PVMs * self._rng.integers(
                    2,
                    size=circuit_binding.size * shots,
                ).reshape(  # Reshape to match the shape of ``pvm_idx``.
                    (*circuit_binding.shape, shots)
                )

        return pvm_idx

    def _get_pvm_bindings_array(self, pvm_idx: np.ndarray) -> BindingsArray:
        """Return the concrete parameter values associated to a PVM label.

        Args:
            pvm_idx: an array of integers with shape ``(*pv, povm_sampler_pub.shots, num_qubits)``.

        Returns:
            Parameter values for the specified PVM.
        """
        t1 = time.time()
        LOGGER.info("Building PVM bindings array")

        # shape is assumed to be (*pv, povm_sampler_pub.shots, num_qubits)
        if pvm_idx.shape[-1] != self.num_qubits:
            raise ValueError(
                "The shape ``pvm_idx`` is expected to be ``(..., num_qubits="
                f"{self.num_qubits})``, but got {pvm_idx.shape}."
            )
        theta: np.ndarray = np.empty(pvm_idx.shape)
        phi: np.ndarray = np.empty(pvm_idx.shape)
        for multi_index in np.ndindex(pvm_idx.shape):
            # multi_index.shape is (*pv, povm_sampler_pub.shots, num_qubits)
            i_qubit = multi_index[-1]
            actual_pvm_idx = pvm_idx[multi_index] % self._num_PVMs
            twirl = pvm_idx[multi_index] // self._num_PVMs

            theta[multi_index] = self.angles[i_qubit, actual_pvm_idx, 0] + np.pi * (twirl > 0)
            phi[multi_index] = self.angles[i_qubit, actual_pvm_idx, 1]

        arr = BindingsArray(data={self._qc_theta: theta, self._qc_phi: phi})

        t2 = time.time()
        LOGGER.info(f"Finished building PVM bindings array. Took {t2 - t1:.6f}s")

        return arr

    def _get_outcome_label(
        self,
        pvm_idx: np.ndarray,
        bitstring_outcome: str,
    ) -> tuple[int, ...]:
        """Transform a PVM index and a bitstring outcome to a POVM outcome.

        The method takes into account the possible twirling of the measurements and un-does its
        effect. For single-qubit projective measurements, it means to perform a bit-flip of the
        classical outcome for each twirled projective measurement.

        Args:
            pvm_idx: qubit-wise index indicating which PVM was used to perform the measurement.
            bitstring_outcome: the outcome of the measurements performed with the PVM label
                by ``pvm_idx``. The order of qubit is assumed to be reversed.

        Returns:
            A tuple of indices indicating the POVM outcomes on each qubit. For each qubit,
            the index goes from :math:``0`` to :math:``2 * self.num_PVM - 1``.
        """
        return tuple(
            (pvm_idx[i] % self._num_PVMs) * 2 + (int(bit) + pvm_idx[i] // self._num_PVMs) % 2
            for i, bit in enumerate(bitstring_outcome[::-1])
        )

    @override
    def definition(self) -> ProductPOVM:
        t1 = time.time()
        LOGGER.info("Building POVM definition")

        stabilizers: np.ndarray = np.zeros((self.num_qubits, self._num_PVMs, 2, 2), dtype=complex)

        stabilizers[:, :, 0, 0] = np.cos(self.angles[:, :, 0] / 2.0)
        stabilizers[:, :, 0, 1] = (
            np.cos(self.angles[:, :, 1]) + 1.0j * np.sin(self.angles[:, :, 1])
        ) * np.sin(self.angles[:, :, 0] / 2.0)
        stabilizers[:, :, 1, 0] = -stabilizers[
            :, :, 0, 1
        ].conjugate()  # up to irrelevant global phase e^(i phi)
        stabilizers[:, :, 1, 1] = stabilizers[:, :, 0, 0]  # up to irrelevant global phase e^(i phi)

        stabilizers = np.multiply(stabilizers.T, np.sqrt(self.bias).T).T
        stabilizers = stabilizers.reshape((self.num_qubits, 2 * self._num_PVMs, 2))

        sq_povms = []
        for vecs in stabilizers:
            sq_povms.append(SingleQubitPOVM.from_vectors(vecs))

        prod = ProductPOVM.from_list(sq_povms)

        t2 = time.time()
        LOGGER.info(f"Finished POVM definition. Took {t2 - t1:.6f}s")

        return prod
