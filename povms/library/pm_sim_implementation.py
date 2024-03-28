"""TODO."""

from __future__ import annotations

from collections import Counter

import numpy as np
from qiskit.circuit import ClassicalRegister, ParameterVector, QuantumCircuit, QuantumRegister

from povms.quantum_info.product_povm import ProductPOVM
from povms.quantum_info.single_qubit_povm import SingleQubitPOVM

from .povm_implementation import POVMImplementation


class RandomizedPMs(POVMImplementation):
    """Class to represent the implementation of randomized projective measurements."""

    def __init__(
        self,
        n_qubit: int,
        bias: np.ndarray,
        angles: np.ndarray,
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
        cr = ClassicalRegister(self.n_qubit, name="povm_meas")
        qc = QuantumCircuit(qr, cr, name="msmt_qc")
        for i in range(self.n_qubit):
            # We apply ``U_dag``, where ``U`` is the unitary operation to go from the computational basis
            # to the new measurement basis.
            qc.u(theta=-theta[i], phi=0.0, lam=-phi[i], qubit=i)

        qc.measure(qr, cr)

        return qc

    def distribute_shots(self, shots: int) -> Counter[tuple]:
        """Return a list with PVM label and associated number of shots.

        In the case of PM-simulable POVMs, each time we perfom a measurement we pick a
        random projective measurement among a given set of PVMs.

        Args:
            shots: total number of shots to be performed.

        Returns:
            The distribution of the shots among the different sets PVMs.
        """
        PVM_idx: np.ndarray = np.zeros((shots, self.n_qubit), dtype=int)

        for i in range(self.n_qubit):
            PVM_idx[:, i] = np.random.choice(self._n_PVMs, size=shots, replace=True, p=self.bias[i])
        counts = Counter(tuple(x) for x in PVM_idx)

        return counts

    def get_pvm_parameter(self, pvm_idx: tuple[int, ...]) -> np.ndarray:
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

    def get_outcome_label(
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
