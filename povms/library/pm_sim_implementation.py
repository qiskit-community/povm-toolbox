"""TODO."""

from __future__ import annotations

from collections import Counter

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit

from povms.quantum_info.product_povm import ProductPOVM
from povms.quantum_info.single_qubit_povm import SingleQubitPOVM

from .povm_implementation import POVMImplementation


class RandomizedPMs(POVMImplementation):
    """TODO."""

    def __init__(
        self,
        n_qubit: int,
        bias: np.ndarray,
        angles: np.ndarray,
    ) -> None:
        """Construct the most generalized version.

        If we extend this interface to support different effects for each qubit in the future, we
        may need to move away from np.ndarray input types to sequences of sequences.

        Args:
            n_qubits: the number of qubits.
            bias: can be either 1D or 2D. If 1D, it should contain float values indicating the bias
                for measuring in each of the effects. I.e. its length equals the number of effects.
                These floats should sum to 1. If 2D, it will have a new set of biases for each
                qubit.
            angles: can be either 1D or 2D. If 1D, it should contain float values to indicate the
                different angles of each effect. I.e. its length equals two times the number of
                effects (since we have 2 angles per effect). If 2D, it will have a new set of angles
                for each qubit.
        """
        super().__init__(n_qubit)

        self.n_PVM = len(bias[0])  # assert == len(angles) / 2
        self.angles = angles.reshape((n_qubit, self.n_PVM, 2))
        self.bias = bias
        if np.any(self.bias < 0.0):
            raise ValueError(
                "There should not be any negative values in the probability distribution parameters."
            )

    def _build_qc(self) -> QuantumCircuit:
        """TODO.

        Returns:
            TODO.
        """
        theta = ParameterVector("theta", length=self.n_qubit)
        phi = ParameterVector("phi", length=self.n_qubit)

        qc = QuantumCircuit(self.n_qubit)
        for i in range(self.n_qubit):
            qc.u(theta=theta[i], phi=phi[i], lam=0, qubit=i)

        return qc

    def get_parameter_and_shot(self, shot: int) -> list[tuple[np.ndarray, int]]:
        """Return a list with concrete parameter values and associated number of shots.

        Each set of parameter values correspond to a specific PVM to be performed. In the
        case of PM-simulable POVMs, each time we perfom a measurement we pick a random
        projective measurement among a given set of PVMS, i.e., we pick a random set of
        parameter values among the pre-defined list of sets.

        Args:
            shot: total number of shots to be performed.

        Returns:
            The distribution of the shots among the different sets of parameter values.
        """
        PVM_idx: np.ndarray = np.zeros((shot, self.n_qubit), dtype=int)

        for i in range(self.n_qubit):
            PVM_idx[:, i] = np.random.choice(self.n_PVM, size=shot, replace=True, p=self.bias[i])
        counts = Counter(tuple(x) for x in PVM_idx)

        param = np.zeros((len(counts), self.n_qubit, 2))
        for i, combination in enumerate(counts):
            for j in range(self.n_qubit):
                param[i, j] = self.angles[j, combination[j]]

        list_param_shot: list[tuple[np.ndarray, int]] = [
            tuple((param[i], counts[combination])) for i, combination in enumerate(counts)
        ]

        return list_param_shot

    # TODO: find a better name
    def to_povm(self) -> ProductPOVM:
        """Return the POVM corresponding to this implementation."""
        stabilizers: np.ndarray = np.zeros((self.n_qubit, self.n_PVM, 2, 2), dtype=complex)

        stabilizers[:, :, 0, 0] = np.cos(self.angles[:, :, 0] / 2.0)
        stabilizers[:, :, 0, 1] = (
            np.cos(self.angles[:, :, 1]) + 1.0j * np.sin(self.angles[:, :, 1])
        ) * np.sin(self.angles[:, :, 0] / 2.0)
        stabilizers[:, :, 1, 0] = stabilizers[:, :, 0, 1].conjugate()
        stabilizers[:, :, 1, 1] = -stabilizers[:, :, 0, 0]

        stabilizers = np.multiply(stabilizers.T, np.sqrt(self.bias).T).T
        stabilizers = stabilizers.reshape((self.n_qubit, 2 * self.n_PVM, 2))

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

        The same as above, but the angles are hard-coded to be X/Y/Z for all qubits.
        """
        angles = np.array(n_qubit * [0.0, 0.0, 0.5 * np.pi, 0.0, 0.5 * np.pi, 0.5 * np.pi])
        assert bias.shape == (n_qubit, 3)
        super().__init__(n_qubit=n_qubit, bias=bias, angles=angles)


class ClassicalShadows(LocallyBiased):
    """TODO."""

    def __init__(
        self,
        n_qubit: int,
    ):
        """Construct a classical shadow POVM.

        The same as above, but also hard-coding the biases to be equally distributed.
        """
        bias = 1.0 / 3.0 * np.ones((n_qubit, 3))
        super().__init__(n_qubit=n_qubit, bias=bias)
