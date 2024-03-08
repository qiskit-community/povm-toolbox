"""TODO."""

from __future__ import annotations

from collections import Counter

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit

from .povm_implementation import POVMImplementation
from .product_povm import ProductPOVM
from .single_qubit_povm import SingleQubitPOVM


class PMSimImplementation(POVMImplementation):
    """TODO."""

    def __init__(
        self,
        n_qubit: int,
        parameters: np.ndarray | None = None,
    ) -> None:
        """TODO.

        Args:
            n_qubit: TODO.
            parameters: TODO.
        """
        super().__init__(n_qubit)
        if parameters is not None:
            self._set_parameters(parameters)

    def _set_parameters(self, parameters: np.ndarray) -> None:
        """TODO.

        Args:
            parameters: TODO.

        Raises:
            ValueError: TODO.
        """
        # n_param = n_qubit*(3*self.n_PVM-1)
        if len(parameters) % self.n_qubit != 0:
            raise ValueError(
                "The length of the parameter array is expected to be multiple of the number of qubits"
            )
        if (len(parameters) / self.n_qubit + 1) % 3 != 0:
            raise ValueError(
                "The number of parameters per qubit is expected to be of the form 3*n_PVM-1"
            )

        # TODO: move this to the __init__ method
        self.n_PVM = int((len(parameters) // self.n_qubit + 1) // 3)
        parameters = parameters.reshape((self.n_qubit, self.n_PVM * 3 - 1))
        self.angles = parameters[:, : 2 * self.n_PVM].reshape((self.n_qubit, self.n_PVM, 2))
        self.PVM_distributions = np.concatenate(
            (parameters[:, 2 * self.n_PVM :], np.ones((self.n_qubit, 1))), axis=1
        )
        if np.any(self.PVM_distributions < 0.0):
            raise ValueError(
                "There should not be any negative values in the probability distribution parameters."
            )
        self.PVM_distributions = (
            self.PVM_distributions / self.PVM_distributions.sum(axis=1)[:, np.newaxis]
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

        Args:
            shot: TODO.

        Returns:
            TODO.
        """
        PVM_idx: np.ndarray = np.zeros((shot, self.n_qubit), dtype=int)

        for i in range(self.n_qubit):
            PVM_idx[:, i] = np.random.choice(
                self.n_PVM, size=int(shot), replace=True, p=self.PVM_distributions[i]
            )
        counts = Counter(tuple(x) for x in PVM_idx)

        param = np.zeros((len(counts), self.n_qubit, 2))
        for i, combination in enumerate(counts):
            for j in range(self.n_qubit):
                param[i, j] = self.angles[j, combination[j]]

        list_param_shot: list[tuple[np.ndarray, int]] = [
            tuple((param[i], counts[combination])) for i, combination in enumerate(counts)
        ]

        return list_param_shot

    def to_povm(self) -> ProductPOVM:
        """TODO.

        Returns:
            TODO.
        """
        stabilizers: np.ndarray = np.zeros((self.n_qubit, self.n_PVM, 2, 2), dtype=complex)

        stabilizers[:, :, 0, 0] = np.cos(self.angles[:, :, 0] / 2.0)
        stabilizers[:, :, 0, 1] = (
            np.cos(self.angles[:, :, 1]) + 1.0j * np.sin(self.angles[:, :, 1])
        ) * np.sin(self.angles[:, :, 0] / 2.0)
        stabilizers[:, :, 1, 0] = stabilizers[:, :, 0, 1].conjugate()
        stabilizers[:, :, 1, 1] = -stabilizers[:, :, 0, 0]

        stabilizers = np.multiply(stabilizers.T, np.sqrt(self.PVM_distributions).T).T
        stabilizers = stabilizers.reshape((self.n_qubit, 2 * self.n_PVM, 2))

        sq_povms = []
        for vecs in stabilizers:
            sq_povms.append(SingleQubitPOVM.from_vectors(vecs))

        return ProductPOVM(sq_povms)
