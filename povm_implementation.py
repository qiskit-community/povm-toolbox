import numpy as np
from collections import Counter
from qiskit.circuit import QuantumCircuit, ParameterVector
from base_povm import Povm
from single_qubit_povm import SingleQubitPOVM
from product_povm import ProductPOVM


class POVMImplementation:

    def __init__(
        self,
        n_qubit: int,
    ) -> None:

        self.n_qubit = n_qubit
        self.parametrized_qc = self._build_qc()

    def _build_qc(self) -> QuantumCircuit:
        raise NotImplementedError("The subclass of POVMImplementation must implement `_build_from_param` method.")

    def get_parameter_and_shot(self, shot: int) -> QuantumCircuit:
        raise NotImplementedError("The subclass of POVMImplementation must implement `get_parameter_and_shot` method.")

    def to_povm(self) -> Povm:
        raise NotImplementedError("The subclass of POVMImplementation must implement `to_povm` method.")


class ProductPVMSimPOVMImplementation(POVMImplementation):

    def __init__(
        self,
        n_qubit: int,
        parameters: np.ndarray | None = None,
    ) -> None:

        super().__init__(n_qubit)
        self._set_parameters(parameters)

    def _set_parameters(self, parameters: np.ndarray) -> None:
        # n_param = n_qubit*(3*self.n_PVM-1)
        if len(parameters) % self.n_qubit != 0:
            raise ValueError('The length of the parameter array is expected to be multiple of the number of qubits')
        elif (len(parameters) / self.n_qubit + 1) % 3 != 0:
            raise ValueError('The number of parameters per qubit is expected to be of the form 3*n_PVM-1')
        else:
            self.n_PVM = int((len(parameters) // self.n_qubit + 1) // 3)
            parameters = parameters.reshape((self.n_qubit, self.n_PVM * 3 - 1))
            self.angles = parameters[:, :2 * self.n_PVM].reshape((self.n_qubit, self.n_PVM, 2))
            self.PVM_distributions = np.concatenate((parameters[:, 2 * self.n_PVM:], np.ones((self.n_qubit, 1))), axis=1)
            if np.any(self.PVM_distributions < 0.):
                raise ValueError('There should not be any negative values in the probability distribution parameters.')
            else:
                self.PVM_distributions = self.PVM_distributions / self.PVM_distributions.sum(axis=1)[:, np.newaxis]

    def _build_qc(self) -> QuantumCircuit:

        theta = ParameterVector('theta', length=self.n_qubit)
        phi = ParameterVector('phi', length=self.n_qubit)

        qc = QuantumCircuit(self.n_qubit)
        for i in range(self.n_qubit):
            qc.u(theta=theta[i], phi=phi[i], lam=0, qubit=i)

        return qc

    def get_parameter_and_shot(self, shot: int) -> QuantumCircuit:
        """
        Returns a list with concrete parameter values and associated number of shots.
        """

        PVM_idx = np.zeros((shot, self.n_qubit), dtype=int)

        for i in range(self.n_qubit):
            PVM_idx[:, i] = np.random.choice(self.n_PVM, size=int(shot), replace=True, p=self.PVM_distributions[i])
        counts = Counter(tuple(x) for x in PVM_idx)

        return [tuple(([self.angles[i, combination[i]] for i in range(self.n_qubit)], counts[combination])) for combination in counts]

    def to_povm(self) -> Povm:

        stabilizers = np.zeros((self.n_qubit, self.n_PVM, 2, 2), dtype=complex)

        stabilizers[:, :, 0, 0] = np.cos(self.angles[:, :, 0] / 2.)
        stabilizers[:, :, 0, 1] = (np.cos(self.angles[:, :, 1]) + 1.j * np.sin(self.angles[:, :, 1])) * np.sin(self.angles[:, :, 0] / 2.)
        stabilizers[:, :, 1, 0] = stabilizers[:, :, 0, 1].conjugate()
        stabilizers[:, :, 1, 1] = -stabilizers[:, :, 0, 0]

        stabilizers = np.multiply(stabilizers.T, np.sqrt(self.PVM_distributions).T).T
        stabilizers = stabilizers.reshape((self.n_qubit, 2 * self.n_PVM, 2))

        sq_povms = []
        for vecs in stabilizers:
            sq_povms.append(SingleQubitPOVM.from_vectors(vecs))

        return ProductPOVM(sq_povms)
