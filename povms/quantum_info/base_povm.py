"""TODO."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator


class BasePOVM(ABC):
    """Abstract base class that contains all methods that any specific POVM subclass should implement."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Give the dimension of the Hilbert space on which the effects act."""

    @property
    def n_subsystems(self) -> int:
        """Return the number of subsystems which the effects act on.

        For qubits, this is always :math:`log2(self.dimension)`.
        """
        return int(np.log2(self.dimension))

    @property
    @abstractmethod
    def n_outcomes(self) -> int:
        """Give the number of outcomes of the POVM."""

    @property
    @abstractmethod
    def alphas(self) -> np.ndarray | list[np.ndarray]:
        """Paremeters of the dual frame."""

    @alphas.setter
    @abstractmethod
    def alphas(self, var: np.ndarray | list[np.ndarray]) -> None:
        """Set paremeters of the dual frame."""

    @abstractmethod
    def _check_validity(self) -> None:
        """Check if POVM axioms are fulfilled."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of outcomes of the POVM."""

    @abstractmethod
    def get_prob(
        self,
        rho: DensityMatrix,
        outcome_idx: Any | set[Any] | None = None,
    ) -> float | dict[Any, float] | np.ndarray:
        """Return the outcome probabilities given a state rho."""

    @abstractmethod
    def get_omegas(
        self,
        obs: Operator,
        outcome_idx: Any | set[Any] | None = None,
    ) -> float | dict[Any, float] | np.ndarray:
        """Return the decomposition weights of obserservable `obs` into the POVM effects."""