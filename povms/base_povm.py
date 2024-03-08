"""TODO."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator


class BasePOVM(ABC):
    """Abstract base class that contains all methods that any specific POVM subclass should implement."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Give the dimension of the Hilbert space on which the effects act."""

    @property
    @abstractmethod
    def n_outcomes(self) -> int:
        """Give the number of outcomes of the POVM."""

    @abstractmethod
    def _check_validity(self) -> bool:
        """Check if POVM axioms are fulfilled."""

    @abstractmethod
    def _clean_povm(self) -> bool:
        """Merge effects thats are proportionnal to each other and reorder effects in a standard way."""

    @abstractmethod
    def __getitem__(self, index: slice) -> Operator | list[Operator]:
        """Return a povm operator or a list of povm operators."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of outcomes of the POVM."""

    @abstractmethod
    def get_prob(self, rho: DensityMatrix) -> np.ndarray:
        """Return the outcome probabilities given a state rho."""

    @abstractmethod
    def get_omegas(self, obs: np.ndarray):
        """Return the decomposition weights of obserservable `obs` into the POVM effects."""
