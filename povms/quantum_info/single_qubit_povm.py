"""TODO."""

from __future__ import annotations

from .multi_qubit_povm import MultiQubitPOVM


class SingleQubitPOVM(MultiQubitPOVM):
    """Class to represent a set of IC single-qubit POVM operators."""

    def _check_validity(self) -> None:
        """TODO.

        Raises:
            ValueError: TODO.
        """
        if not self.dimension == 2:
            raise ValueError(
                f"Dimension of Single Qubit POVM operator space should be 2, not {self.dimension}."
            )
        super()._check_validity()
