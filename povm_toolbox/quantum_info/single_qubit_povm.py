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
