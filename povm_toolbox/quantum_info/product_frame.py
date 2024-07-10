# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""ProductFrame."""

from __future__ import annotations

import math
import sys
import warnings
from collections.abc import Sequence
from typing import Generic, TypeVar

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self  # pragma: no cover

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override  # pragma: no cover


import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

from .base import BaseFrame
from .multi_qubit_frame import MultiQubitFrame

T = TypeVar("T", bound=MultiQubitFrame)


class ProductFrame(BaseFrame[tuple[int, ...]], Generic[T]):
    r"""Class to represent a set of product frame operators.

    A product frame :math:`M` is made of local frames :math:`M1, M2, ...` acting on respective
    subsystems. Each global operator can be written as the tensor product of local operators,
    :math:`M_{k_1, k_2, ...} = M1_{k_1} \otimes M2_{k_2} \otimes \cdots`.

    .. note::
       This is a base class which collects functionality common to various subclasses. As an
       end-user you would not use this class directly. Check out :mod:`povm_toolbox.quantum_info`
       for more general information.
    """

    def __init__(self, frames: dict[tuple[int, ...], T]) -> None:
        """Initialize from a mapping of local frames.

        Args:
            frames: a dictionary mapping from a tuple of subsystem indices to a local frame objects.

        Raises:
            ValueError: if any key in ``frames`` is not a tuple consisting of unique integers. In
                other words, every local frame must act on a distinct set of subsystem indices which
                do not overlap with each other.
            ValueError: if any key in ``frames`` re-uses a previously used subsystem index. In other
                words, all local frames must act on mutually exclusive subsystem indices.
            ValueError: if any key in ``frames`` does not specify the number of subsystem indices,
                which matches the number of systems acted upon by that local frame
                (:meth:`MultiQubitFrame.num_subsystems`).
        """
        subsystem_indices = set()
        self._dimension = 1
        self._num_operators = 1
        shape: list[int] = []
        for idx, frame in frames.items():
            idx_set = set(idx)
            if len(idx) != len(idx_set):
                raise ValueError(
                    "The subsystem indices acted upon by any local frame must be mutually "
                    f"exclusive. The index '{idx}' does not fulfill this criterion."
                )
            if any(i in subsystem_indices for i in idx):
                raise ValueError(
                    "The subsystem indices acted upon by all the local frames must be mutually "
                    f"exclusive. However, one of the indices in '{idx}' was already encountered "
                    "before."
                )
            if len(idx_set) != frame.num_subsystems:
                raise ValueError(
                    "The number of subsystem indices for a local frame must match the number of "
                    "subsystems which it acts upon. This is not satisfied for the local frame "
                    f"specified to act on subsystems '{idx}' but having support on "
                    f"'{frame.num_subsystems}' subsystems."
                )
            subsystem_indices.update(idx_set)
            self._dimension *= frame.dimension
            self._num_operators *= frame.num_operators
            shape.append(frame.num_operators)

        self._informationally_complete: bool = all(
            [frame.informationally_complete for frame in frames.values()]
        )

        self._frames = frames
        self._shape: tuple[int, ...] = tuple(shape)

        self._check_validity()

    def __repr__(self) -> str:
        """Return the string representation of a :class:`.ProductFrame` instance."""
        f_repr = "\n   " + "\n   ".join(f"{name}: {value}" for name, value in self._frames.items())
        return (
            f"{self.__class__.__name__}(num_subsystems={self.num_subsystems})"
            f"<{','.join(map(str, self.shape))}>:{f_repr}"
        )

    @classmethod
    def from_list(cls, frames: Sequence[T]) -> Self:
        """Construct a :class:`.ProductFrame` from a list of :class:`.MultiQubitFrame` objects.

        This is a convenience method to simplify the construction of a :class:`.ProductFrame` for
        the cases in which the local frame objects act on a sequential order of subsystems. In other
        words, this method converts the sequence of frames to a dictionary of frames in accordance
        with the input to :meth:`.ProductFrame.__init__` by using the positions along the sequence
        as subsystem indices.

        Below are some examples:

        >>> from qiskit.quantum_info import Operator
        >>> from povm_toolbox.quantum_info import SingleQubitPOVM, MultiQubitPOVM, ProductPOVM

        >>> sqp = SingleQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
        >>> product = ProductPOVM.from_list([sqp, sqp])
        >>> # is equivalent to
        >>> product = ProductPOVM({(0,): sqp, (1,): sqp})

        >>> mqp = MultiQubitPOVM(
        ...     [
        ...         Operator.from_label("00"),
        ...         Operator.from_label("01"),
        ...         Operator.from_label("10"),
        ...         Operator.from_label("11"),
        ...     ]
        ... )
        >>> product = ProductPOVM.from_list([mqp, mqp])
        >>> # is equivalent to
        >>> product = ProductPOVM({(0, 1): mqp, (2, 3): mqp})

        >>> product = ProductPOVM.from_list([sqp, sqp, mqp])
        >>> # is equivalent to
        >>> product = ProductPOVM({(0,): sqp, (1,): sqp, (2, 3): mqp})

        >>> product = ProductPOVM.from_list([sqp, mqp, sqp])
        >>> # is equivalent to
        >>> product = ProductPOVM({(0,): sqp, (1, 2): mqp, (3,): sqp})

        Args:
            frames: a sequence of :class:`.MultiQubitFrame` objects.

        Returns:
            A new :class:`.ProductFrame` instance.
        """
        frame_dict = {}
        idx = 0
        for frame in frames:
            prev_idx = idx
            idx += frame.num_subsystems
            frame_dict[tuple(range(prev_idx, idx))] = frame
        return cls(frame_dict)

    @property
    def informationally_complete(self) -> bool:
        """If the frame spans the entire Hilbert space."""
        return self._informationally_complete

    @property
    def dimension(self) -> int:
        """The dimension of the Hilbert space on which the effects act."""
        return self._dimension

    @property
    def num_operators(self) -> int:
        """The number of effects of the frame."""
        return self._num_operators

    @property
    def shape(self) -> tuple[int, ...]:
        """Give the number of operators per sub-system."""
        return self._shape

    @property
    def sub_systems(self) -> list[tuple[int, ...]]:
        """Give the number of operators per sub-system."""
        return list(self._frames.keys())

    def _check_validity(self) -> None:
        """Check if frame axioms are fulfilled for all local frames.

        .. note::
           This raises whatever errors the local frames' methods may raise.
        """
        for povm in self._frames.values():
            povm._check_validity()

    def __getitem__(self, sub_system: tuple[int, ...]) -> T:
        r"""Return the :class:`.MultiQubitFrame` acting on the specified sub-system.

        Args:
            sub_system: indicate the sub-system on which the queried frame acts on.

        Returns:
            The :class:`.MultiQubitFrame` acting on the specified sub-system.
        """
        return self._frames[sub_system]

    def __len__(self) -> int:
        """Return the number of outcomes of the product frame."""
        return self.num_operators

    def _trace_of_prod(self, operator: SparsePauliOp, frame_op_idx: tuple[int, ...]) -> float:
        """Return the trace of the product of a Hermitian operator with a specific frame operator.

        Args:
            operator: the input operator to multiply with a frame operator.
            frame_op_idx: the label specifying the frame operator to use. The frame operator is
                labeled by a tuple of integers (one index per local frame).

        Returns:
            The trace of the product of the input operator with the specified frame operator.

        Raises:
            IndexError: when the provided outcome label (tuple of integers) has a number of integers
                which does not correspond to the number of local frames making up the product frame.
            IndexError: when a local index exceeds the number of operators of the corresponding
                local frame.
            ValueError: when the output is not a real number.
        """
        p_idx = 0.0 + 0.0j

        # Second, we iterate over our input operator, ``operator``.
        for label, op_coeff in operator.label_iter():
            summand = op_coeff
            # Third, we iterate over the POVMs stored inside the ProductPOVM.
            #   - ``j`` is the index of the POVM inside the ``ProductPOVM``. This encodes the axis
            #     of the high-dimensional array ``p_init`` along which this local POVM is encoded.
            #   - ``idx`` are the qubit indices on which this local POVM acts.
            #   - ``povm`` is the actual local POVM object.
            for j, (idx, povm) in enumerate(self._frames.items()):
                # Extract the local Pauli term on the qubit indices of this local POVM.
                sublabel = "".join(label[-(i + 1)] for i in idx)
                # Try to obtain the coefficient of the local POVM for this local Pauli term.
                try:
                    local_idx = frame_op_idx[j]
                    coeff = povm.pauli_operators[local_idx][sublabel]
                except KeyError:
                    # If it does not exist, the current summand becomes 0 because it would be
                    # multiplied by 0.
                    summand = 0.0
                    # In this case we can break the iteration over the remaining local POVMs.
                    break
                except IndexError as exc:
                    if len(frame_op_idx) <= j:
                        raise IndexError(
                            f"The outcome label {frame_op_idx} does not match the expected shape. "
                            f"It is supposed to contain {len(self._frames)} integers, but has "
                            f"{len(frame_op_idx)}."
                        ) from exc
                    if povm.num_operators <= frame_op_idx[j]:
                        raise IndexError(
                            f"Outcome index '{frame_op_idx[j]}' is out of range for the local POVM"
                            f" acting on subsystems {idx}. This POVM has {povm.num_operators}"
                            " outcomes."
                        ) from exc
                    raise exc
                else:
                    # If the label does exist, we multiply the coefficient into our summand.
                    # The factor 2*N_qubit comes from Tr[(P_1...P_N)^2] = 2*N.
                    summand *= coeff * 2 * povm.num_subsystems

            # Once we have finished computing our summand, we add it into ``p_init``.
            p_idx += summand
        if abs(p_idx.imag) > operator.atol:
            warnings.warn(f"Expected a real number, instead got {p_idx}.", stacklevel=2)
        return float(p_idx.real)

    @override
    def analysis(
        self,
        hermitian_op: SparsePauliOp | Operator,
        frame_op_idx: tuple[int, ...] | set[tuple[int, ...]] | None = None,
    ) -> float | dict[tuple[int, ...], float] | np.ndarray:
        if not isinstance(hermitian_op, SparsePauliOp):
            # Convert the provided operator to a Pauli operator.
            hermitian_op = SparsePauliOp.from_operator(hermitian_op)

        # Assert matching operator and POVM sizes.
        if hermitian_op.num_qubits != self.num_subsystems:
            raise ValueError(
                f"Size of the operator ({hermitian_op.num_qubits}) does not match the size of the "
                f"povm ({math.log2(self.dimension)})."
            )

        # If frame_op_idx is ``None``, it means all outcomes are queried
        if frame_op_idx is None:
            # Extract the number of outcomes for each local POVM.

            # Create the output probability array as a high-dimensional matrix. This matrix will
            # have its number of dimensions equal to the number of POVMs stored inside the
            # ProductPOVM. The length of each dimension is given by the number of outcomes of the
            # POVM encoded along it.
            p_init: np.ndarray = np.zeros(self.shape, dtype=float)

            # First, we iterate over all the positions of ``p_init``. This corresponds to the
            # different probabilities for the different outcomes whose probability we want to
            # compute.
            #   - ``m`` is the multi-dimensional index into the high-dimensional ``p_init`` array.
            for m, _ in np.ndenumerate(p_init):
                p_init[m] = self._trace_of_prod(hermitian_op, m)
            return p_init
        if isinstance(frame_op_idx, set):
            return {idx: self._trace_of_prod(hermitian_op, idx) for idx in frame_op_idx}
        if isinstance(frame_op_idx, tuple):
            return self._trace_of_prod(hermitian_op, frame_op_idx)
        raise TypeError("Wrong type for ``frame_op_idx``.")
