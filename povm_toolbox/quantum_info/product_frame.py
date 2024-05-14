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

import sys
import warnings
from collections.abc import Sequence
from typing import Generic, TypeVar

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

from .base_frame import BaseFrame
from .multi_qubit_frame import MultiQubitFrame

T = TypeVar("T", bound=MultiQubitFrame)


class ProductFrame(BaseFrame[tuple[int, ...]], Generic[T]):
    r"""Class to represent a set of product frame operators.

    A product frame :math:`M` is made of local frames :math:`M1, M2, ...` acting
    on respective subsystems. Each global effect can be written as the tensor
    product of local operators,
    :math:`M_{k_1, k_2, ...} = M1_{k_1} \otimes M2_{k2} \otimes ...`.
    """

    def __init__(self, povms: dict[tuple[int, ...], T]):
        """Initialize a :class:`.ProductFrame` instance.

        Args:
            povms: a dictionary mapping from a tuple of subsystem indices to a :class:`.MultiQubitFrame`
                object.

        Raises:
            ValueError: if any key in ``povms`` is not a tuple consisting of unique integers. In
                other words, every POVM must act on a distinct set of subsystem indices which do not
                overlap with each other.
            ValueError: if any key in ``povms`` re-uses a previously used subsystem index. In other
                words, all POVMs must act on mutually exclusive subsystem indices.
            ValueError: if any key in ``povms`` does not specify the number of subsystem indices,
                which matches the number of systems acted upon by that POVM
                (:meth:`MultiQubitFrame.n_subsystems`).
        """
        subsystem_indices = set()
        self._dimension = 1
        self._n_operators = 1
        shape: list[int] = []
        for idx, povm in povms.items():
            idx_set = set(idx)
            if len(idx) != len(idx_set):
                raise ValueError(
                    "The subsystem indices acted upon by any POVM must be mutually exclusive. "
                    f"The index '{idx}' does not fulfill this criterion."
                )
            if any(i in subsystem_indices for i in idx):
                raise ValueError(
                    "The subsystem indices acted upon by all the POVMs must be mutually exclusive. "
                    f"However, one of the indices in '{idx}' was already encountered before."
                )
            if len(idx_set) != povm.n_subsystems:
                raise ValueError(
                    "The number of subsystem indices for a POVM must match the number of subsystems"
                    " which it acts upon. This is not satisfied for the POVM specified to act on "
                    f"subsystems '{idx}' but having support on '{povm.n_subsystems}' subsystems."
                )
            subsystem_indices.update(idx_set)
            self._dimension *= povm.dimension
            self._n_operators *= povm.n_operators
            shape.append(povm.n_operators)

        self._informationally_complete: bool = all(
            [povm.informationally_complete for povm in povms.values()]
        )

        self._povms = povms
        self._shape: tuple[int, ...] = tuple(shape)

        self._check_validity()

    @classmethod
    def from_list(cls, povms: Sequence[T]) -> Self:
        """Construct a :class:`.ProductFrame` from a list of :class:`.MultiQubitFrame` objects.

        This is a convenience method to simplify the construction of a :class:`.ProductPOVM` for the cases
        in which the POVM objects act on a sequential order of subsystems. In other words, this
        method converts the sequence of POVMs to a dictionary of POVMs in accordance with the input
        to :meth:`.ProductFrame.__init__` by using the positions along the sequence as subsystem
        indices.

        Below are some examples:

        .. code-block:: python

            sqp = SingleQubitPOVM([Operator.from_label("0"), Operator.from_label("1")])
            product = ProductPOVM.from_list([sqp, sqp])
            # is equivalent to
            product = ProductPOVM({(0,): sqp, (1,): sqp})

            mqp = MultiQubitFrame(
                [
                    Operator.from_label("00"),
                    Operator.from_label("01"),
                    Operator.from_label("10"),
                    Operator.from_label("11"),
                ]
            )
            product = ProductPOVM.from_list([mqp, mqp])
            # is equivalent to
            product = ProductPOVM({(0, 1): mqp, (2, 3): mqp})

            product = ProductPOVM.from_list([sqp, sqp, mqp])
            # is equivalent to
            product = ProductPOVM({(0,): sqp, (1,): sqp, (2, 3): mqp})

            product = ProductPOVM.from_list([sqp, mqp, sqp])
            # is equivalent to
            product = ProductPOVM({(0,): sqp, (1, 2): mqp, (3,): sqp})

        Args:
            povms: a sequence of :class:`.MultiQubitFrame` objects.

        Returns:
            A new :class:`.ProductPOVM` instance.
        """
        povm_dict = {}
        idx = 0
        for povm in povms:
            prev_idx = idx
            idx += povm.n_subsystems
            povm_dict[tuple(range(prev_idx, idx))] = povm
        return cls(povm_dict)

    @property
    def informationally_complete(self) -> bool:
        """Return if the frame spans the entire Hilbert space."""
        return self._informationally_complete

    @property
    def dimension(self) -> int:
        """Give the dimension of the Hilbert space on which the effects act."""
        return self._dimension

    @property
    def n_operators(self) -> int:
        """Give the number of single-qubit operators forming the POVM."""
        return self._n_operators

    @property
    def shape(self) -> tuple[int, ...]:
        """Give the number of operators per sub-system."""
        return self._shape

    @property
    def sub_systems(self) -> list[tuple[int, ...]]:
        """Give the number of operators per sub-system."""
        return list(self._povms.keys())

    def _check_validity(self) -> None:
        """Check if POVM axioms are fulfilled.

        Raises:
            TODO.
        """
        for povm in self._povms.values():
            povm._check_validity()

    def __getitem__(self, sub_system: tuple[int, ...]) -> T:
        r"""Return the :class:`.MultiQubitFrame` acting on the specified sub-system.

        Args:
            sub_system: indicate the sub-system on which the queried frame acts on.

        Returns:
            The :class:`.MultiQubitFrame` acting on the specified sub-system.
        """
        return self._povms[sub_system]

    def __len__(self) -> int:
        """Return the number of outcomes of the POVM."""
        return self.n_operators

    def _trace_of_prod(self, operator: SparsePauliOp, frame_op_idx: tuple[int, ...]) -> float:
        """Return the trace of the product of a Hermitian operator with a specific frame operator.

        Args:
            operator: the input operator to multiply with a frame operator.
            frame_op_idx: the label specifying the frame operator to use. The frame operator is labeled
                by a tuple of integers (one index per local frame).

        Returns:
            The trace of the product of the input operator with the specified frame operator.

        Raises:
            IndexError: when the provided outcome label (tuple of integers) has a number of integers
                which does not correspond to the number of local frames making up the product frame.
            IndexError: when a local index exceed the number of operators of the corresponding local frame.
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
            for j, (idx, povm) in enumerate(self._povms.items()):
                # Extract the local Pauli term on the qubit indices of this local POVM.
                sublabel = "".join(label[-(i + 1)] for i in idx)
                # Try to obtain the coefficient of the local POVM for this local Pauli term.
                try:
                    coeff = povm.pauli_operators[frame_op_idx[j]][sublabel]
                except KeyError:
                    # If it does not exist, the current summand becomes 0 because it would be
                    # multiplied by 0.
                    summand = 0.0
                    # In this case we can break the iteration over the remaining local POVMs.
                    break
                except IndexError as exc:
                    if len(frame_op_idx) <= j:
                        raise IndexError(
                            f"The outcome label {frame_op_idx} does not match the expected shape. It is"
                            f" supposed to contain {len(self._povms)} integers, but has {len(frame_op_idx)}."
                        ) from exc
                    if povm.n_operators <= frame_op_idx[j]:
                        raise IndexError(
                            f"Outcome index '{frame_op_idx[j]}' is out of range for the local POVM"
                            f" acting on subsystems {idx}. This POVM has {povm.n_operators} outcomes."
                        ) from exc
                    raise exc
                else:
                    # If the label does exist, we multiply the coefficient into our summand.
                    # The factor 2*N_qubit comes from Tr[(P_1...P_N)^2] = 2*N.
                    summand *= coeff * 2 * povm.n_subsystems

            # Once we have finished computing our summand, we add it into ``p_init``.
            p_idx += summand
        if abs(p_idx.imag) > operator.atol:
            warnings.warn(f"Expected a real number, instead got {p_idx}.", stacklevel=2)
        return float(p_idx.real)

    def analysis(
        self,
        hermitian_op: SparsePauliOp | Operator,
        frame_op_idx: tuple[int, ...] | set[tuple[int, ...]] | None = None,
    ) -> float | dict[tuple[int, ...], float] | np.ndarray:
        """TODO.

        Args:
            hermitian_op: TODO.
            frame_op_idx: the outcomes for which one queries the trace. Each outcome is labeled
                by a tuple of integers (one index per local POVM). One can query a single outcome or a
                set of outcomes. If ``None``, all outcomes are queried.

        Returns:
            Frame coefficients, specified by ``frame_op_idx``, with respect to the Hermitian operator
            ``hermitian_op``. If a specific coefficient was queried, a ``float`` is returned. If a
            specific set of coefficients was queried, a dictionary mapping labels to coefficients
            is returned. If all coefficients were queried, a high-dimensional array with one dimension
            per local frame stored inside ``self`` is returned. The length of each dimension is given
            by the number of operators of the frame encoded along that axis.

        Raises:
            TypeError: when the provided single or sequence of labels ``frame_op_idx`` does not have
                a valid type.
            ValueError: when the provided ``operator`` does not act on the same number of qubits as
                ``self``.
        """
        if not isinstance(hermitian_op, SparsePauliOp):
            # Convert the provided operator to a Pauli operator.
            hermitian_op = SparsePauliOp.from_operator(hermitian_op)

        # Assert matching operator and POVM sizes.
        if hermitian_op.num_qubits != self.n_subsystems:
            raise ValueError(
                f"Size of the operator {hermitian_op.n_qubits} does not match the size of the povm {len(self)}."
            )

        # If frame_op_idx is ``None``, it means all outcomes are queried
        if frame_op_idx is None:
            # Extract the number of outcomes for each local POVM.

            # Create the output probability array as a high-dimensional matrix. This matrix will have
            # its number of dimensions equal to the number of POVMs stored inside the ProductPOVM. The
            # length of each dimension is given by the number of outcomes of the POVM encoded along it.
            p_init: np.ndarray = np.zeros(self.shape, dtype=float)

            # First, we iterate over all the positions of ``p_init``. This corresponds to the different
            # probabilities for the different outcomes whose probability we want to compute.
            #   - ``m`` is the multi-dimensional index into the high-dimensional ``p_init`` array.
            for m, _ in np.ndenumerate(p_init):
                p_init[m] = self._trace_of_prod(hermitian_op, m)
            return p_init
        if isinstance(frame_op_idx, set):
            return {idx: self._trace_of_prod(hermitian_op, idx) for idx in frame_op_idx}
        if isinstance(frame_op_idx, tuple):
            return self._trace_of_prod(hermitian_op, frame_op_idx)
        raise TypeError("wrong shape of frame_op_idx")
