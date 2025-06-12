# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""MixedPOVMImplementation."""

from __future__ import annotations

import logging
import sys
import time

if sys.version_info < (3, 12):
    pass
else:
    pass  # pragma: no cover

from qiskit.circuit import (
    AncillaRegister,
    QuantumCircuit,
)
from qiskit.primitives.containers import DataBin
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.bit_array import BitArray
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.transpiler import StagedPassManager

from povm_toolbox.quantum_info import ProductPOVM

from .metadata import POVMMetadata
from .povm_implementation import POVMImplementation

LOGGER = logging.getLogger(__name__)


class MixedPOVMImplementation(POVMImplementation[POVMMetadata]):
    """TODO."""

    def __init__(
        self,
        num_qubits: int,
        sub_implementations: list[POVMImplementation],
        *,
        insert_barriers: bool = False,
    ) -> None:
        """Initialize TODO POVM.

        Args:
            num_qubits: the number of qubits.
            sub_implementations: TODO.
            insert_barriers: whether to insert a barrier between the composed circuits. This is not
                done by default but can prove useful when visualizing the composed circuit.

        Raises:
            ValueError: TODO.
        """
        super().__init__(num_qubits, measurement_layout=None, insert_barriers=insert_barriers)

        self._sub_implementations = sub_implementations

        check_num_qubits = 0
        for i, sub_implementation in enumerate(self._sub_implementations):
            sub_implementation.classical_register_name += f"_{i}"
            sub_implementation.measurement_circuit = sub_implementation._build_qc()
            check_num_qubits += sub_implementation.num_qubits

        if self.num_qubits != check_num_qubits:
            raise ValueError()

        # NOTE: this public attribute inherits its docstring from the base class
        self.measurement_circuit = self._build_qc()

    def _build_qc(self) -> QuantumCircuit:
        """Build the quantum circuit that implements the measurement.

        Returns:
            TODO.
        """
        t1 = time.time()
        LOGGER.info("Building POVM circuit")
        qc = QuantumCircuit(self.num_qubits)
        n = 0
        for sub_implementation in self._sub_implementations:
            qc.add_register(*sub_implementation.measurement_circuit.cregs)
            index_layout = []
            for qreg in sub_implementation.measurement_circuit.qregs:
                if isinstance(qreg, AncillaRegister):
                    index_layout += list(range(qc.num_qubits, qc.num_qubits + len(qreg)))
                    qc.add_register(qreg)
                else:
                    index_layout += list(range(n, n + len(qreg)))
                    n += len(qreg)
            qc = qc.compose(
                sub_implementation.measurement_circuit,
                qubits=index_layout,
                clbits=sub_implementation.measurement_circuit.clbits,
            )

        t2 = time.time()
        LOGGER.info(f"Finished circuit construction. Took {t2 - t1:.6f}s")

        return qc

    def definition(self) -> ProductPOVM:
        """Return the corresponding quantum-informational POVM representation."""
        return

    def to_sampler_pub(
        self,
        circuit: QuantumCircuit,
        circuit_binding: BindingsArray,
        shots: int,
        *,
        pass_manager: StagedPassManager | None = None,
    ) -> tuple[SamplerPub, POVMMetadata]:
        return

    def reshape_data_bin(self, data: DataBin) -> DataBin:
        return

    def _povm_outcomes(
        self,
        bit_array: BitArray,
        povm_metadata: POVMMetadata,
        *,
        loc: int | tuple[int, ...] | None = None,
    ) -> list[tuple[int, ...]]:
        return
