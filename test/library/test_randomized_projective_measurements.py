# (C) Copyright IBM 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the RandomizedProjectiveMeasurements class."""

from collections import Counter

import numpy as np
import pytest
import qiskit
from numpy.random import default_rng
from povm_toolbox.library import ClassicalShadows, RandomizedProjectiveMeasurements
from povm_toolbox.library.metadata import POVMMetadata
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.sampler import POVMSampler
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BackendSamplerV2 as Sampler
from qiskit.primitives import StatevectorSampler
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeManilaV2


class TestRandomizedPMs:
    SEED = 239486

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        basis_0 = np.asarray([1.0, 0], dtype=complex)
        basis_1 = np.asarray([0, 1.0], dtype=complex)
        basis_plus = 1.0 / np.sqrt(2) * (basis_0 + basis_1)
        basis_minus = 1.0 / np.sqrt(2) * (basis_0 - basis_1)
        basis_plus_i = 1.0 / np.sqrt(2) * (basis_0 + 1.0j * basis_1)
        basis_minus_i = 1.0 / np.sqrt(2) * (basis_0 - 1.0j * basis_1)

        self.Z0 = np.outer(basis_0, basis_0.conj())
        self.Z1 = np.outer(basis_1, basis_1.conj())
        self.X0 = np.outer(basis_plus, basis_plus.conj())
        self.X1 = np.outer(basis_minus, basis_minus.conj())
        self.Y0 = np.outer(basis_plus_i, basis_plus_i.conj())
        self.Y1 = np.outer(basis_minus_i, basis_minus_i.conj())

    def test_init_errors(self, subtests):
        """Test that the ``__init__`` method raises errors correctly."""
        # Sanity check
        measurement = RandomizedProjectiveMeasurements(
            1, bias=np.asarray([0.5, 0.5]), angles=np.asarray([0.0, 0.0, 0.5, 0.0])
        )
        assert isinstance(measurement, RandomizedProjectiveMeasurements)
        with subtests.test("Incompatible ``bias`` and ``angles`` shapes.") and pytest.raises(
            ValueError
        ):
            RandomizedProjectiveMeasurements(
                1, bias=np.asarray([0.5, 0.5]), angles=np.asarray([0.0, 0.0, 0.5, 0.0, 0.4])
            )
        with subtests.test(
            "Shape of ``bias`` incompatible with number of qubits."
        ) and pytest.raises(ValueError):
            RandomizedProjectiveMeasurements(
                1,
                bias=np.asarray([[0.5, 0.5], [0.5, 0.5]]),
                angles=np.asarray([0.0, 0.0, 0.5, 0.0]),
            )
        with subtests.test("Too many dims in ``bias``.") and pytest.raises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.asarray([[[0.5, 0.5]]]), angles=np.asarray([0.0, 0.0, 0.5, 0.0])
            )
        with subtests.test("Negative value in ``bias``.") and pytest.raises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.asarray([1.5, -0.5]), angles=np.asarray([0.0, 0.0, 0.5, 0.0])
            )
        with subtests.test("``bias`` not summing up to one.") and pytest.raises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.asarray([0.5, 0.4]), angles=np.asarray([0.0, 0.0, 0.5, 0.0])
            )
        with subtests.test("``bias`` not summing up to one.") and pytest.raises(ValueError):
            RandomizedProjectiveMeasurements(
                1,
                bias=np.asarray([[0.5, 0.4], [0.5, 0.6]]),
                angles=np.asarray([0.0, 0.0, 0.5, 0.0]),
            )
        with subtests.test(
            "Shape of ``angles`` incompatible with number of qubits."
        ) and pytest.raises(ValueError):
            RandomizedProjectiveMeasurements(
                1,
                bias=np.asarray([0.5, 0.5]),
                angles=np.asarray([[0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.5, 0.0]]),
            )
        with subtests.test("Too many dims in ``angles``.") and pytest.raises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.asarray([0.5, 0.5]), angles=np.asarray([[[0.0, 0.0, 0.5, 0.0]]])
            )
        with subtests.test("Invalid type for ``seed``.") and pytest.raises(TypeError):
            RandomizedProjectiveMeasurements(
                1, bias=np.asarray([0.5, 0.5]), angles=np.asarray([0.0, 0.0, 0.5, 0.0]), seed=1.2
            )

    def test_init(self):
        """Test options in the ``__init__`` method."""
        num_qubits = 2

        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)

        rng = default_rng(self.SEED)
        measurement = ClassicalShadows(num_qubits, seed=rng)

        povm_sampler = POVMSampler(sampler=StatevectorSampler(seed=default_rng(self.SEED)))
        job = povm_sampler.run([qc], shots=128, povm=measurement)
        pub_result = job.result()[0]
        post_processor = POVMPostProcessor(pub_result)

        observable = SparsePauliOp(["ZI"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        assert np.isclose(exp_value, 0.09374999999999983)
        assert np.isclose(std, 0.15226210145459726)

    def test_repr(self):
        """Test that the ``__repr__`` method works correctly."""
        mub_str = (
            "RandomizedProjectiveMeasurements(num_qubits=1, bias=array([[0.2, "
            "0.8]]), angles=array([[[0., 1.],\n        [2., 3.]]]))"
        )
        povm = RandomizedProjectiveMeasurements(
            1, bias=np.asarray([0.2, 0.8]), angles=np.arange(4, dtype=float)
        )
        assert povm.__repr__() == mub_str

    def test_qc_build(self):
        """Test if we can build a QuantumCircuit."""
        for num_qubits in range(1, 11):
            q = np.random.uniform(0, 5, size=3 * num_qubits).reshape((num_qubits, 3))
            q /= q.sum(axis=1)[:, np.newaxis]

            angles = np.asarray([0.0, 0.0, 0.5 * np.pi, 0.0, 0.5 * np.pi, 0.5 * np.pi])

            cs_implementation = RandomizedProjectiveMeasurements(
                num_qubits=num_qubits, bias=q, angles=angles
            )

            qc = cs_implementation._build_qc()

            assert qc.num_qubits == num_qubits

    @pytest.mark.parametrize(
        ["pauli", "expected_exp_val", "expected_std"],
        [
            ("ZI", 1.0156250000000002, 0.1785027593993689),
            # NOTE: the seemingly random differentiation based on the Qiskit version below are a
            # simple artifact of a change in simulator seed handling. This results in vastly
            # different samples being generated for this circuit which in turn results in vastly
            # different expectation values (since we have a very small number of shots to being
            # with). It is a pure coincidence that this only shows up for the following two test
            # cases and not anywhere else.
            (
                "IZ",
                0.1171875000000002 if qiskit.__version__.startswith("2") else 0.8203125000000001,
                0.17940912620515942 if qiskit.__version__.startswith("2") else 0.16430837874418136,
            ),
            (
                "XI",
                0.04822534966686553 if qiskit.__version__.startswith("2") else 0.48385279322581426,
                0.2793620349223184 if qiskit.__version__.startswith("2") else 0.27607615877584846,
            ),
        ],
    )
    def test_to_sampler_pub(self, pauli: str, expected_exp_val: float, expected_std: float):
        """Test that the ``to_sampler_pub`` method works correctly."""
        num_qubits = 2
        qc = QuantumCircuit(2)
        qc.h(0)

        backend = FakeManilaV2()
        povm_sampler = POVMSampler(
            sampler=Sampler(backend=backend, options={"seed_simulator": self.SEED})
        )

        pm = generate_preset_pass_manager(
            optimization_level=0,
            initial_layout=[0, 1],
            backend=backend,
            seed_transpiler=self.SEED,
        )

        measurement = RandomizedProjectiveMeasurements(
            num_qubits,
            bias=np.asarray([0.2, 0.4, 0.4]),
            angles=np.asarray([0.0, 0.0, 0.8, 0.0, 0.8, 0.8]),
            seed=self.SEED,
        )

        job = povm_sampler.run([qc], shots=128, povm=measurement, pass_manager=pm)
        pub_result = job.result()[0]

        post_processor = POVMPostProcessor(pub_result)

        observable = SparsePauliOp([pauli], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        assert np.isclose(exp_value, expected_exp_val)
        assert np.isclose(std, expected_std)

    def test_binding_parameters(self):
        num_qubits = 2
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.ry(theta=Parameter("theta"), qubit=0)
        qc.rx(theta=Parameter("phi"), qubit=1)

        measurement = RandomizedProjectiveMeasurements(
            num_qubits,
            bias=np.asarray([0.2, 0.4, 0.4]),
            angles=np.asarray([0.0, 0.0, 0.8, 0.0, 0.8, 0.8]),
            seed=self.SEED,
        )

        pv_shape = (5, 3)
        pv = np.arange(np.prod(pv_shape) * qc.num_parameters).reshape(
            (*pv_shape, qc.num_parameters)
        )
        binding = BindingsArray.coerce({tuple(qc.parameters): pv})
        shots = 16

        pub, metadata = measurement.to_sampler_pub(qc, binding, shots=shots)

        assert pub.shape == (*pv_shape, shots)
        assert np.all(
            [
                np.all(pub.parameter_values.data[("phi", "theta")][..., i, :] == pv)
                for i in range(shots)
            ]
        )
        assert np.allclose(
            measurement._get_pvm_bindings_array(metadata.pvm_keys).data[
                ("phi_measurement[0]", "phi_measurement[1]")
            ],
            pub.parameter_values.data[("phi_measurement[0]", "phi_measurement[1]")],
        )
        assert np.allclose(
            measurement._get_pvm_bindings_array(metadata.pvm_keys).data[
                ("theta_measurement[0]", "theta_measurement[1]")
            ],
            pub.parameter_values.data[("theta_measurement[0]", "theta_measurement[1]")],
        )

    def test_get_counts(self, subtests):
        num_qubits = 2
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.ry(theta=Parameter("theta"), qubit=0)
        qc.rx(theta=Parameter("phi"), qubit=1)

        measurement = RandomizedProjectiveMeasurements(
            num_qubits,
            bias=np.array([0.2, 0.4, 0.4]),
            angles=np.array([0.0, 0.0, 0.8, 0.0, 0.8, 0.8]),
            seed=self.SEED,
        )

        pv_shape = (5, 3)
        pv = np.arange(np.prod(pv_shape) * qc.num_parameters).reshape(
            (*pv_shape, qc.num_parameters)
        )
        shots = 64

        povm_sampler = POVMSampler(sampler=StatevectorSampler(seed=default_rng(self.SEED)))

        job = povm_sampler.run(
            [(qc.assign_parameters([1.2, -0.3])), (qc, pv[0]), (qc, pv)],
            shots=shots,
            povm=measurement,
        )

        pub_result = job.result()[0]
        with subtests.test("loc: integer") and pytest.raises(ValueError):
            _ = pub_result.get_counts(loc=1)
        with subtests.test("loc: tuple[int,...]") and pytest.raises(ValueError):
            _ = pub_result.get_counts(loc=(1,))
        with subtests.test("loc: None"):
            counts = pub_result.get_counts(loc=None)
            assert isinstance(counts, Counter)
            assert counts == Counter(
                {
                    (2, 2): 16,
                    (4, 3): 9,
                    (0, 4): 7,
                    (0, 2): 6,
                    (4, 5): 6,
                    (2, 4): 5,
                    (2, 5): 5,
                    (2, 0): 3,
                    (3, 4): 2,
                    (5, 1): 2,
                    (5, 3): 1,
                    (4, 1): 1,
                    (0, 0): 1,
                }
            )
        with subtests.test("loc: Ellipsis"):
            counts = pub_result.get_counts(loc=...)
            assert isinstance(counts, np.ndarray)
            assert counts.shape == (1,)
            assert np.all(
                counts
                == np.asarray(
                    [
                        Counter(
                            {
                                (2, 2): 16,
                                (4, 3): 9,
                                (0, 4): 7,
                                (0, 2): 6,
                                (4, 5): 6,
                                (2, 4): 5,
                                (2, 5): 5,
                                (2, 0): 3,
                                (3, 4): 2,
                                (5, 1): 2,
                                (5, 3): 1,
                                (4, 1): 1,
                                (0, 0): 1,
                            }
                        )
                    ]
                )
            )

        pub_result = job.result()[1]
        with subtests.test("loc: integer"):
            counts = pub_result.get_counts(loc=1)
            assert isinstance(counts, Counter)
            assert counts == Counter(
                {
                    (3, 5): 11,
                    (5, 5): 9,
                    (3, 2): 7,
                    (3, 3): 4,
                    (1, 2): 4,
                    (3, 1): 3,
                    (5, 3): 3,
                    (1, 3): 3,
                    (5, 1): 3,
                    (4, 3): 2,
                    (1, 1): 2,
                    (4, 5): 2,
                    (2, 3): 2,
                    (3, 0): 1,
                    (3, 4): 1,
                    (4, 2): 1,
                    (5, 2): 1,
                    (0, 5): 1,
                    (0, 3): 1,
                    (1, 4): 1,
                    (1, 5): 1,
                    (5, 0): 1,
                }
            )
        with subtests.test("loc: tuple[int,...]"):
            counts = pub_result.get_counts(loc=(1,))
            assert isinstance(counts, Counter)
            assert counts == Counter(
                {
                    (3, 5): 11,
                    (5, 5): 9,
                    (3, 2): 7,
                    (3, 3): 4,
                    (1, 2): 4,
                    (3, 1): 3,
                    (5, 3): 3,
                    (1, 3): 3,
                    (5, 1): 3,
                    (4, 3): 2,
                    (1, 1): 2,
                    (4, 5): 2,
                    (2, 3): 2,
                    (3, 0): 1,
                    (3, 4): 1,
                    (4, 2): 1,
                    (5, 2): 1,
                    (0, 5): 1,
                    (0, 3): 1,
                    (1, 4): 1,
                    (1, 5): 1,
                    (5, 0): 1,
                }
            )
        with subtests.test("loc: tuple[int,...]") and pytest.raises(ValueError):
            _ = pub_result.get_counts(loc=(1, 2))
        with subtests.test("loc: None"):
            counts = pub_result.get_counts(loc=None)
            assert isinstance(counts, Counter)
            assert counts == Counter(
                {
                    (3, 2): 17,
                    (2, 3): 11,
                    (3, 5): 11,
                    (5, 5): 10,
                    (1, 4): 9,
                    (4, 3): 9,
                    (3, 4): 8,
                    (5, 2): 7,
                    (4, 4): 7,
                    (5, 3): 7,
                    (2, 5): 7,
                    (4, 2): 6,
                    (4, 5): 6,
                    (0, 3): 6,
                    (2, 4): 6,
                    (5, 4): 5,
                    (5, 0): 5,
                    (1, 2): 5,
                    (2, 2): 5,
                    (1, 3): 5,
                    (3, 3): 5,
                    (3, 0): 4,
                    (4, 0): 4,
                    (0, 5): 4,
                    (0, 4): 3,
                    (2, 0): 3,
                    (3, 1): 3,
                    (5, 1): 3,
                    (2, 1): 3,
                    (1, 1): 2,
                    (4, 1): 2,
                    (1, 0): 1,
                    (1, 5): 1,
                    (0, 1): 1,
                    (0, 2): 1,
                }
            )
        with subtests.test("loc: Ellipsis"):
            counts = pub_result.get_counts(loc=...)
            assert isinstance(counts, np.ndarray)
            assert counts.shape == (3,)
            assert np.all(
                counts
                == np.asarray(
                    [
                        Counter(
                            {
                                (3, 2): 10,
                                (1, 4): 8,
                                (3, 4): 7,
                                (5, 2): 6,
                                (5, 4): 5,
                                (4, 4): 4,
                                (5, 0): 4,
                                (5, 3): 3,
                                (3, 0): 3,
                                (2, 2): 2,
                                (4, 2): 2,
                                (1, 3): 2,
                                (4, 0): 2,
                                (2, 0): 2,
                                (1, 2): 1,
                                (0, 4): 1,
                                (2, 3): 1,
                                (1, 0): 1,
                            }
                        ),
                        Counter(
                            {
                                (3, 5): 11,
                                (5, 5): 9,
                                (3, 2): 7,
                                (3, 3): 4,
                                (1, 2): 4,
                                (3, 1): 3,
                                (5, 3): 3,
                                (1, 3): 3,
                                (5, 1): 3,
                                (4, 3): 2,
                                (1, 1): 2,
                                (4, 5): 2,
                                (2, 3): 2,
                                (3, 0): 1,
                                (3, 4): 1,
                                (4, 2): 1,
                                (5, 2): 1,
                                (0, 5): 1,
                                (0, 3): 1,
                                (1, 4): 1,
                                (1, 5): 1,
                                (5, 0): 1,
                            }
                        ),
                        Counter(
                            {
                                (2, 3): 8,
                                (2, 5): 7,
                                (4, 3): 7,
                                (2, 4): 6,
                                (0, 3): 5,
                                (4, 5): 4,
                                (0, 5): 3,
                                (4, 4): 3,
                                (4, 2): 3,
                                (2, 1): 3,
                                (2, 2): 3,
                                (4, 1): 2,
                                (0, 4): 2,
                                (4, 0): 2,
                                (5, 3): 1,
                                (2, 0): 1,
                                (3, 3): 1,
                                (0, 1): 1,
                                (0, 2): 1,
                                (5, 5): 1,
                            }
                        ),
                    ]
                )
            )

        pub_result = job.result()[2]
        with subtests.test("loc: integer") and pytest.raises(ValueError):
            _ = pub_result.get_counts(loc=1)
        with subtests.test("loc: tuple[int,...]"):
            counts = pub_result.get_counts(loc=(1, 2))
            assert isinstance(counts, Counter)
            assert counts == Counter(
                {
                    (4, 3): 9,
                    (4, 1): 9,
                    (2, 5): 7,
                    (4, 5): 6,
                    (2, 3): 5,
                    (4, 2): 5,
                    (2, 1): 3,
                    (5, 3): 3,
                    (4, 4): 3,
                    (0, 5): 3,
                    (0, 4): 2,
                    (0, 3): 2,
                    (0, 1): 2,
                    (2, 2): 1,
                    (3, 1): 1,
                    (2, 4): 1,
                    (0, 2): 1,
                    (5, 5): 1,
                }
            )
        with subtests.test("loc: None"):
            counts = pub_result.get_counts(loc=None)
            assert isinstance(counts, Counter)
            assert counts == Counter(
                {
                    (5, 2): 58,
                    (4, 4): 49,
                    (5, 5): 46,
                    (3, 5): 45,
                    (4, 3): 43,
                    (2, 3): 42,
                    (4, 5): 41,
                    (5, 3): 39,
                    (4, 2): 39,
                    (3, 3): 39,
                    (5, 4): 37,
                    (2, 2): 33,
                    (2, 4): 32,
                    (3, 4): 29,
                    (3, 2): 28,
                    (2, 5): 28,
                    (4, 1): 27,
                    (2, 1): 26,
                    (1, 4): 25,
                    (5, 0): 23,
                    (3, 0): 22,
                    (0, 5): 22,
                    (0, 3): 22,
                    (1, 5): 21,
                    (5, 1): 19,
                    (1, 3): 18,
                    (1, 2): 17,
                    (3, 1): 17,
                    (0, 2): 14,
                    (1, 0): 13,
                    (0, 1): 11,
                    (0, 4): 10,
                    (2, 0): 9,
                    (4, 0): 6,
                    (1, 1): 5,
                    (0, 0): 5,
                }
            )
        with subtests.test("loc: Ellipsis"):
            counts = pub_result.get_counts(loc=...)
            assert isinstance(counts, np.ndarray)
            assert counts.shape, 5 == 3
            assert np.all(
                counts
                == np.asarray(
                    [
                        [
                            Counter(
                                {
                                    (5, 4): 7,
                                    (5, 2): 7,
                                    (4, 4): 6,
                                    (1, 4): 5,
                                    (5, 0): 5,
                                    (3, 4): 5,
                                    (2, 4): 3,
                                    (3, 0): 3,
                                    (3, 2): 3,
                                    (4, 2): 3,
                                    (2, 2): 3,
                                    (1, 0): 3,
                                    (1, 2): 2,
                                    (2, 0): 2,
                                    (3, 5): 2,
                                    (5, 3): 1,
                                    (0, 2): 1,
                                    (4, 0): 1,
                                    (4, 3): 1,
                                    (1, 5): 1,
                                }
                            ),
                            Counter(
                                {
                                    (5, 5): 12,
                                    (3, 5): 7,
                                    (3, 1): 6,
                                    (3, 3): 5,
                                    (5, 3): 4,
                                    (0, 5): 3,
                                    (1, 3): 3,
                                    (4, 3): 2,
                                    (5, 2): 2,
                                    (1, 5): 2,
                                    (3, 4): 2,
                                    (0, 3): 2,
                                    (3, 2): 2,
                                    (2, 2): 2,
                                    (5, 1): 1,
                                    (2, 3): 1,
                                    (3, 0): 1,
                                    (1, 2): 1,
                                    (1, 1): 1,
                                    (4, 5): 1,
                                    (4, 0): 1,
                                    (5, 0): 1,
                                    (2, 5): 1,
                                    (0, 0): 1,
                                }
                            ),
                            Counter(
                                {
                                    (2, 1): 9,
                                    (0, 3): 5,
                                    (2, 5): 5,
                                    (4, 3): 5,
                                    (2, 4): 4,
                                    (2, 3): 4,
                                    (4, 1): 4,
                                    (4, 5): 4,
                                    (4, 2): 3,
                                    (0, 4): 3,
                                    (4, 4): 3,
                                    (5, 2): 2,
                                    (2, 2): 2,
                                    (0, 5): 2,
                                    (3, 3): 2,
                                    (0, 1): 2,
                                    (5, 3): 2,
                                    (5, 4): 1,
                                    (5, 5): 1,
                                    (0, 0): 1,
                                }
                            ),
                        ],
                        [
                            Counter(
                                {
                                    (4, 4): 10,
                                    (5, 2): 8,
                                    (1, 4): 6,
                                    (2, 2): 5,
                                    (2, 4): 4,
                                    (4, 2): 4,
                                    (5, 4): 4,
                                    (4, 5): 3,
                                    (3, 4): 3,
                                    (5, 0): 3,
                                    (1, 0): 2,
                                    (4, 0): 2,
                                    (1, 5): 2,
                                    (3, 0): 2,
                                    (1, 2): 2,
                                    (5, 3): 1,
                                    (0, 3): 1,
                                    (3, 1): 1,
                                    (3, 3): 1,
                                }
                            ),
                            Counter(
                                {
                                    (3, 5): 11,
                                    (1, 3): 6,
                                    (5, 5): 6,
                                    (5, 3): 5,
                                    (1, 5): 5,
                                    (3, 3): 3,
                                    (3, 2): 3,
                                    (5, 2): 3,
                                    (3, 4): 3,
                                    (3, 1): 2,
                                    (5, 4): 2,
                                    (5, 1): 2,
                                    (1, 2): 2,
                                    (5, 0): 2,
                                    (1, 1): 1,
                                    (4, 1): 1,
                                    (2, 3): 1,
                                    (3, 0): 1,
                                    (1, 4): 1,
                                    (1, 0): 1,
                                    (2, 5): 1,
                                    (4, 3): 1,
                                    (2, 4): 1,
                                }
                            ),
                            Counter(
                                {
                                    (4, 3): 9,
                                    (4, 1): 9,
                                    (2, 5): 7,
                                    (4, 5): 6,
                                    (2, 3): 5,
                                    (4, 2): 5,
                                    (2, 1): 3,
                                    (5, 3): 3,
                                    (4, 4): 3,
                                    (0, 5): 3,
                                    (0, 4): 2,
                                    (0, 3): 2,
                                    (0, 1): 2,
                                    (2, 2): 1,
                                    (3, 1): 1,
                                    (2, 4): 1,
                                    (0, 2): 1,
                                    (5, 5): 1,
                                }
                            ),
                        ],
                        [
                            Counter(
                                {
                                    (4, 2): 10,
                                    (4, 4): 9,
                                    (2, 4): 7,
                                    (2, 2): 6,
                                    (5, 2): 6,
                                    (5, 0): 4,
                                    (5, 4): 4,
                                    (3, 2): 3,
                                    (3, 0): 2,
                                    (2, 3): 2,
                                    (2, 0): 2,
                                    (1, 0): 1,
                                    (5, 1): 1,
                                    (1, 2): 1,
                                    (2, 5): 1,
                                    (3, 4): 1,
                                    (1, 4): 1,
                                    (1, 1): 1,
                                    (1, 3): 1,
                                    (2, 1): 1,
                                }
                            ),
                            Counter(
                                {
                                    (5, 3): 8,
                                    (3, 5): 7,
                                    (3, 2): 7,
                                    (5, 2): 6,
                                    (3, 3): 6,
                                    (1, 5): 5,
                                    (5, 1): 5,
                                    (3, 0): 4,
                                    (5, 5): 3,
                                    (5, 4): 3,
                                    (1, 2): 2,
                                    (3, 4): 2,
                                    (3, 1): 2,
                                    (4, 2): 1,
                                    (1, 3): 1,
                                    (0, 2): 1,
                                    (1, 0): 1,
                                }
                            ),
                            Counter(
                                {
                                    (2, 3): 12,
                                    (4, 5): 8,
                                    (2, 1): 5,
                                    (0, 1): 4,
                                    (4, 3): 4,
                                    (0, 2): 4,
                                    (4, 1): 4,
                                    (3, 3): 3,
                                    (2, 5): 3,
                                    (4, 2): 2,
                                    (4, 4): 2,
                                    (0, 5): 2,
                                    (5, 1): 2,
                                    (5, 3): 1,
                                    (5, 5): 1,
                                    (0, 4): 1,
                                    (5, 4): 1,
                                    (2, 4): 1,
                                    (0, 3): 1,
                                    (5, 2): 1,
                                    (3, 5): 1,
                                    (2, 2): 1,
                                }
                            ),
                        ],
                        [
                            Counter(
                                {
                                    (4, 4): 7,
                                    (2, 2): 7,
                                    (1, 4): 6,
                                    (5, 2): 5,
                                    (2, 4): 4,
                                    (5, 4): 4,
                                    (4, 2): 3,
                                    (5, 0): 3,
                                    (4, 3): 3,
                                    (0, 0): 3,
                                    (3, 4): 2,
                                    (3, 2): 2,
                                    (1, 2): 2,
                                    (3, 0): 2,
                                    (2, 3): 2,
                                    (5, 3): 1,
                                    (1, 5): 1,
                                    (4, 1): 1,
                                    (4, 5): 1,
                                    (2, 0): 1,
                                    (0, 3): 1,
                                    (2, 1): 1,
                                    (1, 0): 1,
                                    (3, 3): 1,
                                }
                            ),
                            Counter(
                                {
                                    (5, 5): 8,
                                    (3, 5): 8,
                                    (5, 2): 6,
                                    (3, 0): 5,
                                    (5, 3): 5,
                                    (3, 4): 4,
                                    (3, 3): 4,
                                    (5, 4): 4,
                                    (1, 3): 3,
                                    (3, 2): 3,
                                    (1, 1): 2,
                                    (1, 5): 2,
                                    (4, 5): 1,
                                    (1, 0): 1,
                                    (1, 4): 1,
                                    (3, 1): 1,
                                    (5, 1): 1,
                                    (0, 3): 1,
                                    (1, 2): 1,
                                    (5, 0): 1,
                                    (0, 5): 1,
                                    (4, 2): 1,
                                }
                            ),
                            Counter(
                                {
                                    (0, 5): 9,
                                    (2, 3): 7,
                                    (4, 3): 6,
                                    (4, 5): 6,
                                    (5, 5): 4,
                                    (2, 5): 4,
                                    (3, 3): 3,
                                    (0, 2): 3,
                                    (5, 1): 3,
                                    (4, 1): 3,
                                    (0, 1): 3,
                                    (5, 3): 3,
                                    (0, 3): 2,
                                    (4, 4): 2,
                                    (3, 5): 2,
                                    (2, 1): 1,
                                    (5, 4): 1,
                                    (3, 1): 1,
                                    (5, 2): 1,
                                }
                            ),
                        ],
                        [
                            Counter(
                                {
                                    (2, 4): 7,
                                    (4, 4): 7,
                                    (2, 2): 6,
                                    (4, 3): 4,
                                    (2, 0): 4,
                                    (4, 2): 4,
                                    (0, 4): 3,
                                    (4, 5): 3,
                                    (0, 3): 3,
                                    (2, 1): 3,
                                    (0, 2): 3,
                                    (2, 3): 3,
                                    (5, 2): 3,
                                    (3, 4): 2,
                                    (4, 0): 2,
                                    (1, 4): 2,
                                    (3, 3): 1,
                                    (5, 0): 1,
                                    (1, 3): 1,
                                    (1, 0): 1,
                                    (2, 5): 1,
                                }
                            ),
                            Counter(
                                {
                                    (5, 2): 8,
                                    (5, 5): 8,
                                    (5, 4): 6,
                                    (3, 2): 5,
                                    (3, 4): 5,
                                    (1, 2): 4,
                                    (5, 3): 3,
                                    (1, 3): 3,
                                    (5, 0): 3,
                                    (3, 3): 3,
                                    (1, 0): 2,
                                    (1, 4): 2,
                                    (3, 5): 2,
                                    (4, 2): 2,
                                    (4, 3): 2,
                                    (3, 0): 2,
                                    (1, 5): 1,
                                    (0, 4): 1,
                                    (4, 5): 1,
                                    (3, 1): 1,
                                }
                            ),
                            Counter(
                                {
                                    (3, 3): 7,
                                    (4, 5): 7,
                                    (4, 3): 6,
                                    (2, 3): 5,
                                    (4, 1): 5,
                                    (2, 5): 5,
                                    (3, 5): 5,
                                    (0, 3): 4,
                                    (5, 1): 4,
                                    (2, 1): 3,
                                    (5, 5): 2,
                                    (3, 1): 2,
                                    (0, 5): 2,
                                    (5, 3): 2,
                                    (1, 5): 2,
                                    (1, 4): 1,
                                    (0, 2): 1,
                                    (4, 2): 1,
                                }
                            ),
                        ],
                    ]
                )
            )

    @pytest.mark.parametrize(
        ["pauli", "expected_exp_val", "expected_std"],
        [
            ("ZI", 1.0078125000000002, 0.1257341109337995),
            ("IZ", 0.2343749999999999, 0.15468582228868283),
            ("ZY", 0.42187499999999994, 0.24164416287543897),
            ("IX", 1.1718749999999998, 0.12987983496490826),
        ],
    )
    def test_twirling(self, pauli: str, expected_exp_val: float, expected_std: float):
        """Test if the twirling option works correctly."""
        qc = QuantumCircuit(2)
        qc.h(0)

        num_qubits = qc.num_qubits
        measurement = ClassicalShadows(num_qubits, seed=self.SEED, measurement_twirl=True)

        povm_sampler = POVMSampler(sampler=StatevectorSampler(seed=default_rng(self.SEED)))

        job = povm_sampler.run([qc], shots=128, povm=measurement)
        pub_result = job.result()[0]

        post_processor = POVMPostProcessor(pub_result)

        observable = SparsePauliOp([pauli], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        assert np.isclose(exp_value, expected_exp_val)
        assert np.isclose(std, expected_std)

    def test_shot_repetitions(self):
        """Test if the twirling option works correctly."""
        qc = QuantumCircuit(2)
        qc.h(0)

        num_qubits = qc.num_qubits
        measurement = ClassicalShadows(num_qubits, seed=self.SEED, shot_repetitions=7)

        povm_sampler = POVMSampler(sampler=StatevectorSampler(seed=default_rng(self.SEED)))

        job = povm_sampler.run([qc], shots=128, povm=measurement)
        pub_result = job.result()[0]

        assert sum(pub_result.get_counts(loc=...)[0].values()) == 128 * 7
        assert pub_result.data.povm_measurement_creg.num_shots == 128 * 7

        post_processor = POVMPostProcessor(pub_result)

        observable = SparsePauliOp(["ZI"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        assert np.isclose(exp_value, 0.9843750000000002)
        assert np.isclose(std, 0.04708403113719653)

    def test_povm_outcomes_errors(self, subtests):
        """Test that errors in ``_povm_outcomes`` method are raised correctly."""
        measurement = RandomizedProjectiveMeasurements(
            2,
            bias=np.asarray([0.3, 0.4, 0.3]),
            angles=np.asarray([0.0, 0.0, 0.5, 0.0, 0.5, 0.5]),
            seed=self.SEED,
        )
        qc = QuantumCircuit(2)
        qc.h(0)
        povm_sampler = POVMSampler(sampler=StatevectorSampler(seed=default_rng(self.SEED)))
        job = povm_sampler.run([qc], shots=10, povm=measurement)
        pub_result = job.result()[0]

        bit_array = getattr(pub_result.data, measurement.classical_register_name)
        povm_metadata = pub_result.metadata

        with subtests.test("Sanity check"):
            outcomes = measurement._povm_outcomes(bit_array, povm_metadata, loc=tuple())
            assert outcomes == [
                (0, 2),
                (2, 2),
                (5, 0),
                (5, 4),
                (0, 2),
                (4, 3),
                (0, 2),
                (0, 0),
                (0, 4),
                (2, 2),
            ]

        with subtests.test("``pvm_keys`` attribute missing``.") and pytest.raises(AttributeError):
            faulty_metadata = POVMMetadata(
                povm_metadata.povm_implementation, povm_metadata.composed_circuit
            )
            measurement._povm_outcomes(bit_array, faulty_metadata, loc=tuple())

        with subtests.test("Invalid ``loc`` argument.") and pytest.raises(ValueError):
            measurement._povm_outcomes(bit_array, povm_metadata, loc=0)

    def test_get_pvm_bindings_array_errors(self):
        """Test that errors in ``_get_pvm_bindings_array`` methods are raised correctly."""

        measurement = RandomizedProjectiveMeasurements(
            2,
            bias=np.asarray([0.3, 0.4, 0.3]),
            angles=np.asarray([0.0, 0.0, 0.5, 0.0, 0.5, 0.5]),
            seed=self.SEED,
        )
        with pytest.raises(ValueError):
            # ``pvm_idx.shape`` is supposed to be ``(..., povm_sampler_pub.shots, num_qubits)``
            measurement._get_pvm_bindings_array(pvm_idx=np.zeros(10))

    def test_definition(self):
        """Test that the ``definition`` method works correctly."""
        rng = default_rng()
        num_qubits = 1
        num_pvms = 5

        # randomly pick a bias (distribution) for each qubit
        bias = rng.dirichlet(alpha=rng.uniform(0, 10, size=num_pvms), size=num_qubits)

        # uniformly sample points on the Bloch sphere to define the effects
        phi = rng.uniform(0, 2 * np.pi, size=num_pvms * num_qubits).reshape((num_qubits, num_pvms))
        costheta = rng.uniform(-1, 1, size=num_pvms * num_qubits).reshape((num_qubits, num_pvms))
        theta = np.arccos(costheta)
        angles = np.stack((theta, phi), axis=2).reshape((num_qubits, 2 * num_pvms))

        # define measurement and the quantum-informational POVM
        measurement = RandomizedProjectiveMeasurements(num_qubits, bias=bias, angles=angles)
        measurement_circuit = measurement.measurement_circuit
        povm = measurement.definition()

        for i_pvm in range(num_pvms):  # loop on the projective measurements
            # bound the parameters corresponding to the pvm
            bc = measurement_circuit.assign_parameters(
                np.concatenate((phi[:, i_pvm], theta[:, i_pvm]))
            )
            bc.remove_final_measurements()
            # define the change of basis from Z-basis to arbitrary basis defined by (theta, phi)
            unitary_transformation = Operator(bc.inverse()).data
            for k in range(2):  # loop on outcomes {0,1}
                # compute the POVM effect implemented by the circuit
                vec = unitary_transformation[:, k]
                effect = bias[0, i_pvm] * np.outer(vec, vec.conj())
                # check that the circuit implements the correct POVM effect
                assert np.allclose(effect, povm[(0,)][2 * i_pvm + k].data)

    def test_non_ic_measurement(self, subtests):
        """Test the implementation of a RPM that is not IC."""

        qc = QuantumCircuit(1)
        qc.u(0.4, -0.1, 0.1, qubit=0)

        measurement = RandomizedProjectiveMeasurements(
            num_qubits=1,
            angles=np.asarray([0.0, 0.0, np.pi / 2, np.pi / 2]),
            bias=np.asarray([0.5, 0.5]),
            seed=self.SEED,
        )

        with subtests.test("Test that the POVM is not IC."):
            assert not measurement.definition().informationally_complete

        sampler = StatevectorSampler(seed=default_rng(self.SEED))
        povm_sampler = POVMSampler(sampler=sampler)

        job = povm_sampler.run([qc], shots=32, povm=measurement)
        pub_result = job.result()[0]

        post_processor = POVMPostProcessor(pub_result)

        with subtests.test("Test the dual frame."):
            assert post_processor.dual.is_dual_to(measurement.definition())

        with subtests.test("Test with compatible observable."):
            observable = SparsePauliOp(["Z"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            assert np.isclose(exp_value, 1.4999999999999993)
            assert np.isclose(std, 0.1555427542095637)

        with subtests.test("Test with incompatible observable."):
            observable = SparsePauliOp(["X"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            assert np.isclose(exp_value, 0.0)
            assert np.isclose(std, 0.0)

    def test_array_like_arguments(self):
        """Test the initialization of a RPM with array-like arguments ."""

        qc = QuantumCircuit(1)
        qc.u(0.4, -0.1, 0.1, qubit=0)

        measurement = RandomizedProjectiveMeasurements(
            num_qubits=1,
            angles=[0.2, 0.4, 0.5, 0.4, 0.5, 1],
            bias=[0.5, 0.2, 0.3],
            seed=self.SEED,
        )
        sampler = StatevectorSampler(seed=default_rng(self.SEED))
        povm_sampler = POVMSampler(sampler=sampler)

        job = povm_sampler.run([qc], shots=32, povm=measurement)
        pub_result = job.result()[0]

        post_processor = POVMPostProcessor(pub_result)
        observable = SparsePauliOp(["Z"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        assert np.isclose(exp_value, 2.22338143795974)
        assert np.isclose(std, 0.3435308147247512)
