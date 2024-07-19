# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the RandomizedProjectiveMeasurements class."""

from unittest import TestCase

import numpy as np
from numpy.random import default_rng
from povm_toolbox.library import ClassicalShadows, RandomizedProjectiveMeasurements
from povm_toolbox.library.metadata import POVMMetadata
from povm_toolbox.post_processor import POVMPostProcessor
from povm_toolbox.sampler import POVMSampler
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2


class TestRandomizedPMs(TestCase):
    SEED = 239486

    def setUp(self) -> None:
        super().setUp()

        basis_0 = np.array([1.0, 0], dtype=complex)
        basis_1 = np.array([0, 1.0], dtype=complex)
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

    def test_init_errors(self):
        """Test that the ``__init__`` method raises errors correctly."""
        # Sanity check
        measurement = RandomizedProjectiveMeasurements(
            1, bias=np.array([0.5, 0.5]), angles=np.array([0.0, 0.0, 0.5, 0.0])
        )
        self.assertIsInstance(measurement, RandomizedProjectiveMeasurements)
        with self.subTest("Incompatible ``bias`` and ``angles`` shapes.") and self.assertRaises(
            ValueError
        ):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([0.5, 0.5]), angles=np.array([0.0, 0.0, 0.5, 0.0, 0.4])
            )
        with self.subTest(
            "Shape of ``bias`` incompatible with number of qubits."
        ) and self.assertRaises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([[0.5, 0.5], [0.5, 0.5]]), angles=np.array([0.0, 0.0, 0.5, 0.0])
            )
        with self.subTest("Too many dims in ``bias``.") and self.assertRaises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([[[0.5, 0.5]]]), angles=np.array([0.0, 0.0, 0.5, 0.0])
            )
        with self.subTest("Negative value in ``bias``.") and self.assertRaises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([1.5, -0.5]), angles=np.array([0.0, 0.0, 0.5, 0.0])
            )
        with self.subTest("``bias`` not summing up to one.") and self.assertRaises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([0.5, 0.4]), angles=np.array([0.0, 0.0, 0.5, 0.0])
            )
        with self.subTest("``bias`` not summing up to one.") and self.assertRaises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([[0.5, 0.4], [0.5, 0.6]]), angles=np.array([0.0, 0.0, 0.5, 0.0])
            )
        with self.subTest(
            "Shape of ``angles`` incompatible with number of qubits."
        ) and self.assertRaises(ValueError):
            RandomizedProjectiveMeasurements(
                1,
                bias=np.array([0.5, 0.5]),
                angles=np.array([[0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.5, 0.0]]),
            )
        with self.subTest("Too many dims in ``angles``.") and self.assertRaises(ValueError):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([0.5, 0.5]), angles=np.array([[[0.0, 0.0, 0.5, 0.0]]])
            )
        with self.subTest("Invalid type for ``seed``.") and self.assertRaises(TypeError):
            RandomizedProjectiveMeasurements(
                1, bias=np.array([0.5, 0.5]), angles=np.array([0.0, 0.0, 0.5, 0.0]), seed=1.2
            )

    def test_init(self):
        """Test options in the ``__init__`` method."""
        num_qubits = 2

        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)

        with self.subTest("Initialization with ``seed`` of type ``Generator``."):
            rng = default_rng(self.SEED)
            measurement = ClassicalShadows(num_qubits, seed=rng)

            povm_sampler = POVMSampler(sampler=StatevectorSampler(seed=self.SEED))
            job = povm_sampler.run([qc], shots=128, povm=measurement)
            pub_result = job.result()[0]
            post_processor = POVMPostProcessor(pub_result)

            observable = SparsePauliOp(["ZI"], coeffs=[1.0])
            exp_value, std = post_processor.get_expectation_value(observable)
            self.assertAlmostEqual(exp_value, 0.9843750000000002)
            self.assertAlmostEqual(std, 0.12499231029496984)

    def test_repr(self):
        """Test that the ``__repr__`` method works correctly."""
        mub_str = (
            "RandomizedProjectiveMeasurements(num_qubits=1, bias=array([[0.2, "
            "0.8]]), angles=array([[[0, 1],\n        [2, 3]]]))"
        )
        povm = RandomizedProjectiveMeasurements(1, bias=np.array([0.2, 0.8]), angles=np.arange(4))
        self.assertEqual(povm.__repr__(), mub_str)

    def test_qc_build(self):
        """Test if we can build a QuantumCircuit."""
        for num_qubits in range(1, 11):
            q = np.random.uniform(0, 5, size=3 * num_qubits).reshape((num_qubits, 3))
            q /= q.sum(axis=1)[:, np.newaxis]

            angles = np.array([0.0, 0.0, 0.5 * np.pi, 0.0, 0.5 * np.pi, 0.5 * np.pi])

            cs_implementation = RandomizedProjectiveMeasurements(
                num_qubits=num_qubits, bias=q, angles=angles
            )

            qc = cs_implementation._build_qc()

            self.assertEqual(qc.num_qubits, num_qubits)

    def test_to_sampler_pub(self):
        """Test that the ``to_sampler_pub`` method works correctly."""
        num_qubits = 2
        qc = QuantumCircuit(2)
        qc.h(0)

        backend = FakeManilaV2()
        backend.set_options(seed_simulator=self.SEED)
        povm_sampler = POVMSampler(sampler=RuntimeSampler(mode=backend))

        pm = generate_preset_pass_manager(optimization_level=2, backend=backend)

        measurement = RandomizedProjectiveMeasurements(
            num_qubits,
            bias=np.array([0.2, 0.4, 0.4]),
            angles=np.array([0.0, 0.0, 0.8, 0.0, 0.8, 0.8]),
            seed=self.SEED,
        )

        job = povm_sampler.run([qc], shots=128, povm=measurement, pass_manager=pm)
        pub_result = job.result()[0]

        post_processor = POVMPostProcessor(pub_result)

        observable = SparsePauliOp(["ZI"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 1.015624999999999)
        self.assertAlmostEqual(std, 0.17850275939936872)
        observable = SparsePauliOp(["IZ"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 0.2734374999999998)
        self.assertAlmostEqual(std, 0.17806471515772768)
        observable = SparsePauliOp(["XI"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 0.2660390714463396)
        self.assertAlmostEqual(std, 0.2783956923005193)

    def test_binding_parameters(self):
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
        binding = BindingsArray.coerce({tuple(qc.parameters): pv})
        shots = 16

        pub, metadata = measurement.to_sampler_pub(qc, binding, shots=shots)

        self.assertEqual(pub.shape, (*pv_shape, shots))
        self.assertTrue(
            np.all(
                [
                    np.all(pub.parameter_values.data[("phi", "theta")][..., i, :] == pv)
                    for i in range(shots)
                ]
            )
        )
        self.assertTrue(
            np.allclose(
                measurement._get_pvm_bindings_array(metadata.pvm_keys).data[
                    ("phi_measurement[0]", "phi_measurement[1]")
                ],
                pub.parameter_values.data[("phi_measurement[0]", "phi_measurement[1]")],
            )
        )
        self.assertTrue(
            np.allclose(
                measurement._get_pvm_bindings_array(metadata.pvm_keys).data[
                    ("theta_measurement[0]", "theta_measurement[1]")
                ],
                pub.parameter_values.data[("theta_measurement[0]", "theta_measurement[1]")],
            )
        )

    def test_twirling(self):
        """Test if the twirling option works correctly."""
        qc = QuantumCircuit(2)
        qc.h(0)

        num_qubits = qc.num_qubits
        measurement = ClassicalShadows(num_qubits, seed=self.SEED, measurement_twirl=True)

        povm_sampler = POVMSampler(sampler=StatevectorSampler(seed=self.SEED))

        job = povm_sampler.run([qc], shots=128, povm=measurement)
        pub_result = job.result()[0]

        post_processor = POVMPostProcessor(pub_result)

        observable = SparsePauliOp(["ZI"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 1.0078125000000002)
        self.assertAlmostEqual(std, 0.1257341109337995)
        observable = SparsePauliOp(["IZ"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 0.32812500000000006)
        self.assertAlmostEqual(std, 0.15333777198692233)
        observable = SparsePauliOp(["ZY"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, -0.42187500000000006)
        self.assertAlmostEqual(std, 0.24164416287543897)
        observable = SparsePauliOp(["IX"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 1.1718749999999998)
        self.assertAlmostEqual(std, 0.12987983496490826)

    def test_shot_repetitions(self):
        """Test if the twirling option works correctly."""
        qc = QuantumCircuit(2)
        qc.h(0)

        num_qubits = qc.num_qubits
        measurement = ClassicalShadows(num_qubits, seed=self.SEED, shot_repetitions=7)

        povm_sampler = POVMSampler(sampler=StatevectorSampler(seed=self.SEED))

        job = povm_sampler.run([qc], shots=128, povm=measurement)
        pub_result = job.result()[0]

        self.assertEqual(sum(pub_result.get_counts()[0].values()), 128 * 7)
        self.assertEqual(pub_result.data.povm_measurement_creg.num_shots, 128 * 7)

        post_processor = POVMPostProcessor(pub_result)

        observable = SparsePauliOp(["ZI"], coeffs=[1.0])
        exp_value, std = post_processor.get_expectation_value(observable)
        self.assertAlmostEqual(exp_value, 0.9843750000000002)
        self.assertAlmostEqual(std, 0.04708403113719653)

    def test_povm_outcomes_errors(self):
        """Test that errors in ``_povm_outcomes`` method are raised correctly."""
        measurement = RandomizedProjectiveMeasurements(
            2,
            bias=np.array([0.3, 0.4, 0.3]),
            angles=np.array([0.0, 0.0, 0.5, 0.0, 0.5, 0.5]),
            seed=self.SEED,
        )
        qc = QuantumCircuit(2)
        qc.h(0)
        povm_sampler = POVMSampler(sampler=StatevectorSampler(seed=self.SEED))
        job = povm_sampler.run([qc], shots=10, povm=measurement)
        pub_result = job.result()[0]

        bit_array = getattr(pub_result.data, measurement.classical_register_name)
        povm_metadata = pub_result.metadata

        with self.subTest("Sanity check"):
            outcomes = measurement._povm_outcomes(bit_array, povm_metadata)
            self.assertSequenceEqual(
                outcomes,
                [(0, 2), (2, 2), (4, 0), (4, 4), (0, 2), (4, 2), (0, 2), (0, 0), (0, 4), (2, 2)],
            )

        with self.subTest("``pvm_keys`` attribute missing``.") and self.assertRaises(
            AttributeError
        ):
            faulty_metadata = POVMMetadata(
                povm_metadata.povm_implementation, povm_metadata.composed_circuit
            )
            measurement._povm_outcomes(bit_array, faulty_metadata)

        with self.subTest("Invalid ``loc`` argument.") and self.assertRaises(ValueError):
            measurement._povm_outcomes(bit_array, povm_metadata, loc=0)

    def test_get_pvm_bindings_array_errors(self):
        """Test that errors in ``_get_pvm_bindings_array`` methods are raised correctly."""

        measurement = RandomizedProjectiveMeasurements(
            2,
            bias=np.array([0.3, 0.4, 0.3]),
            angles=np.array([0.0, 0.0, 0.5, 0.0, 0.5, 0.5]),
            seed=self.SEED,
        )
        with self.assertRaises(ValueError):
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
                self.assertTrue(np.allclose(effect, povm[(0,)][2 * i_pvm + k].data))
