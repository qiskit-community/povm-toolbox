# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the POVMSamplerPub class."""

from unittest import TestCase

from povm_toolbox.library import ClassicalShadows
from povm_toolbox.library.metadata import POVMMetadata
from povm_toolbox.sampler import POVMSamplerPub
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.sampler_pub import SamplerPub


class TestPOVMSamplerPub(TestCase):
    """Tests for the ``POVMSamplerPub`` class."""

    def setUp(self) -> None:
        super().setUp()

    def test_initialization(self):
        """Test that the ``__init__`` method works correctly."""
        qc = QuantumCircuit(1)
        cs = ClassicalShadows(1)
        with self.subTest("Initialization with validation."):
            pub = POVMSamplerPub(circuit=qc, parameter_values=None, shots=1, povm=cs, validate=True)
            self.assertIsInstance(pub, POVMSamplerPub)

        theta = Parameter("theta")
        qc.ry(theta=theta, qubit=0)
        parameter_values = BindingsArray({theta: [0, 1, 2]})
        with self.subTest("Initialization with `parameter_values` and validation."):
            pub = POVMSamplerPub(
                circuit=qc, parameter_values=parameter_values, shots=1, povm=cs, validate=True
            )
            self.assertIsInstance(pub, POVMSamplerPub)
        with self.subTest("Initialization with `parameter_values` and without validation."):
            pub = POVMSamplerPub(
                circuit=qc, parameter_values=parameter_values, shots=1, povm=cs, validate=False
            )
            self.assertIsInstance(pub, POVMSamplerPub)

    def test_properties(self):
        """Test that the public attributes work correctly."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.ry(theta=theta, qubit=0)
        parameter_values = BindingsArray({theta: [0, 1, 2]})
        cs = ClassicalShadows(1)
        pub = POVMSamplerPub(
            circuit=qc, parameter_values=parameter_values, shots=13, povm=cs, validate=False
        )
        with self.subTest("Test ``circuit`` attribute."):
            self.assertIs(pub.circuit, qc)
        with self.subTest("Test ``parameter_values`` attribute."):
            self.assertIs(pub.parameter_values, parameter_values)
        with self.subTest("Test ``shots`` attribute."):
            self.assertEqual(pub.shots, 13)
        with self.subTest("Test ``povm`` attribute."):
            self.assertIs(pub.povm, cs)
        with self.subTest("Test ``shape`` attribute."):
            self.assertEqual(pub.shape, (3,))

    def test_coerce(self):
        """Test that the ``coerce`` method works correctly."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.ry(theta=theta, qubit=0)
        parameter_values = BindingsArray({theta: [0, 1, 2]})
        cs = ClassicalShadows(1)
        pub = POVMSamplerPub(
            circuit=qc, parameter_values=parameter_values, shots=13, povm=cs, validate=True
        )
        with self.subTest("Test to coerce pub into pub."):
            pub_test = POVMSamplerPub.coerce(pub)
            self.assertIs(pub_test, pub)
        pub = POVMSamplerPub(
            circuit=qc, parameter_values=parameter_values, shots=None, povm=cs, validate=False
        )
        with self.subTest("Test to coerce pub without `shots` specified into pub."):
            pub_test = POVMSamplerPub.coerce(pub, shots=5)
            self.assertIsNot(pub_test, pub)
            self.assertEqual(pub_test.shots, 5)
        with self.subTest("Test to coerce a `QuantumCircuit` pub-like object."):
            pub_test = POVMSamplerPub.coerce(pub=QuantumCircuit(1), shots=7, povm=cs)
            self.assertIsInstance(pub_test, POVMSamplerPub)
            self.assertEqual(pub_test.shots, 7)
            self.assertIs(pub_test.povm, cs)
        with self.subTest("Test to coerce a `QuantumCircuit` pub-like object."):
            pub_test_1 = POVMSamplerPub.coerce(pub=(qc, parameter_values), shots=9, povm=cs)
            pub_test_2 = POVMSamplerPub.coerce(pub=(qc, parameter_values, None), shots=9, povm=cs)
            pub_test_3 = POVMSamplerPub.coerce(
                pub=(qc, parameter_values, None, None), shots=9, povm=cs
            )
            self.assertDictEqual(pub_test_1.__dict__, pub_test_2.__dict__)
            self.assertDictEqual(pub_test_2.__dict__, pub_test_3.__dict__)
            pub_test = POVMSamplerPub.coerce(pub=(qc, [0, 1], 13, cs), shots=11)
            self.assertEqual(pub_test.shots, 13)
            self.assertIs(pub_test.povm, cs)
            pub_test = POVMSamplerPub.coerce(pub=(qc, [0, 1], 13, None), povm=cs)
            self.assertEqual(pub_test.shots, 13)
            self.assertIs(pub_test.povm, cs)
            pub_test = POVMSamplerPub.coerce(pub=(qc, [0, 1], None, cs), shots=11)
            self.assertEqual(pub_test.shots, 11)
            self.assertIs(pub_test.povm, cs)
            pub_test = POVMSamplerPub.coerce(pub=(qc, [0, 1], 13, cs), povm=ClassicalShadows(1))
            self.assertEqual(pub_test.shots, 13)
            self.assertIs(pub_test.povm, cs)

    def test_coerce_errors(self):
        """Test that the ``coerce`` method raises errors correctly."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.ry(theta=theta, qubit=0)
        parameter_values = BindingsArray({theta: [0, 1, 2]})
        cs = ClassicalShadows(1)
        with self.subTest("Test invalid type `shots` argument.") and self.assertRaises(TypeError):
            POVMSamplerPub.coerce(pub=(qc, parameter_values, None, cs), shots=1.2)
        with self.subTest("Test non-positive `shots` argument.") and self.assertRaises(ValueError):
            POVMSamplerPub.coerce(pub=(qc, parameter_values, None, cs), shots=0)
        with self.subTest(
            "Test missing `shots` argument when `pub_like` is a circuit."
        ) and self.assertRaises(ValueError):
            POVMSamplerPub.coerce(pub=qc, povm=cs)
        with self.subTest(
            "Test missing `povm` argument when `pub_like` is a circuit."
        ) and self.assertRaises(ValueError):
            POVMSamplerPub.coerce(pub=qc, shots=10)
        with self.subTest("Test too short pub-like tuple object.") and self.assertRaises(
            ValueError
        ):
            POVMSamplerPub.coerce(pub=tuple())
        with self.subTest("Test too long pub-like tuple object.") and self.assertRaises(ValueError):
            POVMSamplerPub.coerce(pub=(qc, parameter_values, None, cs, None))
        with self.subTest("Test type for pub-like object.") and self.assertRaises(TypeError):
            POVMSamplerPub.coerce(pub=[qc, parameter_values, 12, cs])

    def test_validate(self):
        """Test that the ``validate`` method works correctly."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.ry(theta=theta, qubit=0)
        parameter_values = BindingsArray({theta: [0, 1, 2]})
        cs = ClassicalShadows(1)
        with self.subTest("Test valid arguments."):
            pub = POVMSamplerPub(circuit=qc, parameter_values=parameter_values, shots=10, povm=cs)
            self.assertIs(pub.circuit, qc)
            self.assertIs(pub.parameter_values, parameter_values)
            self.assertEqual(pub.shots, 10)
            self.assertIs(pub.povm, cs)
        with self.subTest("Test invalid type for `QuantumCircuit` argument.") and self.assertRaises(
            TypeError
        ):
            POVMSamplerPub(
                circuit=qc.to_instruction(), parameter_values=parameter_values, shots=10, povm=cs
            )
        with self.subTest("Test `None` for `shots` argument.") and self.assertRaises(ValueError):
            pub = POVMSamplerPub(circuit=qc, parameter_values=parameter_values, shots=None, povm=cs)
        with self.subTest("Test invalid type `shots` argument.") and self.assertRaises(TypeError):
            POVMSamplerPub(circuit=qc, parameter_values=parameter_values, shots=1.2, povm=cs)
        with self.subTest("Test non-positive `shots` argument.") and self.assertRaises(ValueError):
            POVMSamplerPub(circuit=qc, parameter_values=parameter_values, shots=0, povm=cs)
        with self.subTest("Test too many parameters to bind.") and self.assertRaises(ValueError):
            POVMSamplerPub(
                circuit=qc,
                parameter_values=BindingsArray({theta: [0, 1], Parameter("phi"): [3, 4]}),
                shots=10,
                povm=cs,
            )
        with self.subTest(
            "Test zero parameters to bind when expecting some."
        ) and self.assertRaises(ValueError):
            POVMSamplerPub(circuit=qc, parameter_values=BindingsArray(), shots=10, povm=cs)
        with self.subTest("Test `None` for `povm` argument.") and self.assertRaises(ValueError):
            pub = POVMSamplerPub(circuit=qc, parameter_values=parameter_values, shots=10, povm=None)
        with self.subTest("Test invalid type for `povm` argument.") and self.assertRaises(
            TypeError
        ):
            pub = POVMSamplerPub(
                circuit=qc, parameter_values=parameter_values, shots=10, povm=cs.definition()
            )

    def test_to_sampler_pub(self):
        """Test that the ``to_sampler_pub`` method works correctly."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.ry(theta=theta, qubit=0)
        parameter_values = BindingsArray({theta: [0, 1, 2]})
        cs = ClassicalShadows(1)
        sampler_pub, metadata = POVMSamplerPub(
            circuit=qc, parameter_values=parameter_values, shots=10, povm=cs
        ).to_sampler_pub()
        self.assertIsInstance(sampler_pub, SamplerPub)
        self.assertIsInstance(metadata, POVMMetadata)
