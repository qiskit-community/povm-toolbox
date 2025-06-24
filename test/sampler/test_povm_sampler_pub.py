# (C) Copyright IBM 2024, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the POVMSamplerPub class."""

import pytest
from povm_toolbox.library import ClassicalShadows
from povm_toolbox.library.metadata import POVMMetadata
from povm_toolbox.sampler import POVMSamplerPub
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.sampler_pub import SamplerPub


class TestPOVMSamplerPub:
    """Tests for the ``POVMSamplerPub`` class."""

    def test_initialization(self, subtests):
        """Test that the ``__init__`` method works correctly."""
        qc = QuantumCircuit(1)
        cs = ClassicalShadows(1)
        with subtests.test("Initialization with validation."):
            pub = POVMSamplerPub(circuit=qc, parameter_values=None, shots=1, povm=cs, validate=True)
            assert isinstance(pub, POVMSamplerPub)

        theta = Parameter("theta")
        qc.ry(theta=theta, qubit=0)
        parameter_values = BindingsArray({theta: [0, 1, 2]})
        with subtests.test("Initialization with `parameter_values` and validation."):
            pub = POVMSamplerPub(
                circuit=qc, parameter_values=parameter_values, shots=1, povm=cs, validate=True
            )
            assert isinstance(pub, POVMSamplerPub)
        with subtests.test("Initialization with `parameter_values` and without validation."):
            pub = POVMSamplerPub(
                circuit=qc, parameter_values=parameter_values, shots=1, povm=cs, validate=False
            )
            assert isinstance(pub, POVMSamplerPub)

    def test_properties(self, subtests):
        """Test that the public attributes work correctly."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.ry(theta=theta, qubit=0)
        parameter_values = BindingsArray({theta: [0, 1, 2]})
        cs = ClassicalShadows(1)
        pub = POVMSamplerPub(
            circuit=qc, parameter_values=parameter_values, shots=13, povm=cs, validate=False
        )
        with subtests.test("Test ``circuit`` attribute."):
            assert pub.circuit is qc
        with subtests.test("Test ``parameter_values`` attribute."):
            assert pub.parameter_values is parameter_values
        with subtests.test("Test ``shots`` attribute."):
            assert pub.shots == 13
        with subtests.test("Test ``povm`` attribute."):
            assert pub.povm is cs
        with subtests.test("Test ``shape`` attribute."):
            assert pub.shape == (3,)

    def test_coerce(self, subtests):
        """Test that the ``coerce`` method works correctly."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.ry(theta=theta, qubit=0)
        parameter_values = BindingsArray({theta: [0, 1, 2]})
        cs = ClassicalShadows(1)
        pub = POVMSamplerPub(
            circuit=qc, parameter_values=parameter_values, shots=13, povm=cs, validate=True
        )
        with subtests.test("Test to coerce pub into pub."):
            pub_test = POVMSamplerPub.coerce(pub)
            assert pub_test is pub
        pub = POVMSamplerPub(
            circuit=qc, parameter_values=parameter_values, shots=None, povm=cs, validate=False
        )
        with subtests.test("Test to coerce pub without `shots` specified into pub."):
            pub_test = POVMSamplerPub.coerce(pub, shots=5)
            assert pub_test is not pub
            assert pub_test.shots == 5
        with subtests.test("Test to coerce a `QuantumCircuit` pub-like object."):
            pub_test = POVMSamplerPub.coerce(pub=QuantumCircuit(1), shots=7, povm=cs)
            assert isinstance(pub_test, POVMSamplerPub)
            assert pub_test.shots == 7
            assert pub_test.povm is cs
        with subtests.test("Test to coerce a `QuantumCircuit` pub-like object."):
            pub_test_1 = POVMSamplerPub.coerce(pub=(qc, parameter_values), shots=9, povm=cs)
            pub_test_2 = POVMSamplerPub.coerce(pub=(qc, parameter_values, None), shots=9, povm=cs)
            pub_test_3 = POVMSamplerPub.coerce(
                pub=(qc, parameter_values, None, None), shots=9, povm=cs
            )
            assert pub_test_1.__dict__ == pub_test_2.__dict__
            assert pub_test_2.__dict__ == pub_test_3.__dict__
            pub_test = POVMSamplerPub.coerce(pub=(qc, [0, 1], 13, cs), shots=11)
            assert pub_test.shots == 13
            assert pub_test.povm is cs
            pub_test = POVMSamplerPub.coerce(pub=(qc, [0, 1], 13, None), povm=cs)
            assert pub_test.shots == 13
            assert pub_test.povm is cs
            pub_test = POVMSamplerPub.coerce(pub=(qc, [0, 1], None, cs), shots=11)
            assert pub_test.shots == 11
            assert pub_test.povm is cs
            pub_test = POVMSamplerPub.coerce(pub=(qc, [0, 1], 13, cs), povm=ClassicalShadows(1))
            assert pub_test.shots == 13
            assert pub_test.povm is cs

    def test_coerce_errors(self, subtests):
        """Test that the ``coerce`` method raises errors correctly."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.ry(theta=theta, qubit=0)
        parameter_values = BindingsArray({theta: [0, 1, 2]})
        cs = ClassicalShadows(1)
        with subtests.test("Test invalid type `shots` argument.") and pytest.raises(TypeError):
            POVMSamplerPub.coerce(pub=(qc, parameter_values, None, cs), shots=1.2)
        with subtests.test("Test non-positive `shots` argument.") and pytest.raises(ValueError):
            POVMSamplerPub.coerce(pub=(qc, parameter_values, None, cs), shots=0)
        with subtests.test(
            "Test missing `shots` argument when `pub_like` is a circuit."
        ) and pytest.raises(ValueError):
            POVMSamplerPub.coerce(pub=qc, povm=cs)
        with subtests.test(
            "Test missing `povm` argument when `pub_like` is a circuit."
        ) and pytest.raises(ValueError):
            POVMSamplerPub.coerce(pub=qc, shots=10)
        with subtests.test("Test too short pub-like tuple object.") and pytest.raises(ValueError):
            POVMSamplerPub.coerce(pub=tuple())
        with subtests.test("Test too long pub-like tuple object.") and pytest.raises(ValueError):
            POVMSamplerPub.coerce(pub=(qc, parameter_values, None, cs, None))
        with subtests.test("Test type for pub-like object.") and pytest.raises(TypeError):
            POVMSamplerPub.coerce(pub=[qc, parameter_values, 12, cs])

    def test_validate(self, subtests):
        """Test that the ``validate`` method works correctly."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.ry(theta=theta, qubit=0)
        parameter_values = BindingsArray({theta: [0, 1, 2]})
        cs = ClassicalShadows(1)
        with subtests.test("Test valid arguments."):
            pub = POVMSamplerPub(circuit=qc, parameter_values=parameter_values, shots=10, povm=cs)
            assert pub.circuit is qc
            assert pub.parameter_values is parameter_values
            assert pub.shots == 10
            assert pub.povm is cs
        with subtests.test("Test invalid type for `QuantumCircuit` argument.") and pytest.raises(
            TypeError
        ):
            POVMSamplerPub(
                circuit=qc.to_instruction(), parameter_values=parameter_values, shots=10, povm=cs
            )
        with subtests.test("Test `None` for `shots` argument.") and pytest.raises(ValueError):
            pub = POVMSamplerPub(circuit=qc, parameter_values=parameter_values, shots=None, povm=cs)
        with subtests.test("Test invalid type `shots` argument.") and pytest.raises(TypeError):
            POVMSamplerPub(circuit=qc, parameter_values=parameter_values, shots=1.2, povm=cs)
        with subtests.test("Test non-positive `shots` argument.") and pytest.raises(ValueError):
            POVMSamplerPub(circuit=qc, parameter_values=parameter_values, shots=0, povm=cs)
        with subtests.test("Test too many parameters to bind.") and pytest.raises(ValueError):
            POVMSamplerPub(
                circuit=qc,
                parameter_values=BindingsArray({theta: [0, 1], Parameter("phi"): [3, 4]}),
                shots=10,
                povm=cs,
            )
        with subtests.test("Test zero parameters to bind when expecting some.") and pytest.raises(
            ValueError
        ):
            POVMSamplerPub(circuit=qc, parameter_values=BindingsArray(), shots=10, povm=cs)
        with subtests.test("Test `None` for `povm` argument.") and pytest.raises(ValueError):
            pub = POVMSamplerPub(circuit=qc, parameter_values=parameter_values, shots=10, povm=None)
        with subtests.test("Test invalid type for `povm` argument.") and pytest.raises(TypeError):
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
        assert isinstance(sampler_pub, SamplerPub)
        assert isinstance(metadata, POVMMetadata)
