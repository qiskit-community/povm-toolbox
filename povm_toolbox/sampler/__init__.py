# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A module of sampling tools for POVMs.

.. currentmodule:: povm_toolbox.sampler

At their core, POVMs are used in combination with sampling the state of a quantum circuit.
In Qiskit, this functionality is provided via the
`Sampler primitive <https://docs.quantum.ibm.com/guides/primitives>`_.

To this end, this module provides a number of tools for sampling the state of
:class:`~qiskit.circuit.QuantumCircuit` objects using a :class:`.POVMImplementation`.

The Sampler
-----------

As a consumer of this library, you will be mostly concerned with the :class:`.POVMSampler` class as
your entry point for submitting POVM sampling jobs.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   POVMSampler

Additional Classes
------------------

However, this module also contains these additional classes which you may come in contact with while
working with the in- and outputs of the :class:`.POVMSampler`.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   POVMSamplerPub
   POVMSamplerJob
   POVMPubResult

.. autoclass:: POVMSamplerPubLike
   :members:
"""

from .povm_pub_result import POVMPubResult
from .povm_sampler import POVMSampler
from .povm_sampler_job import POVMSamplerJob
from .povm_sampler_pub import POVMSamplerPub, POVMSamplerPubLike

__all__ = [
    "POVMSampler",
    "POVMSamplerJob",
    "POVMSamplerPub",
    "POVMSamplerPubLike",
    "POVMPubResult",
]
