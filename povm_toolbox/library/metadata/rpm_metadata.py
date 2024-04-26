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

import dataclasses
from dataclasses import dataclass

import numpy as np

from .povm_metadata import POVMMetadata


@dataclass(repr=False)
class RPMMetadata(POVMMetadata):
    """TODO."""

    pvm_keys: np.ndarray
    """Shape of `pvm_keys` is assumed to be ``(*pv.shape, num_batches, n_qubit)``,
    where ``pv`` is the bindings array provided by the user to run with the
    parametrized quantum circuit supplied in the :meth:`.POVMSampler.run` method.
    """

    def __repr__(self):
        """Implement the default ``__repr__`` method to avoid printing large ``numpy.array``.

        The attribute ``pvm_keys`` will typically be a large ``numpy.ndarray`` object.
        With the default ``dataclass.__repr__``, it would be entirely printed. As this
        is recursive, the full array would be printed when printing the :class:`.PrimitiveResult`
        object returned by the :meth:`.POVMSampler.run` method. The redefinition here avoids this.
        """
        lst_fields = []
        for field in dataclasses.fields(self):
            f_name = field.name
            f_val = getattr(self, field.name)
            f_val = (
                f_val
                if not isinstance(f_val, np.ndarray)
                else f'np.ndarray<{",".join(map(str, f_val.shape))}>'
            )
            lst_fields.append((f_name, f_val))
        f_repr = ", ".join(f"{name}={value}" for name, value in lst_fields)
        return f"{self.__class__.__name__}({f_repr})"
