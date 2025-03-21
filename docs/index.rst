############
POVM Toolbox
############

.. image:: _static/read-me-figure.jpg

This is a toolbox for working with positive operator-valued measures (POVMs).
It enables users to use POVMs for sampling the state of quantum circuits (see
also :mod:`povm_toolbox.sampler`) and compute expectation values of any
observable of interest (see also :mod:`povm_toolbox.post_processor`).
The toolbox includes a library of pre-defined POVMs (see
:mod:`povm_toolbox.library`) which provide ready-to-go POVM circuit definitions.
You can also implement your own POVM circuits by following the provided
interface.
Additionally, you can work with POVMs on a quantum-informational theoretical
footing (see :mod:`povm_toolbox.quantum_info`).

In this documentation you can find a number of resources including:

- `explanations <explanations/index.html>`_ to learn more about POVMs
- how to get started with coding using one of the `tutorials <tutorials/index.html>`_
- dive into more specific features with the `how-to guides <how_tos/index.html>`_
- and, of course, look up specific details of the `API <apidocs/povm_toolbox.html>`_

Documentation
-------------

All documentation is available `here <https://qiskit-community.github.io/povm-toolbox/>`_.

Installation
------------

We encourage installing this package via ``pip``, when possible:

.. code-block:: bash

   pip install 'povm-toolbox'


For more installation information refer to the `installation instructions <install.rst>`_ in the documentation.

Deprecation Policy
------------------

We follow `semantic versioning <https://semver.org/>`_ and are guided by the principles in
`Qiskit's deprecation policy <https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md>`_.
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
`release notes <https://qiskit-community.github.io/povm-toolbox/release-notes.html>`_.

Contributing
------------

The source code is available `on GitHub <https://github.com/qiskit-community/povm-toolbox>`_.

The developer guide is located at `CONTRIBUTING.md <https://github.com/qiskit-community/povm-toolbox/blob/main/CONTRIBUTING.md>`_
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/qiskit-community/povm-toolbox/issues/new/choose>`_ for tracking requests and bugs.


Citation
--------

If you use this project, please cite the following reference:

    Laurin E. Fischer, Timothée Dao, Ivano Tavernelli, and Francesco Tacchino;
    "Dual-frame optimization for informationally complete quantum measurements";
    Phys. Rev. A 109, 062415;
    DOI: https://doi.org/10.1103/PhysRevA.109.062415

License
-------

`Apache License 2.0 <https://github.com/qiskit-community/povm-toolbox/blob/main/LICENSE.txt>`_


.. toctree::
  :hidden:

   Documentation Home <self>
   Installation Instructions <install>
   Tutorials <tutorials/index>
   How-To Guides <how_tos/index>
   Explanations <explanations/index>
   API Reference <apidocs/povm_toolbox>
   GitHub <https://github.com/qiskit-community/povm-toolbox>
   Release Notes <release-notes>
