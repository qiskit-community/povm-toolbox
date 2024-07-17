############
POVM Toolbox
############

.. image:: _static/read-me-figure.jpg

This is a toolbox for working with positive operator-valued measures (POVMs).
It enables users to use POVMs for sampling the state of quantum circuits (see
also :mod:`povm_toolbox.sampler`) and compute expectation values of any
observable of interest (see also :mod:`povm_toolbox.post_processor`).
The toolbox includes a library of pre-defined POVMs (see
:mod:`povm_toolbox.library`) but also allows users to define their own POVMs
(see :mod:`povm_toolbox.quantum_info`).

In this documentation you can find a number of resources including:

- `explanations <explanations/index.html>`_ to learn more about POVMs
- how to get started with coding using one of the `tutorials <tutorials/index.html>`_
- dive into more specific features with the `how-to guides <how_tos/index.html>`_
- and, of course, look up specific details of the `API <apidocs/povm_toolbox.html>`_

Installation
------------

You can install this code via pip:

.. code-block:: bash

   git clone git@github.com:qiskit-community/povm-toolbox.git
   cd povm-toolbox
   pip install .

Make sure that you have the correct Python environment active, into which you
want to install this code, before running the above.

Citation
--------

If you use this project, please cite the following reference:

    Laurin E. Fischer, Timothée Dao, Ivano Tavernelli, and Francesco Tacchino;
    "Dual-frame optimization for informationally complete quantum measurements";
    Phys. Rev. A 109, 062415;
    DOI: https://doi.org/10.1103/PhysRevA.109.062415


.. toctree::
  :hidden:

   Documentation Home <self>
   Tutorials <tutorials/index>
   How-To Guides <how_tos/index>
   Explanations <explanations/index>
   API Reference <apidocs/povm_toolbox>
   GitHub <https://github.com/qiskit-community/povm-toolbox>
   Release Notes <release-notes>
