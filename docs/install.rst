Installation Instructions
=========================

Let's see how to install the package. The first
thing to do is choose how you're going to run and install the
packages. There are two primary ways to do this:

- :ref:`Option 1`
- :ref:`Option 2`

Pre-Installation
^^^^^^^^^^^^^^^^

First, create a minimal environment with only Python installed in it. We recommend using `Python virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__.

.. code:: sh

    python3 -m venv /path/to/virtual/environment

Activate your new environment.

.. code:: sh

    source /path/to/virtual/environment/bin/activate

Note: If you are using Windows, use the following commands in PowerShell:

.. code:: pwsh

    python3 -m venv c:\path\to\virtual\environment
    c:\path\to\virtual\environment\Scripts\Activate.ps1


.. _Option 1:

Option 1: Install from PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most straightforward way to install the ``povm-toolbox`` package is via ``PyPI``.

.. code:: sh

    pip install 'povm-toolbox'


.. _Option 2:

Option 2: Install from Source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users who wish to develop in the repository or run the notebooks locally may want to install from source.

If so, the first step is to clone the ``povm-toolbox`` repository.

.. code:: sh

    git clone git@github.com:qiskit-community/povm-toolbox.git

Next, upgrade ``pip`` and enter the repository.

.. code:: sh

    pip install --upgrade pip
    cd povm-toolbox

The next step is to install ``povm-toolbox`` to the virtual environment. If you plan on running the notebooks, install the
notebook dependencies in order to run all the visualizations in the notebooks. If you plan on developing in the repository, you
may want to install the ``dev`` dependencies.

Adjust the options below to suit your needs.

.. code:: sh

    pip install tox notebook -e '.[notebook-dependencies,dev]'

If you installed the notebook dependencies, you can get started by running the notebooks in the docs.

.. code::

    cd docs/
    jupyter lab
