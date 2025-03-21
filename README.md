<!-- SHIELDS -->
<div align="left">

  [![Release](https://img.shields.io/pypi/v/povm-toolbox.svg?label=Release)](https://github.com/qiskit-community/povm-toolbox/releases)
  ![Platform](https://img.shields.io/badge/%F0%9F%92%BB%20Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
  [![Python](https://img.shields.io/pypi/pyversions/povm-toolbox?label=Python&logo=python)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit%20-%20%3E%3D1.4%20-%20%236133BD?logo=Qiskit)](https://github.com/Qiskit/qiskit)
<br />
  [![Docs (stable)](https://img.shields.io/badge/%F0%9F%93%84%20Docs-stable-blue.svg)](https://qiskit-community.github.io/povm-toolbox/)
  [![License](https://img.shields.io/github/license/qiskit-community/povm-toolbox?label=License)](LICENSE.txt)
  [![Downloads](https://img.shields.io/pypi/dm/povm-toolbox.svg?label=Downloads)](https://pypi.org/project/povm-toolbox/)
  [![Tests](https://github.com/qiskit-community/povm-toolbox/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/qiskit-community/povm-toolbox/actions/workflows/test_latest_versions.yml)
  [![Coverage](https://coveralls.io/repos/github/qiskit-community/povm-toolbox/badge.svg?branch=main)](https://coveralls.io/github/qiskit-community/povm-toolbox?branch=main)
</div>

# POVMs

### Table of contents

* [About](#about)
* [Documentation](#documentation)
* [Installation](#installation)
* [Deprecation Policy](#deprecation-policy)
* [Contributing](#contributing)
* [Citation](#citation)
* [License](#license)

----------------------------------------------------------------------------------------------------

### About

![overview](https://raw.githubusercontent.com/qiskit-community/povm-toolbox/main/docs/_static/read-me-figure.jpg)

This is a toolbox for working with positive operator-valued measures (POVMs).
It enables users to use POVMs for sampling the state of quantum circuits (see
also `povm_toolbox.sampler`) and compute expectation values of any observable of
interest (see also `povm_toolbox.post_processor`).
The toolbox includes a library of pre-defined POVMs (see `povm_toolbox.library`)
which provide ready-to-go POVM circuit definitions. You can also implement your
own POVM circuits by following the provided interface.
Additionally, you can work with POVMs on a quantum-informational theoretical
footing (see `povm_toolbox.quantum_info`).


----------------------------------------------------------------------------------------------------

### Documentation

All documentation is available at https://qiskit-community.github.io/povm-toolbox/.

----------------------------------------------------------------------------------------------------

### Installation

We encourage installing this package via `pip`, when possible:

```bash
pip install 'povm-toolbox'
```

For more installation information refer to these [installation instructions](docs/install.rst).

----------------------------------------------------------------------------------------------------

### Deprecation Policy

We follow [semantic versioning](https://semver.org/) and are guided by the principles in
[Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md).
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
[release notes](https://qiskit-community.github.io/povm-toolbox/release-notes.html).

----------------------------------------------------------------------------------------------------

### Contributing

The source code is available [on GitHub](https://github.com/qiskit-community/povm-toolbox).

The developer guide is located at [CONTRIBUTING.md](https://github.com/qiskit-community/povm-toolbox/blob/main/CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

We use [GitHub issues](https://github.com/qiskit-community/povm-toolbox/issues/new/choose) for tracking requests and bugs.

----------------------------------------------------------------------------------------------------

### Citation

If you use this project, please cite the following reference:

> Laurin E. Fischer, Timothée Dao, Ivano Tavernelli, and Francesco Tacchino
> "Dual-frame optimization for informationally complete quantum measurements"
> Phys. Rev. A 109, 062415
> DOI: https://doi.org/10.1103/PhysRevA.109.062415

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)
