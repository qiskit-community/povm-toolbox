# POVMs

<p align="center">
  <img src="/docs/_static/read-me-figure.jpg" height="350">
</p>

This is a toolbox for working with positive operator-valued measures (POVMs).
It enables users to use POVMs for sampling the state of quantum circuits (see
also `povm_toolbox.sampler`) and compute expectation values of any observable of
interest (see also `povm_toolbox.post_processor`).
The toolbox includes a library of pre-defined POVMs (see `povm_toolbox.library`)
which provide ready-to-go POVM circuit definitions. You can also implement your
own POVM circuits by following the provided interface.
Additionally, you can work with POVMs on a quantum-informational theoretical
footing (see `povm_toolbox.quantum_info`).

## Installation

Make sure that you have the correct Python environment active, into which you
want to install this code, before running the below.

You can install this code via pip:
```
pip install povm-toolbox
```

Alternatively, you can install it from source:
```
git clone git@github.com:qiskit-community/povm-toolbox.git
cd povm-toolbox
pip install -e .
```
This performs an editable install to simplify code development.

If you intend to develop on this code, you should consider reading the
[contributing guide](CONTRIBUTING.md).

## Documentation

You can find the documentation hosted
[here](https://qiskit-community.github.io/povm-toolbox/).

## Citation

If you use this project, please cite the following reference:

> Laurin E. Fischer, Timothée Dao, Ivano Tavernelli, and Francesco Tacchino
> "Dual-frame optimization for informationally complete quantum measurements"
> Phys. Rev. A 109, 062415
> DOI: https://doi.org/10.1103/PhysRevA.109.062415
