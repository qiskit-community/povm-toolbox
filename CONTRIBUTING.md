# Contributing

This document explains the various things you should know in order to be able to
contribute to this repository.
When contributing to this project you will need to sign this [contribution
license agreement](https://cla-assistant.io/Qiskit/qiskit).

## Installation

We suggest that you use an
[editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)
for the installation into your development Python environment. This simplifies
your development because you will not have to re-install the package after every
edit to the code base.
```
git clone git@github.com:qiskit-community/povm-toolbox.git
cd povm-toolbox
pip install -e ".[dev]"
```

Make sure that you have the correct Python environment active, into which you
want to install this code, before running the above.

The above will install the code of this repository along with all the
development requirements listed in `pyproject.toml`.

## Tooling and Testing

We use [`tox`](https://tox.wiki/en/latest/) for automating and standardizing the
testing of our code.

Within `tox.ini` we have specified a number of environments to test different
aspects.

### Linting and Formatting

⚠️  This will use Python 3.10, so ensure that you have this installed.

Use the `lint` environment to check your code quality:
```
tox -e lint
```
This will run the following tools:
- [`ruff`](https://docs.astral.sh/ruff/): to check our code formatting and some
  basic linting rules
- [`nbqa`](https://nbqa.readthedocs.io/en/latest/index.html): to check our
  jupyter notebooks
- [`mypy`](https://mypy.readthedocs.io/en/stable/): to check our type hints
- [`pylint`](https://pylint.readthedocs.io/en/stable/): for additional linting
  rules

Note, that `tox` will stop as soon as the first linter fails. So after fixing
one, be sure to re-run the linting check to see if the other tools will pass.

Some of these tools provide "automatic" fixing for their rules which you can
apply to your code by running:
```
tox -e style
```

If this does not fix all of the linter complains, you will need to fix them by
hand. This is often easy enough upon reading the error messages produces by the
complaining linter.

### Testing

You can run the suite of unittests against different Python versions (assuming
that you have these installed on your system) using:
```
tox -e py311
```
This will run the tests against Python 3.11.
In CI we test against all supported Python versions: 3.9, 3.10, 3.11, and 3.12.

Under the hood, we use [`pytest`](https://docs.pytest.org/en/stable/) for
finding and executing the unittests. You can pass command-line arguments to it,
for example, for selecting a subset of tests to run, like so:
```
tox -e py311 -- <path/to/test_file.py>
```

#### Coverage

We strive towards complete coverage of our unittest suite. This means, we want
our test suite to execute 100% of our code. We use
[`coverage`](https://coverage.readthedocs.io/en/7.4.3/) integrated with `pytest`
to measure this and our CI will complain when this criterion is not met.

You can test this locally with Python 3.10 using the following:
```
tox -e coverage
```

### Notebooks

We also test the execution of our Jupyter notebooks using
[`nbmake`](https://github.com/treebeardtech/nbmake):
```
tox -e notebook
```
