# opera-utils

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]

[![Conda-Forge][conda-badge]][conda-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/opera-adt/opera-utils/actions/workflows/ci.yml/badge.svg
[actions-link]:             https://github.com/opera-adt/opera-utils/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/opera-utils
[conda-link]:               https://github.com/conda-forge/opera-utils-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/opera-adt/opera-utils/discussions
[pypi-link]:                https://pypi.org/project/opera-utils/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/opera-utils
[pypi-version]:             https://img.shields.io/pypi/v/opera-utils

<!-- prettier-ignore-end -->

## Install

The `opera_utils` package is available on PyPI and conda-forge:

```bash
pip install opera-utils
```
```bash
# if mamba is not already installed: conda install -n base mamba
mamba install -c conda-forge opera-utils
```
(Note: [using `mamba`](https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install) is recommended for conda-forge packages, but miniconda can also be used.)

While not required for all, some utilities use the GDAL package, which can be installed most easily on conda-forge:
```bash
mamba env update --file environment-geo.yml
```

## Setup for Developers

To contribute to the development of `opera-utils`, you can fork the repository and install the package in development mode.
We encourage new features to be developed on a new branch of your fork, and then submitted as a pull request to the main repository.

To install locally,

1. Download source code:
```bash
git clone https://github.com/opera-adt/opera-utils.git && cd opera-utils
```
2. Install dependencies:
```bash
mamba env create --name my-opera-env --file environment.yml
mamba install gdal
```

3. Install the source in editable mode:
```bash
mamba activate my-opera-env
python -m pip install -e .
```

The extra packages required for testing and building the documentation can be installed:
```bash
# Run "pip install -e" to install with extra development requirements
python -m pip install -e ".[docs,test]"
```

We use [`pre-commit`](https://pre-commit.com/) to automatically run linting and formatting:
```bash
# Get pre-commit hooks so that linting/formatting is done automatically
pre-commit install
```
This will set up the linters and formatters to run on any staged files before you commit them.

After making functional changes, you can rerun the existing tests and any new ones you have added using:
```bash
python -m pytest
```
