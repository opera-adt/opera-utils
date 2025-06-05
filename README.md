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
# if mamba is not already installed: conda install -n base mamba
mamba install -c conda-forge opera-utils
```

(Note: [using `mamba`](https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install) is recommended for conda-forge packages, but miniconda can also be used.)

```bash
pip install opera-utils
```

The `pip` dependency list for the base package is smaller; the optional depencies may be added with

```bash
pip install opera-utils[geopandas] # for just geopandas data frame support
pip install opera-utils[disp] # For remote access to DISP files
pip install opera-utils[all]  # all optional dependencies
```

## Example Usage

### Parsing Sentinel-1 Burst IDs

```python
import opera_utils
print(opera_utils.get_burst_id("OPERA_L2_CSLC-S1_T087-185683-IW2_20230322T161649Z_20240504T185235Z_S1A_VV_v1.1.h5"))
't087_185683_iw2'
```

### Get DISP-S1 Frame metadata

```python
In [4]: opera_utils.get_frame_to_burst_mapping(11114)
Out[4]:
{'epsg': 32610,
 'is_land': True,
 'is_north_america': True,
 'xmin': 546450,
 'ymin': 4204110,
 'xmax': 833790,
 'ymax': 4409070,
 'burst_ids': ['t042_088905_iw1',
  ...
 ]
  ```

To just get the burst IDs for a Frame:
```python
In [3]: opera_utils.get_burst_ids_for_frame(11114)
Out[3]:
['t042_088905_iw1',
 't042_088905_iw2',
 ...
 't042_088913_iw2',
 't042_088913_iw3']
 ```

## DISP-S1 Usage examples

```python
from opera_utils.disp import search, reader, DispProductStack

# Search for DISP-S1 products on CMR
products = search.get_products(frame_id=9154)

# Create the product stack for parsing frame metadata
stack = DispProductStack(products)

# Read a a lon/lat box
data = reader.read_stack_lonlat(
    stack,
    lons=slice(-120.45, -120.3),
    lats=slice(34.07, 34.01)
)
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
```

3. Install the source in editable mode:
```bash
mamba activate my-opera-env
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
pytest
```
