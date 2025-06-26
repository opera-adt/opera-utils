# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial: create a time series stack from OPERA DISP-S1 data

# %% [markdown]
# ## 0. Environment Setup
#
# This will require an Earthdata username/password, either as a `~/.netrc` file:
#
# First install `opera-utils` with the optional disp dependency (or via conda-forge):

# %%
import opera_utils

# %%
HAS_LATEST_OPERA_UTILS = False
try:
    import opera_utils

    HAS_LATEST_OPERA_UTILS = opera_utils.__version__ >= "0.23.2"
except ImportError:
    pass
# if not HAS_LATEST_OPERA_UTILS:
# !pip install "opera-utils[disp]>=0.23.2"

# %% [markdown]
# Next, make sure we have our ~/.netrc file set up with earthdata credentials.
# If you don't have a profile, you can [register here](https://urs.earthdata.nasa.gov/users/new).
#
# ```
# $ cat ~/.netrc
# machine urs.earthdata.nasa.gov
#         login your_username
#         password your_password
# ```
#
# or set as environment variables `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`:
#
# ```bash
# export EARTHDATA_USERNAME="your_username"
# export EARTHDATA_PASSWORD="your_password"
# ```
#

# %%

# You can also set here to avoid creating the ~./netrc
# os.environ["EARTHDATA_USERNAME"] = "myusername"
# os.environ["EARTHDATA_PASSWORD"] = "password"

import opera_utils.credentials

try:
    opera_utils.credentials.get_earthdata_username_password(
        earthdata_username=None, earthdata_password=None
    )
except opera_utils.credentials.EarthdataLoginError:
    print("Please set up your earthdata credentials")
    raise

# %% [markdown]
# ## 1. Download a Frame Subset Over West Texas
#
# Now we'll run the download command to get a geographic subset of the DISP-S1 NetCDF files.
#
#
# Other options include:
#
# - `--wkt`: define a polygon to download based on a well-known text string
# - `--url-type S3`: download directly from S3 (if you are running this on an EC2 instance in the "us-west-2" region)
# - `--start-datetime` and `--end-datetime`: define a time range to download

# %% language="bash"
# opera-utils disp-s1-download \
#     --bbox -102.71 31.35 -102.6 31.45 \
#     --frame-id 20697 \
#     --output-dir subsets-west-texas \
#     --start-datetime 2021-01-01 \
#     --end-datetime 2024-01-01 \
#     --num-workers 4
#

# %% [markdown]
# ## 2. Reformat the data to remove the "moving reference"
#
# We'll create one 3D stack of data with all the layers in it that has taken care of the moving reference date.
# This lets us plot the continuous time series from the start without jumps in the time series.
# We're going to use the "border" referencing, since we're downloading a small land area surrounding one uplift.
# This will spatially reference each time series epoch to the average of the border pixels.
#
# To see all options for setting up this stack, run `opera-utils disp-s1-reformat --help`

# %% language="bash"
# opera-utils disp-s1-reformat \
#     --output-name stack-west-texas.zarr \
#     --input-files subsets-west-texas/OPERA_L3_DISP-S1*.nc \
#     --reference-method BORDER

# %% [markdown]
# ## 3. Plot the results
#
#

# %%
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Failed to import matplotlib: {e}")
    # !pip install matplotlib
import numpy as np
import xarray as xr

# %matplotlib inline

# %%
ds = xr.open_dataset("stack-west-texas.zarr", consolidated=False)
ds

# %%
fig, axes = plt.subplots(ncols=2, figsize=(10, 4))

# Plot the cumulative displacement
last_displacement = ds.displacement.isel(time=-1)
last_displacement.plot.imshow(ax=axes[0])

# Plot the time series at the maximum uplift
row, col = np.unravel_index(
    np.argmax(last_displacement.values), last_displacement.shape
)
ds.displacement.isel(y=row, x=col).plot(ax=axes[1], marker=".")
fig.tight_layout()
