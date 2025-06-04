"""Module to concatenate displacement tiles into a CF-compliant virtual dataset.

Note that in-place modification of the virtual file will change the original
underlying data.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np


def create_virtual_stack(
    input_files: Sequence[Path | str],
    output: Path | str,
    dataset_names: Sequence[str] | None = None,
    drop_vars: Sequence[str] | None = None,
) -> None:
    """Concatenate single-date NetCDF displacement files into a stack.

    Uses the virtual HDF5 stack along the time dimension.
    See https://docs.h5py.org/en/stable/vds.html for more info.

    Parameters
    ----------
    input_files : list[pathlib.Path]
        Source NetCDF/CF files.
    output : pathlib.Path
        Path of the VDS file to create. Suffix may be “.nc” or “.h5”.
    dataset_names : Sequence[str], optional
        Names of the 2-D variable to concatenate and include in `output`.
        If None provided, creates for all 2D datasets in `input_files`.
    drop_vars : Sequence[str], optional
        Names of variables to drop from the dataset before saving.
        Useful if you are saving all but a few variables from `input_files`.

    """
    if not input_files:
        msg = "No input files provided."
        raise ValueError(msg)

    if dataset_names is None:
        dataset_names = _get_2d_datasets(input_files[0])

    if drop_vars is None:
        drop_vars = []

    data_vars = [name for name in dataset_names if name not in drop_vars]

    # Inspect the first file to determine shapes, dtype, & metadata
    with h5py.File(input_files[0], "r") as hf0:
        ny, nx = hf0["displacement"].shape
        name_to_dtype = {name: hf0[name].dtype for name in dataset_names}

    n_time = len(input_files)

    with h5py.File(output, "r+", libver="latest") as hf:
        time_ds = hf["time"]
        y_ds = hf["y"]
        x_ds = hf["x"]
        for dataset_name in data_vars:
            dtype = name_to_dtype[dataset_name]
            layout = h5py.VirtualLayout(shape=(n_time, ny, nx), dtype=dtype)
            for k, src in enumerate(map(str, input_files)):
                layout[k, :, :] = h5py.VirtualSource(src, dataset_name, shape=(ny, nx))
            dset = hf.create_virtual_dataset(dataset_name, layout, fillvalue=np.nan)

            # Add attributes
            dset.attrs.update(hf[dataset_name].attrs)

            # Attach dimension scales → proper CF & GDAL recognition
            dset.dims[0].attach_scale(time_ds)
            dset.dims[1].attach_scale(y_ds)
            dset.dims[2].attach_scale(x_ds)
            dset.attrs["grid_mapping"] = "spatial_ref"


def _get_2d_datasets(filename: Path | str) -> list[str]:
    names = []

    def _add_name(name):
        if isinstance(hf[name], h5py.Dataset) and hf[name].ndim == 2:
            names.append(name)

    with h5py.File(filename) as hf:
        hf.visit(_add_name)
    return names
