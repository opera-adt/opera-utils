from __future__ import annotations

import warnings

import dask.array as da
import numpy as np

from ._product import DispProduct, DispProductStack
from ._remote import open_h5

try:
    import xarray as xr
    from dask import delayed
except ImportError:
    warnings.warn(
        "xarray and dask are required for xarray functionality.", stacklevel=2
    )


def stack_to_dataarray(
    stack: DispProductStack,
    var: str = "displacement",
    chunks: tuple[int, int] = (1024, 1024),
    dtype=np.float32,
) -> xr.DataArray:
    """Convert a `DispProductStack` to an xarray.DataArray backed by a dask array.

    Parameters
    ----------
    stack : DispProductStack
        Ordered list of DISP products from one frame.
    var : str
        Name of the dataset inside each NetCDF file.
        Default is "displacement".
    chunks : (int, int)
        Chunk size for (y, x) dimensions.
        Default is (1024, 1024).
        Note that `time` is chunked as 1 file per chunk.
    dtype : NumPy dtype, default float32
        Expected dtype of `var` in the files.

    Returns
    -------
    xr.DataArray
        Lazy 3-D array with dims (time, y, x).

    """
    # The shape is known from the frame database
    ny, nx = stack.shape[1:]

    # Build slices that cover the 2-D grid with (chunks[0] x chunks[1]) blocks
    y_slices = [slice(i, min(i + chunks[0], ny)) for i in range(0, ny, chunks[0])]
    x_slices = [slice(j, min(j + chunks[1], nx)) for j in range(0, nx, chunks[1])]

    # Construct one delayed array per file
    delayed_rows = []
    for prod in stack.products:
        # For each file build a 2-D dask array of blocks
        delayed_blocks = [
            [
                da.from_delayed(
                    delayed(_load_block)(prod, var, (rs, cs)),
                    shape=(rs.stop - rs.start, cs.stop - cs.start),
                    dtype=dtype,
                )
                for cs in x_slices
            ]
            for rs in y_slices
        ]

        file_array = da.block(delayed_blocks)
        delayed_rows.append(file_array)

    t = stack.secondary_dates
    # Note: the coordinates are also known from the frame database
    y = stack.y  # 1-D north-to-south
    x = stack.x  # 1-D west-to-east

    data = da.stack(delayed_rows, axis=0)
    da_out = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={
            "time": ("time", t),
            "y": ("y", y),
            "x": ("x", x),
        },
        name=var,
        attrs={
            "crs": f"EPSG:{stack.epsg}",
            "transform": stack.products[0].transform,
            # TODO: this might be some easy way to make this `rioxarray` compatible
        },
    )
    return da_out


def _load_block(
    product: DispProduct, var: str, block_slices: tuple[slice, slice]
) -> np.ndarray:
    """Read a (row, col) window from one file and returns a NumPy 2-D array.

    Low-level helper called by dask.
    """
    rslice, cslice = block_slices
    with open_h5(product) as hf:
        return hf[var][rslice, cslice]
