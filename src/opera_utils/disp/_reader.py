from __future__ import annotations

import multiprocessing as mp
from enum import Enum
from functools import partial

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ._product import DispProduct, DispProductStack
from ._rebase import rebase_timeseries
from ._remote import open_h5


class ReferenceMethod(str, Enum):
    """Method to use for spatially referencing displacement results."""

    none = "none"
    point = "point"
    median = "median"
    border = "border"

    def __str__(self) -> str:
        return self.value


def read_lonlat(
    product: DispProduct,
    lon_slice: slice | float,
    lat_slice: slice | float,
    dset: str = "displacement",
) -> np.ndarray:
    """Read data from a single product for a longitude/latitude box.

    Parameters
    ----------
    product : DispProduct
        The product to read from
    lon_slice : slice or float
        Longitude slice (min, max), or single value
    lat_slice : slice or float
        Latitude slice (min, max), or single value
    dset : str, default="displacement"
        Dataset name to read from the HDF5 file

    Returns
    -------
    np.ndarray
        Data within the specified longitude/latitude box

    """
    rows, cols = _get_rows_cols(lon_slice, lat_slice, product)

    # Read the data
    with open_h5(product) as hf:
        dset_obj = hf[dset]
        return dset_obj[rows, cols]


def read_stack_lonlat(
    stack: DispProductStack,
    lons: float | slice,
    lats: float | slice,
    reference_method: ReferenceMethod = ReferenceMethod.none,
    ref_lon: float | None = None,
    ref_lat: float | None = None,
    max_workers: int | None = None,
    dset: str = "displacement",
) -> tuple[np.ndarray, dict[str, str | float]]:
    """Process a stack using the lon/lat box method with optional multiprocessing.

    Parameters
    ----------
    stack : DispProductStack
        Stack of products to read from
    lons : slice or float
        Longitude slice or single value
    lats : slice or float
        Latitude slice or single value
    dset : str, default="displacement"
        Dataset name to read from the HDF5 file
    reference_method : ReferenceMethod or str
        Method for referencing:
        - "none": No referencing
        - "point": Reference to specified ref_lon/ref_lat
        - "median": Reference to median of each date
        - "border": Reference to median value of data border
    ref_lon : float | None, optional
        Longitude of the reference point
    ref_lat : float | None, optional
        Latitude of the reference point
    max_workers : int, optional
        Number of processes to use for parallel reading.
        If None, uses the number of CPU cores.

    Returns
    -------
    referenced_data : np.ndarray
        3D array with dimensions (time, lat, lon) containing the requested data
    attrs : dict[str, str|float]
        Attributes for the dataset

    """
    if ref_lon is not None and ref_lat is not None:
        reference_method = ReferenceMethod.point

    if reference_method == ReferenceMethod.point and (
        ref_lon is None or ref_lat is None
    ):
        msg = "ref_lon and ref_lat must be provided when using point referencing"
        raise ValueError(msg)
    elif reference_method not in set(ReferenceMethod):
        msg = f"Unknown reference_method: {reference_method}"
        raise ValueError(msg)

    # Create a partial function with fixed parameters
    read_func = partial(read_lonlat, lon_slice=lons, lat_slice=lats, dset=dset)

    # Use multiple processes, or default to CPU count (capped to 20)
    max_workers = max_workers or min(20, max(1, mp.cpu_count()))

    # Create a process pool, using spawn to avoid problems with h5py
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max_workers) as pool:
        results = list(
            tqdm(
                pool.imap(read_func, stack.products),
                total=len(stack.products),
                desc="Reading products",
            )
        )

    with open_h5(stack.products[0]) as hf:
        # pop off internal netcdf4 attributes
        attrs = dict(hf[dset].attrs)
        for k in [
            "_Netcdf4Coordinates",
            "_FillValue",
            "_Netcdf4Dimid",
            "DIMENSION_LIST",
        ]:
            attrs.pop(k, None)
        attrs["reference_datetime"] = stack.reference_dates[0].isoformat()
        attrs["reference_method"] = reference_method.value
        attrs["reference_lon"] = ref_lon or "None"
        attrs["reference_lat"] = ref_lat or "None"

    # Stack the results and adjust reference (if needed)
    unreffed_data = np.stack(results)
    if "displacement" not in dset:
        return unreffed_data

    # Only the displacement/short_wavelength_displacement needs extra work
    rebased_stack = rebase_timeseries(unreffed_data, stack.reference_dates)

    # Apply referencing if needed
    if reference_method == ReferenceMethod.none:
        return rebased_stack, attrs

    if reference_method == ReferenceMethod.point:
        read_func = partial(
            read_lonlat,
            lon_slice=slice(ref_lon, ref_lon),
            lat_slice=slice(ref_lat, ref_lat),
            dset=dset,
        )
        with ctx.Pool(processes=max_workers) as pool:
            ref_value_list = list(
                tqdm(
                    pool.imap(read_func, stack.products),
                    total=len(stack.products),
                    desc="Reading reference points",
                )
            )
        ref_values = np.stack(ref_value_list)
    elif reference_method == ReferenceMethod.median:
        ref_values = np.nanmedian(rebased_stack, axis=(1, 2), keepdims=True)
    elif reference_method == ReferenceMethod.border:
        ref_values = _get_border(rebased_stack)
    else:
        msg = f"Unknown {reference_method = }"
        raise ValueError(msg)

    referenced_data = rebased_stack - ref_values
    return referenced_data, attrs


def _get_rows_cols(
    lon_slice: float | slice, lat_slice: float | slice, product: DispProduct
) -> tuple[slice, slice]:
    if isinstance(lon_slice, float):
        lon_slice = slice(lon_slice, lon_slice)
    if isinstance(lat_slice, float):
        lat_slice = slice(lat_slice, lat_slice)

    # Convert lon/lat to row/col
    lon_left, lon_right = lon_slice.start, lon_slice.stop
    lat_top, lat_bottom = lat_slice.start, lat_slice.stop
    if lat_bottom > lat_top:
        lat_top, lat_bottom = lat_bottom, lat_top
    # Get row/col indices
    row_start, col_start = product.lonlat_to_rowcol(lon_left, lat_top)
    row_stop, col_stop = product.lonlat_to_rowcol(lon_right, lat_bottom)

    # Handle edge cases - ensure we have at least a 1x1 window
    if col_stop < col_start:
        msg = f"Invalid column range: {col_start}, {col_stop}"
        raise ValueError(msg)
    elif col_stop == col_start:
        col_stop += 1

    if row_stop < row_start:
        msg = f"Invalid row range: {row_start}, {row_stop}"
        raise ValueError(msg)
    elif row_stop == row_start:
        row_stop += 1
    return slice(row_start, row_stop), slice(col_start, col_stop)


def _get_border(data_arrays: NDArray[np.floating]) -> NDArray[np.floating]:
    top_row = data_arrays[:, 0, :]
    bottom_row = data_arrays[:, -1, :]
    left_col = data_arrays[:, :, 0]
    right_col = data_arrays[:, :, -1]
    all_pixels = np.hstack([top_row, bottom_row, left_col, right_col])
    return np.nanmedian(all_pixels, axis=1)[:, np.newaxis, np.newaxis]
