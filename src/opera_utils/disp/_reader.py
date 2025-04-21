from __future__ import annotations

import multiprocessing as mp
from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from functools import partial

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ._product import DispProduct, DispProductStack
from ._remote import open_h5


class ReferenceMethod(Enum):
    """Method to use for spatially referencing displacement results."""

    NONE = "none"
    POINT = "point"
    MEDIAN = "median"
    BORDER = "border"


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
    reference_method: ReferenceMethod = ReferenceMethod.NONE,
    # ref_lon: Optional[float] = None,
    # ref_lat: Optional[float] = None,
    # ref_pixel: Optional[tuple[int, int]] = None,
    max_workers: int | None = None,
    dset: str = "displacement",
) -> np.ndarray:
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
    max_workers : int, optional
        Number of processes to use for parallel reading.
        If None, uses the number of CPU cores.

    Returns
    -------
    np.ndarray
        3D array with dimensions (time, lat, lon) containing the requested data
    """
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

    # Stack the results and adjust reference (if needed)
    unreffed_data = np.stack(results)
    if "displacement" not in dset:
        return unreffed_data

    # Only the displacement/short_wavelength_displacement needs extra work
    rebased_stack = rebase_timeseries(unreffed_data, stack.reference_dates)

    # Apply referencing if needed
    if reference_method == "none":
        referenced_data = rebased_stack
    elif reference_method == "median":
        ref_values = np.nanmedian(rebased_stack, axis=(1, 2), keepdims=True)
        referenced_data = rebased_stack - ref_values
    elif reference_method == "point":
        # TODO: figure out where in the stack it is
        raise NotImplementedError()
    elif reference_method == "border":
        referenced_data = _get_border(rebased_stack)
    else:
        raise ValueError(f"Unknown {reference_method = }")

    return referenced_data


def _get_rows_cols(
    lon_slice: float | slice, lat_slice: float | slice, product: DispProduct
):
    if isinstance(lon_slice, float):
        lon_slice = slice(lon_slice, lon_slice)
    if isinstance(lat_slice, float):
        lat_slice = slice(lat_slice, lat_slice)

    # Convert lon/lat to row/col
    lon_start, lon_stop = lon_slice.start, lon_slice.stop
    lat_start, lat_stop = lat_slice.start, lat_slice.stop
    # Get row/col indices
    row_start, col_start = product.lonlat_to_rowcol(lon_start, lat_start)
    row_stop, col_stop = product.lonlat_to_rowcol(lon_stop, lat_stop)

    # Handle edge cases - ensure we have at least a 1x1 window
    if col_stop < col_start:
        raise ValueError(f"Invalid column range: {col_start}, {col_stop}")
    elif col_stop == col_start:
        col_stop += 1

    if row_stop < row_start:
        raise ValueError(f"Invalid row range: {row_start}, {row_stop}")
    elif row_stop == row_start:
        row_stop += 1
    return


def _get_border(data_arrays: NDArray[np.floating]) -> NDArray[np.floating]:
    top_row = data_arrays[:, 0, :]
    bottom_row = data_arrays[:, -1, :]
    left_col = data_arrays[:, :, 0]
    right_col = data_arrays[:, :, -1]
    all_pixels = np.hstack([top_row, bottom_row, left_col, right_col])
    return np.nanmedian(all_pixels, axis=1)[:, np.newaxis, np.newaxis]


def rebase_timeseries(
    unreffed_data: np.ndarray, reference_dates: Sequence[datetime]
) -> np.ndarray:
    """Adjust for moving reference dates to create a continuous time series.

    Parameters
    ----------
    unreffed_data : np.ndarray
        3D array of displacement values [time, height, width]
    reference_dates : Sequence[datetime]
        Reference dates for each time step

    Returns
    -------
    np.ndarray
        Continuous displacement time series with consistent reference date
    """
    if len(set(reference_dates)) == 1:
        return unreffed_data.copy()

    shape2d = unreffed_data.shape[1:]
    cumulative_offset = np.zeros(shape2d, dtype=np.float32)
    last_displacement = np.zeros(shape2d, dtype=np.float32)

    # Set initial reference date
    current_reference_date = reference_dates[0]

    output = np.zeros_like(unreffed_data)
    # Process each time step
    for product, current_displacement, output in zip(
        reference_dates, unreffed_data, output
    ):
        # Check for shift in temporal reference date
        if product != current_reference_date:
            # When reference date changes, accumulate the previous displacement
            cumulative_offset += last_displacement
            current_reference_date = product

        # Store current displacement for next iteration
        last_displacement = current_displacement.copy()

        # Add cumulative offset to get consistent reference
        output[:] = current_displacement + cumulative_offset

    return output
