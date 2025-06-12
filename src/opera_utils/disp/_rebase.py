from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import xarray as xr

from ._utils import _ensure_chunks

logger = logging.getLogger("opera_utils")

__all__ = [
    "create_rebased_displacement",
]


class NaNPolicy(str, Enum):
    """Policy for handling NaN values in rebase_timeseries."""

    propagate = "propagate"
    omit = "omit"

    def __str__(self) -> str:
        return self.value


def create_rebased_displacement(
    da_displacement: xr.DataArray,
    reference_datetimes: Sequence[datetime | pd.DatetimeIndex],
    process_chunk_size: tuple[int, int] = (512, 512),
    add_reference_time: bool = False,
    nan_policy: str | NaNPolicy = NaNPolicy.propagate,
) -> xr.DataArray:
    """Rebase and stack displacement products with different reference dates.

    This function combines displacement products that may have different reference
    dates by accumulating displacements when the reference date changes.
    When a new reference date is encountered, the displacement values from the
    previous stack's final epoch are added to all epochs in the new stack.

    Parameters
    ----------
    da_displacement : xr.DataArray
        Displacement dataarray to rebase.
    reference_datetimes : Sequence[datetime | pd.DatetimeIndex]
        Reference datetime for each epoch.
        Must be same length as `da_displacement.time`.
    process_chunk_size : tuple[int, int], optional
        Chunk size for processing. Defaults to (512, 512).
    add_reference_time : bool, optional
        Whether to add a zero array for the reference time.
        Defaults to False.
    nan_policy : choices = ["propagate", "omit"]
        Whether to propagate or omit (zero out) NaNs in the data.
        By default "propagate", which means any ministack, or any "reference crossover"
        product, with nan at a pixel causes all subsequent data to be nan.
        If "omit", then any nan causes the pixel to be zeroed out, which is
        equivalent to assuming that 0 displacement occurred during that time.

    Returns
    -------
    xr.DataArray
        Stacked displacement dataarray with rebased displacements.

    """
    logger.info("Starting displacement stack rebasing")

    process_chunks = {
        "time": -1,
        "y": process_chunk_size[0],
        "x": process_chunk_size[1],
    }
    process_chunks = _ensure_chunks(process_chunks, da_displacement.shape)

    # Make the map_blocks-compatible function to accumulate the displacement
    def process_block(arr: xr.DataArray) -> xr.DataArray:
        out = rebase_timeseries(
            arr.to_numpy(), reference_datetimes, nan_policy=nan_policy
        )
        return xr.DataArray(out, coords=arr.coords, dims=arr.dims)

    # Process the dataset in blocks
    rebased_da = da_displacement.chunk(process_chunks).map_blocks(process_block)

    if add_reference_time:
        # Add initial reference epoch of zeros, and rechunk
        rebased_da = xr.concat(
            [xr.full_like(rebased_da[0], 0), rebased_da],
            dim="time",
        )
        # Ensure correct dimension order
        rebased_da = rebased_da.transpose("time", "y", "x")

    return rebased_da


def rebase_timeseries(
    raw_data: np.ndarray,
    reference_dates: Sequence[datetime],
    nan_policy: str | NaNPolicy = NaNPolicy.propagate,
) -> np.ndarray:
    """Adjust for moving reference dates to create a continuous time series.

    DISP-S1 products have a reference date which changes over time.
    For example, shortening to YYYY-MM-DD notation, the products may be

        (2020-01-01, 2020-01-13)
        (2020-01-01, 2020-01-25)
        ...
        (2020-01-01, 2020-06-17)
        (2020-06-17, 2020-06-29)
        ...


    This function sums up the "crossover" values (the displacement image where the
    reference date moves forward) so that the output is referenced to the first input
    time.

    Parameters
    ----------
    raw_data : np.ndarray
        3D array of displacement values with moving reference dates
        shape = (time, rows, cols)
    reference_dates : Sequence[datetime]
        Reference dates for each time step
    nan_policy : choices = ["propagate", "omit"]
        Whether to propagate or omit (zero out) NaNs in the data.
        By default "propagate", which means any ministack, or any "reference crossover"
        product, with nan at a pixel causes all subsequent data to be nan.
        If "omit", then any nan causes the pixel to be zeroed out, which is
        equivalent to assuming that 0 displacement occurred during that time.

    Returns
    -------
    np.ndarray
        Continuous displacement time series with consistent reference date

    """
    if len(set(reference_dates)) == 1:
        return raw_data.copy()

    shape2d = raw_data.shape[1:]
    cumulative_offset = np.zeros(shape2d, dtype=np.float32)
    previous_displacement = np.zeros(shape2d, dtype=np.float32)

    # Set initial reference date
    current_reference_date = reference_dates[0]

    output = np.zeros_like(raw_data)
    # Process each time step
    for cur_ref_date, current_displacement, out_layer in zip(
        reference_dates, raw_data, output
    ):
        # Check for shift in temporal reference date
        if cur_ref_date != current_reference_date:
            # When reference date changes, accumulate the previous displacement
            if nan_policy == NaNPolicy.omit:
                np.nan_to_num(previous_displacement, copy=False)
            cumulative_offset += previous_displacement
            current_reference_date = cur_ref_date

        # Store current displacement for next iteration
        previous_displacement = current_displacement.copy()

        # Add cumulative offset to get consistent reference
        out_layer[:] = current_displacement + cumulative_offset

    return output
