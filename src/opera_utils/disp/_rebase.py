from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from typing import TypeVar

import dask.array
import numpy as np
import pandas as pd
import xarray as xr

from opera_utils._helpers import flatten

from ._utils import _clamp_chunk_dict

logger = logging.getLogger("opera_utils")

__all__ = [
    "create_rebased_displacement",
]

T = TypeVar("T")


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
    process_chunks = _clamp_chunk_dict(process_chunks, da_displacement.shape)

    def process_block(arr: xr.DataArray) -> xr.DataArray:
        return xr.DataArray(
            rebase_timeseries(
                arr.to_numpy(),
                reference_dates=reference_datetimes,
                secondary_dates=da_displacement.time,
                nan_policy=nan_policy,
            ),
            coords=arr.coords,
            dims=arr.dims,
        )

    d_chunked = da_displacement.chunk(process_chunks)
    template = xr.DataArray(
        data=dask.array.empty_like(d_chunked),
        coords=da_displacement.coords,
    )

    # Process the dataset in blocks
    rebased_da = d_chunked.map_blocks(process_block, template=template)

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
    secondary_dates: Sequence[datetime],
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
        Reference date for each time step
    secondary_dates : Sequence[datetime]
        Secondary date for each time step
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
    A = get_incidence_matrix(list(zip(reference_dates, secondary_dates)))
    pA = np.linalg.pinv(A)

    pixels = np.nan_to_num(raw_data.reshape(raw_data.shape[0], -1))
    pixels_rebased = pA @ pixels
    out_stack = pixels_rebased.reshape(raw_data.shape)
    if nan_policy == NaNPolicy.omit:
        return out_stack

    nan_mask = np.isnan(raw_data)
    # Cumulatively sum the mask to find where we should hide after the fact
    nans_propagated = np.cumsum(nan_mask.astype(int), axis=0)
    out_stack[nans_propagated > 0] = np.nan
    return out_stack


def get_incidence_matrix(
    ifg_pairs: Sequence[tuple[T, T]],
    sar_idxs: Sequence[T] | None = None,
    delete_first_date_column: bool = True,
) -> np.ndarray:
    """Build the indicator matrix from a list of ifg pairs (index 1, index 2).

    Parameters
    ----------
    ifg_pairs : Sequence[tuple[T, T]]
        List of ifg pairs represented as tuples of (day 1, day 2)
        Can be ints, datetimes, etc.
    sar_idxs : Sequence[T], optional
        If provided, used as the total set of indexes which `ifg_pairs`
        were formed from.
        Otherwise, created from the unique entries in `ifg_pairs`.
        Only provide if there are some dates which are not present in `ifg_pairs`.
    delete_first_date_column : bool
        If True, removes the first column of the matrix to make it full column rank.
        Size will be `n_sar_dates - 1` columns.
        Otherwise, the matrix will have `n_sar_dates`, but rank `n_sar_dates - 1`.

    Returns
    -------
    A : np.array 2D
        The incident-like matrix for the system: A*phi = dphi
        Each row corresponds to an ifg, each column to a SAR date.
        The value will be -1 on the early (reference) ifgs, +1 on later (secondary)
        since the ifg phase = (later - earlier)
        Shape: (n_ifgs, n_sar_dates - 1)

    """
    if sar_idxs is None:
        sar_idxs = sorted(set(flatten(ifg_pairs)))

    M = len(ifg_pairs)
    col_iter = sar_idxs[1:] if delete_first_date_column else sar_idxs
    N = len(col_iter)
    A = np.zeros((M, N))

    # Create a dictionary mapping sar dates to matrix columns
    # We take the first SAR acquisition to be time 0, leave out of matrix
    date_to_col = {date: i for i, date in enumerate(col_iter)}
    for i, (early, later) in enumerate(ifg_pairs):
        if early in date_to_col:
            A[i, date_to_col[early]] = -1
        if later in date_to_col:
            A[i, date_to_col[later]] = +1

    return A
