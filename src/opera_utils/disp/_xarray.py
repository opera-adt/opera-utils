from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

import pandas as pd
import xarray as xr

from ._rebase import rebase_timeseries
from ._utils import _ensure_chunks

logger = logging.getLogger("opera_utils")

__all__ = [
    "create_rebased_displacement",
]


def create_rebased_displacement(
    da_displacement: xr.DataArray,
    reference_datetimes: Sequence[datetime | pd.DatetimeIndex],
    process_chunk_size: tuple[int, int] = (512, 512),
    add_reference_time: bool = False,
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
        out = rebase_timeseries(arr.to_numpy(), reference_datetimes)
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
