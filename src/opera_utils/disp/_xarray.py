from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from ._product import DispProductStack

logger = logging.getLogger("opera_utils")

DEFAULT_CHUNKS: dict[str, int] = {"time": 10, "x": 1024, "y": 1024}

__all__ = [
    "create_rebased_stack",
]


def create_rebased_stack(
    dps: DispProductStack,
    chunks: dict[str, int] | None = None,
    data_vars: Sequence[str] = ["displacement", "short_wavelength_displacement"],
) -> xr.Dataset:
    """Rebase and stack displacement products with different reference dates.

    This function combines displacement products that may have different reference
    dates by accumulating displacements when the reference date changes.
    When a new reference date is encountered, the displacement values from the
    previous stack's final epoch are added to all epochs in the new stack.

    Parameters
    ----------
    dps : DispProductStack
        Stack of displacement products to combine.
    chunks : Dict[str, int], optional
        Chunking configuration for xarray. Defaults to DEFAULT_CHUNKS.
    data_vars : Sequence[str], optional
        Data variables to process for rebasing.
        Defaults to ["displacement", "short_wavelength_displacement"].

    Returns
    -------
    xr.Dataset
        Stacked displacement dataset with rebased displacements.

    """
    logger.info("Starting displacement stack rebasing")

    chunks = _get_chunks(chunks, dps)
    logger.info(f"Using chunk configuration: {chunks}")

    # Get dataframe and group by reference dates
    df = dps.to_dataframe()
    substacks = df.groupby(["reference_datetime", "secondary_datetime"]).apply(
        lambda x: x, include_groups=False
    )
    reference_dates = pd.Series(df.reference_datetime.unique())

    # Process each reference date's substack, create the stacked dataset
    rebased_stacks = _accumulate_displacements(substacks, reference_dates, data_vars)

    # Add initial reference epoch of zeros
    initial_epoch = _create_initial_epoch(df, rebased_stacks[0])
    # Combine and rechunk
    ds_combined = xr.concat([initial_epoch, *rebased_stacks], dim="time").chunk(chunks)
    try:
        ds_combined.rio.write_crs(f"EPSG:{dps.epsg}", inplace=True)
    except AttributeError as e:
        logger.warning("Could not write CRS to dataset: %s", e)

    logger.info(f"Successfully combined {len(rebased_stacks)} substacks")
    return ds_combined


def _get_chunks(chunks: dict[str, int] | None, dps: DispProductStack) -> dict[str, int]:
    """Ensure chunks are smaller than the downloaded size."""
    chunks = {**DEFAULT_CHUNKS, **(chunks or {})}
    with xr.open_dataset(dps.products[0].filename) as ds:
        data_shape = ds["displacement"].shape
        chunks["x"] = min(chunks["x"], data_shape[1])
        chunks["y"] = min(chunks["y"], data_shape[0])
    chunks["time"] = min(chunks["time"], len(dps.products))
    return chunks


def _accumulate_displacements(
    substacks: pd.DataFrame,
    reference_dates: pd.Series,
    data_vars: Sequence[str] = ["displacement", "short_wavelength_displacement"],
) -> list[xr.Dataset]:
    """Open each substack, process into cumulative displacement."""
    rebased_stacks: list[xr.Dataset] = []

    for idx, ref_date in enumerate(reference_dates):
        logger.debug(f"Processing ministack for reference date: {ref_date}")

        # Get files for this reference date
        substack_df = substacks.loc[ref_date]
        stack_files = substack_df.sort_index().filename.to_list()

        # Open the current ministack's files
        stack = xr.open_mfdataset(stack_files, engine="h5netcdf")

        # Append first epoch of new ministack to last epochs of previous
        if idx > 0:
            for ds_name in data_vars:
                stack[ds_name] += rebased_stacks[-1].isel(time=-1)[ds_name]

        rebased_stacks.append(stack)

    return rebased_stacks


def _create_initial_epoch(df: pd.DataFrame, first_stack: xr.Dataset) -> xr.Dataset:
    """Create initial reference epoch with zero displacement."""
    # Get earliest date from the dataframe
    first_epoch = df.reference_datetime.min()
    first_epoch_dt64 = np.datetime64(first_epoch.to_pydatetime(), "ns")

    # Create zero-valued dataset matching spatial structure
    initial_ds = xr.full_like(first_stack.isel(time=0), 0)
    initial_ds["time"] = first_epoch_dt64
    initial_ds["reference_time"] = initial_ds["time"]

    # Expand time dimension
    initial_ds = initial_ds.expand_dims("time")

    return initial_ds
