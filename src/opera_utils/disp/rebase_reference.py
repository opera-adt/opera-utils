#!/usr/bin/env python3
# /// script
# dependencies = ["numpy", "dolphin", "tqdm", "tyro"]
# ///
"""Convert a series of OPERA DISP-S1 products to a single-reference stack.

The OPERA L3 InSAR displacement netCDF files have reference dates which
move forward in time. Each displacement is relative between two SAR acquisition dates.

This converts these files into a single continuous displacement time series.
The current format is a stack of geotiff rasters.

Usage:
    python -m opera_utils.disp.rebase_reference single-reference-out/ OPERA_L3_DISP-S1_*.nc
"""

import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import tyro
from dolphin import io
from dolphin.utils import flatten, format_dates
from tqdm.auto import trange

from ._product import DispProduct

CORRECTION_DATASETS = [
    "/corrections/solid_earth_tide",
    "/corrections/ionospheric_delay",
]


def _make_gtiff_output_paths(output_dir: Path, all_dates: list[datetime], dataset: str):
    ref_date = all_dates[0]
    suffix = ".tif"

    out_paths = [
        output_dir / f"{dataset}_{format_dates(ref_date, d)}{suffix}"
        for d in all_dates[1:]
    ]
    output_dir.mkdir(exist_ok=True, parents=True)
    return out_paths


def rereference(
    output_dir: Path,
    nc_files: list[Path | str],
    dataset: str = "displacement",
    apply_corrections: bool = True,
    nodata: float = np.nan,
    keep_bits: int = 9,
):
    """Create a single-reference stack from a list of OPERA displacement files.

    Parameters
    ----------
    output_dir : str
        File path to the output directory.
    nc_files : list[str]
        One or more netCDF files, each containing a 'displacement' dataset
        for a reference_date -> secondary_date interferogram.
    dataset : str
        Name of HDF5 dataset within product to convert.
    apply_corrections : bool
        Apply corrections to the data.
        Default is True.
    block_shape : tuple[int, int]
        Size of chunks of data to load at once.
        Default is (256, 256)
    nodata : float
        Value to use in translated rasters as nodata value.
        Default is np.nan
    keep_bits : int
        Number of floating point mantissa bits to retain in the output rasters.
        Default is 9.

    """
    ifg_date_pairs = []
    for f in nc_files:
        dp = DispProduct.from_filename(f)
        ifg_date_pairs.append((dp.reference_datetime, dp.secondary_datetime))

    # Flatten all dates, find unique sorted list of SAR epochs
    all_dates = sorted(set(flatten(ifg_date_pairs)))

    # open a GDAL dataset for the first file just to get the shape/geoinformation
    # All netCDF files for a frame are on the same grid.
    gdal_str = io.format_nc_filename(nc_files[0], dataset)
    ncols, nrows = io.get_raster_xysize(gdal_str)

    # Create the main displacement dataset.
    output_paths = _make_gtiff_output_paths(
        output_dir,
        all_dates=all_dates,
        dataset=dataset,  # like_filename=gdal_str,
    )

    reader = io.HDF5StackReader.from_file_list(
        nc_files, dset_names=dataset, nodata=nodata
    )
    if apply_corrections:
        corrections_readers = [
            io.HDF5StackReader.from_file_list(
                nc_files, dset_names=correction_dataset, nodata=nodata
            )
            for correction_dataset in CORRECTION_DATASETS
        ]
    else:
        corrections_readers = []

    num_dates = len(nc_files)
    # Make a "cumulative offset" which adds up the phase each time theres a reference
    # date changeover.
    cumulative_offset = np.zeros((nrows, ncols), dtype=np.float32)
    last_displacement = np.zeros((nrows, ncols), dtype=np.float32)
    last_reference_date = ifg_date_pairs[0][0]

    write_threads = []
    for idx in trange(num_dates, desc="Summing dates"):
        # Read all 3D array of shape (M, block_rows, block_cols)
        current_displacement = np.squeeze(reader[idx, :, :])
        if isinstance(current_displacement, np.ma.MaskedArray):
            current_displacement = current_displacement.filled(0)

        # Apply corrections if needed
        for r in corrections_readers:
            current_displacement -= np.squeeze(r[idx, :, :])

        cur_ref, cur_sec = ifg_date_pairs[idx]
        if cur_ref != last_reference_date:
            # e.g. we had (1,2), (1,3), now we hit (3,4)
            # So to write out (1,4), we need to add the running total
            # to the current displacement
            last_reference_date = cur_ref
            cumulative_offset += last_displacement
        last_displacement = current_displacement

        # Write current output raster
        def write_arr(arr, output_name, like_filename, nodata):
            io.round_mantissa(arr, keep_bits=keep_bits)
            io.write_arr(
                arr=arr,
                output_name=output_name,
                like_filename=like_filename,
                nodata=nodata,
            )

        # Use a background thread to write the data to avoid holding up the reading loop
        t = threading.Thread(
            target=write_arr,
            args=(
                current_displacement + cumulative_offset,
                output_paths[idx],
                gdal_str,
                nodata,
            ),
        )
        t.start()
        write_threads.append(t)

    # Then wait for just those threads
    for t in write_threads:
        t.join()

    print(f"Saved displacement stack to {output_dir}")


if __name__ == "__main__":
    tyro.cli(rereference)
