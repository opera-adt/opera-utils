"""NetCDF to Zarr reformatter for displacement product stacks."""

# /// script
# dependencies = ['zarr>=3', 'xarray', 'opera_utils[disp]']
# ///
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import tyro
import xarray as xr
from zarr.codecs import BloscCodec

from opera_utils import disp

from ._enums import SAME_PER_MINISTACK_DATASETS, UNIQUE_PER_DATE_DATASETS


def reformat_stack(
    input_files: list[Path],
    output_name: str,
    out_chunks: tuple[int, int, int] = (5, 256, 256),
    drop_vars: Sequence[str] | None = None,
) -> None:
    """Reformat NetCDF displacement files to Zarr format with compression.

    Parameters
    ----------
    input_files : Sequence[Path]
        Input NetCDF files.
    output_name : str
        Name of the output file.
        Must end in ".nc" or ".zarr".
    out_chunks : tuple[int, int, int]
        Chunking configuration for output DataArray.
        Defaults to DEFAULT_CHUNKS, which is {"time": 5, "x": 256, "y": 256}
    drop_vars : list of str, optional
        list of variable names to drop from the dataset before saving.
        Example: ["estimated_phase_quality"]

    """
    start_time = time.time()
    # TODO: is there a special way to save geo metadata with zarr?
    # or GeoZarr still too in flux?
    if Path(output_name).suffix == ".nc":
        out_format = "h5netcdf"
    elif Path(output_name).suffix == ".zarr":
        out_format = "zarr"
    else:
        msg = "Only .nc and .zarr output formats are supported"
        raise ValueError(msg)

    # Set default chunks if none provided
    out_chunk_dict = dict(zip(["time", "y", "x"], out_chunks))
    dps = disp.DispProductStack.from_file_list(input_files)
    df = dps.to_dataframe()

    # #####################
    # Write minimal dataset
    # #####################
    ds = xr.open_mfdataset(dps.filenames)
    # Drop specified variables if requested
    if drop_vars:
        print(f"Dropping variables: {drop_vars}")
        ds = ds.drop_vars(drop_vars, errors="ignore")

    # Here we just want the output template/coordinates/water mask
    all_vars = list(ds.data_vars)
    minimal_vars = ["spatial_ref", "reference_time", "water_mask"]
    ds_minimal = ds.drop_vars([v for v in all_vars if v not in minimal_vars])
    # Only keep one water mask, as it's repeated for all frame outputs
    ds_minimal["water_mask"] = ds_minimal["water_mask"].isel(time=0)

    # Configure compression encoding
    if out_format == "zarr":
        encoding = _get_zarr_encoding(ds_minimal, out_chunks, add_coords=True)
        ds_minimal.to_zarr(output_name, encoding=encoding, mode="w")
    else:
        encoding = _get_netcdf_encoding(ds_minimal, out_chunks)
        ds_minimal.to_netcdf(
            output_name, engine="h5netcdf", encoding=encoding, mode="w"
        )
    print(f"Wrote minimal dataset: {time.time() - start_time:.1f}s")

    # #########################
    # Rebase displacement array
    # #########################
    print(f"Creating rebased stack with chunks: {out_chunk_dict}")
    _write_rebased_stack(ds, df, output_name, out_chunks, out_format)
    print(f"Wrote displacement: {time.time() - start_time:.1f}s")

    # #########################
    # Write remaining variables
    # #########################
    ds_remaining = ds[UNIQUE_PER_DATE_DATASETS + SAME_PER_MINISTACK_DATASETS].chunk(
        out_chunk_dict
    )
    print(f"Writing remaining variables: {ds_remaining.data_vars}")
    # Now here, we'll use the virtual dataset feature of HDF5 if we're writing NetCDF
    if out_format == "zarr":
        encoding = _get_zarr_encoding(ds_remaining, out_chunks)
        ds_remaining.to_zarr(output_name, encoding=encoding, mode="a")

    else:
        from ._netcdf import create_virtual_stack

        create_virtual_stack(
            input_files=dps.filenames,
            output=output_name,
            dataset_names=[
                str(ds) for ds in UNIQUE_PER_DATE_DATASETS + SAME_PER_MINISTACK_DATASETS
            ],
        )
    print(f"Wrote remaining: {time.time() - start_time:.1f}s")


def _write_rebased_stack(
    ds: xr.Dataset,
    df: pd.DataFrame,
    output_name: Path | str,
    out_chunks: tuple[int, int, int],
    out_format: str = "zarr",
) -> None:
    da_disp = disp.create_rebased_displacement(
        ds.displacement,
        # Need to strip timezone to match the ds.time coordinates
        df.reference_datetime.dt.tz_localize(None),
    )
    da_disp = da_disp.assign_coords(spatial_ref=ds.spatial_ref)
    ds_disp = da_disp.to_dataset(name="displacement")
    if out_format == "zarr":
        encoding = _get_zarr_encoding(ds_disp, out_chunks)
        ds_disp.to_zarr(output_name, encoding=encoding, mode="a")
    else:
        encoding = _get_netcdf_encoding(ds_disp, out_chunks)
        ds_disp.to_netcdf(output_name, engine="h5netcdf", encoding=encoding, mode="a")


def _get_netcdf_encoding(
    ds: xr.Dataset,
    chunks: tuple[int, int, int],
    compression_level: int = 6,
    data_vars: Sequence[str] = [],
) -> dict:
    encoding = {}
    comp = {"zlib": True, "complevel": compression_level, "chunksizes": chunks}
    if not data_vars:
        data_vars = list(ds.data_vars)
    encoding = {var: comp for var in data_vars if ds[var].ndim >= 2}
    for var in data_vars:
        if ds[var].ndim < 2:
            continue
        encoding[var] = comp
        if ds[var].ndim == 2:
            encoding[var]["chunksizes"] = chunks[-2:]
    return encoding


def _get_zarr_encoding(
    ds: xr.Dataset,
    chunks: tuple[int, int, int],
    add_coords: bool = False,
    compression_name: str = "zstd",
    compression_level: int = 6,
    data_vars: Sequence[str] = [],
) -> dict[str, dict]:
    encoding_per_var = {
        "compressors": [
            BloscCodec(
                cname=compression_name,
                clevel=compression_level,
            )
        ],
        "chunks": chunks,
    }
    if not data_vars:
        data_vars = list(ds.data_vars)
    encoding = {var: encoding_per_var for var in data_vars if ds[var].ndim >= 2}
    if not add_coords:
        return encoding
    # Handle coordinate compression
    encoding.update({var: {"compressors": []} for var in ds.coords})
    return encoding


if __name__ == "__main__":
    tyro.cli(reformat_stack)
