"""NetCDF to Zarr reformatter for displacement product stacks."""

# /// script
# dependencies = ['dask', 'zarr', 'xarray', 'opera_utils[disp]']
# ///
import time
from collections.abc import Sequence
from pathlib import Path

import dask.distributed
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
    num_workers: int = 1,
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
    num_workers : int, optional
        Number of workers for Dask distributed processing. If 1, skips using
        the distributed client.

    """
    start_time = time.time()
    # TODO: how to handle netcdf/zarr output
    # TODO: is there a special way to save geo metadata with zarr?
    # or GeoZarr still too in flux?
    if Path(output_name).suffix == ".nc":
        out_format = "h5netcdf"
    elif Path(output_name).suffix == ".zarr":
        out_format = "zarr"
    else:
        msg = "Only .nc and .zarr output formats are supported"
        raise ValueError(msg)

    # Set up parallel processing
    if num_workers == 1:
        client = None
    else:
        client_kwargs = {"n_workers": num_workers}
        client = dask.distributed.Client(**client_kwargs)
        print("Setting up Dask distributed client")
        print(f"Dask client: {client}")

    # Set default chunks if none provided
    out_chunk_dict = dict(zip(["time", "y", "x"], out_chunks))
    dps = disp.DispProductStack.from_file_list(input_files)
    df = dps.to_dataframe()

    # First, save the template/coordinates/water mask only
    ds = xr.open_mfdataset(dps.filenames)
    # Drop specified variables if requested
    if drop_vars:
        print(f"Dropping variables: {drop_vars}")
        ds = ds.drop_vars(drop_vars, errors="ignore")

    # Drop all except x/y/time
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

    # ############
    # Create rebased stack with specified chunking
    print(f"Creating rebased stack with chunks: {out_chunk_dict}")
    da_disp = disp.create_rebased_displacement(
        ds.displacement,
        df.reference_datetime,
    )
    da_disp = da_disp.assign_coords(spatial_ref=ds.spatial_ref)
    ds_disp = da_disp.to_dataset(name="displacement")

    if out_format == "zarr":
        encoding = _get_zarr_encoding(ds_disp, out_chunks)
        ds_disp.to_zarr(output_name, encoding=encoding, mode="a")
    else:
        encoding = _get_netcdf_encoding(ds_disp, out_chunks)
        ds_disp.to_netcdf(output_name, engine="h5netcdf", encoding=encoding, mode="a")
    print(f"Wrote displacement: {time.time() - start_time:.1f}s")

    ds_remaining = ds[UNIQUE_PER_DATE_DATASETS + SAME_PER_MINISTACK_DATASETS].chunk(
        out_chunk_dict
    )

    # ############
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

    _print_summary(output_name, start_time)


def _print_summary(output_name: Path | str, start_time: float) -> None:
    # Report completion
    elapsed_time = time.time() - start_time
    file_size = sum(
        f.stat().st_size for f in Path(output_name).rglob("*") if f.is_file()
    )
    file_size_mb = file_size / (1024 * 1024)
    print(f"Successfully created {output_name}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Completed in {elapsed_time:.1f} seconds")


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


# Data variables:
#  spatial_ref                     (time) int64 240B 0 0
#  reference_time                  (time) datetime64[ns]
#  displacement                    (time, y, x) float32
#  short_wavelength_displacement   (time, y, x) float32
#  recommended_mask                (time, y, x) float32
#  connected_component_labels      (time, y, x) float32
#  temporal_coherence              (time, y, x) float32
#  estimated_phase_quality         (time, y, x) float32
#  persistent_scatterer_mask       (time, y, x) float32
#  shp_counts                      (time, y, x) float32
#  water_mask                      (time, y, x) float32
#  phase_similarity                (time, y, x) float32
#  timeseries_inversion_residuals  (time, y, x) float32


if __name__ == "__main__":
    tyro.cli(reformat_stack)
