"""NetCDF to Zarr reformatter for displacement product stacks."""

# /// script
# dependencies = ['zarr>=3', 'xarray', 'opera_utils[disp]']
# ///
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
import xarray as xr
from zarr.codecs import BloscCodec

from opera_utils import disp

from ._enums import (
    SAME_PER_MINISTACK_DATASETS,
    UNIQUE_PER_DATE_DATASETS,
    CorrectionDataset,
    DisplacementDataset,
    QualityDataset,
)
from ._netcdf import create_virtual_stack
from ._utils import _ensure_chunks, round_mantissa


def reformat_stack(
    input_files: list[Path],
    output_name: str,
    out_chunks: tuple[int, int, int] = (4, 256, 256),
    drop_vars: Sequence[str] | None = None,
    apply_solid_earth_corrections: bool = True,
    apply_ionospheric_corrections: bool = False,
    quality_dataset: QualityDataset | None = QualityDataset.RECOMMENDED_MASK,
    quality_threshold: float = 0.5,
    process_chunk_size: tuple[int, int] = (2048, 2048),
    do_round: bool = True,
    max_workers: int = 1,
) -> None:
    """Reformat NetCDF DISP-S1 files into one NetCDF/Zarr stack.

    This function reformats a set of input DISP-S1 files, adjusts
    the `displacement` layer to be rebased to the first reference time,
    and creates a single NetCDF/Zarr stack for quicker browsing access.

    Parameters
    ----------
    input_files : Sequence[Path]
        Input DISP-S1 NetCDF files.
    output_name : str
        Name of the output file.
        Must end in ".nc" or ".zarr".
    out_chunks : tuple[int, int, int]
        Chunking configuration for output DataArray.
        Defaults to DEFAULT_CHUNKS, which is {"time": 4, "x": 256, "y": 256}
    drop_vars : list of str, optional
        list of variable names to drop from the dataset before saving.
        Example: ["estimated_phase_quality"]
    apply_solid_earth_corrections : bool
        Apply solid earth tide correction to the data.
        Default is True.
    apply_ionospheric_corrections : bool
        Apply ionospheric delay correction to the data.
        Default is False.
    quality_dataset : QualityDataset | None
        Name of the quality dataset to use as a mask when accumulating
        displacement for re-referencing.
        If None, no masking is performed.
    quality_threshold : float
        Threshold for the quality dataset to use as a mask.
        Default is 0.5.
    process_chunk_size : tuple[int, int]
        Chunking configuration for processing DataArray.
        Defaults to (2048, 2048).
    do_round : bool
        If True, rounds mantissa bits of floating point rasters to compress the data.
    max_workers : int
        Number of workers to use for parallel processing.
        Default is 1.

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

    if max_workers == 1:
        client = None
    else:
        import dask.distributed

        cluster = dask.distributed.LocalCluster(n_workers=max_workers)
        print(f"Cluster: {cluster}")
        client = dask.distributed.Client(cluster)
        print("Dashboard link:", client.dashboard_link)
        print(f"Dask client: {client}")

    corrections: list[CorrectionDataset] = []
    if apply_solid_earth_corrections:
        corrections.append(CorrectionDataset.SOLID_EARTH_TIDE)
    if apply_ionospheric_corrections:
        corrections.append(CorrectionDataset.IONOSPHERIC_DELAY)
    # Set default chunks if none provided
    out_chunk_dict = dict(zip(["time", "y", "x"], out_chunks))
    dps = disp.DispProductStack.from_file_list(input_files)
    df = dps.to_dataframe()

    # #####################
    # Write minimal dataset
    # #####################
    process_chunk_dict = {
        "time": 1,
        "y": process_chunk_size[0],
        "x": process_chunk_size[1],
    }
    ds = xr.open_mfdataset(dps.filenames, engine="h5netcdf", chunks=process_chunk_dict)

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
    if corrections:
        correction_names = [str(c).split("/")[-1] for c in corrections]
        ds_corrections = xr.open_mfdataset(
            dps.filenames,
            engine="h5netcdf",
            group="corrections",
            chunks=process_chunk_dict,
        )[correction_names]
    else:
        ds_corrections = None
    _write_rebased_stack(
        ds,
        df,
        output_name,
        out_chunks,
        data_var=DisplacementDataset.DISPLACEMENT,
        out_format=out_format,
        ds_corrections=ds_corrections,
        quality_dataset=quality_dataset,
        quality_threshold=quality_threshold,
        process_chunk_size=process_chunk_size,
        do_round=do_round,
    )
    print(f"Wrote displacement at {time.time() - start_time:.1f}s")
    if str(DisplacementDataset.SHORT_WAVELENGTH) in ds.data_vars:
        _write_rebased_stack(
            ds,
            df,
            output_name,
            out_chunks,
            data_var=DisplacementDataset.SHORT_WAVELENGTH,
            out_format=out_format,
            process_chunk_size=process_chunk_size,
            do_round=do_round,
        )
        print(f"Wrote short_wavelength_displacement at {time.time() - start_time:.1f}s")

    # #########################
    # Write remaining variables
    # #########################
    # TODO: we could just read once per ministack, then tile, then write
    ds_remaining = ds[UNIQUE_PER_DATE_DATASETS + SAME_PER_MINISTACK_DATASETS].chunk(
        {
            "time": out_chunk_dict["time"],
            "y": process_chunk_dict["y"],
            "x": process_chunk_dict["x"],
        }
    )
    for var in ds_remaining.data_vars:
        # Round, if it's a float32
        d = ds_remaining[var]
        if do_round and np.issubdtype(d.dtype, np.floating):
            d.data = round_mantissa(d.data, keep_bits=7)
    print(f"Writing remaining variables: {ds_remaining.data_vars}")
    # Now here, we'll use the virtual dataset feature of HDF5 if we're writing NetCDF
    if out_format == "zarr":
        encoding = _get_zarr_encoding(ds_remaining, out_chunks)
        ds_remaining.to_zarr(output_name, encoding=encoding, mode="a")

    else:
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
    data_var: DisplacementDataset = DisplacementDataset.DISPLACEMENT,
    out_format: str = "zarr",
    ds_corrections: xr.Dataset | None = None,
    quality_dataset: QualityDataset | None = None,
    quality_threshold: float = 0.5,
    do_round: bool = True,
    process_chunk_size: tuple[int, int] = (2048, 2048),
) -> None:
    da_displacement = ds[str(data_var)]

    # For this, we want to work on the entire time stack at once
    # Otherwise the summation in `create_rebased_displacement` won't work
    process_chunks = {
        "time": -1,
        "y": process_chunk_size[0],
        "x": process_chunk_size[1],
    }
    process_chunks = _ensure_chunks(process_chunks, da_displacement.shape)
    if ds_corrections:
        ds_corrections = ds_corrections.chunk(process_chunks)
        for var in ds_corrections.data_vars:
            if ds_corrections[var].shape != da_displacement.shape:
                continue
            da_displacement = da_displacement - ds_corrections[var]

    if quality_dataset is not None:
        quality_da = ds[quality_dataset].chunk(process_chunks)
        da_displacement = da_displacement.where(quality_da > quality_threshold)

    da_disp = disp.create_rebased_displacement(
        da_displacement,
        # Need to strip timezone to match the ds.time coordinates
        reference_datetimes=df.reference_datetime.dt.tz_localize(None),
        process_chunk_size=process_chunk_size,
    )
    da_disp = da_disp.assign_coords(spatial_ref=ds.spatial_ref)
    if do_round and np.issubdtype(da_disp.dtype, np.floating):
        da_disp.data = round_mantissa(da_disp.data, keep_bits=10)
    ds_disp = da_disp.to_dataset(name=str(data_var))
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
