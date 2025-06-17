"""NetCDF to Zarr reformatter for displacement product stacks."""

# /// script
# dependencies = ['zarr>=3', 'xarray', 'opera_utils[disp]']
# ///
from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable, Sequence
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
import xarray as xr
from affine import Affine
from numpy.typing import ArrayLike
from pyproj import CRS
from zarr.codecs import BloscCodec

from opera_utils import disp

from ._enums import (
    SAME_PER_MINISTACK_DATASETS,
    UNIQUE_PER_DATE_DATASETS,
    CorrectionDataset,
    DisplacementDataset,
    QualityDataset,
    ReferenceMethod,
)
from ._netcdf import create_virtual_stack
from ._rebase import NaNPolicy
from ._reference import (
    _get_reference_row_col,
    get_reference_values,
)
from ._utils import _ensure_chunks, round_mantissa

logger = logging.getLogger("opera_utils")


def reformat_stack(
    input_files: list[Path],
    output_name: str,
    out_chunks: tuple[int, int, int] = (4, 256, 256),
    shard_factors: tuple[int, int, int] = (1, 4, 4),
    drop_vars: Sequence[str] | None = None,
    apply_solid_earth_corrections: bool = True,
    apply_ionospheric_corrections: bool = False,
    quality_datasets: Sequence[QualityDataset] | None = [
        QualityDataset.RECOMMENDED_MASK
    ],
    quality_thresholds: Sequence[float] | None = [0.5],
    reference_method: ReferenceMethod = ReferenceMethod.HIGH_COHERENCE,
    reference_row: int | None = None,
    reference_col: int | None = None,
    reference_lon: float | None = None,
    reference_lat: float | None = None,
    reference_border_pixels: int = 3,
    reference_coherence_threshold: float = 0.7,
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
        Defaults to (4, 256, 256).
    shard_factors : tuple[int, int, int]
        For Zarr outputs, sharding configuration for output DataArray.
        The factors are applied respectively to the chunks sizes in `out_chunks` to
        create fewer output files in the Zarr store.
        Defaults to (1, 4, 4).
    drop_vars : list of str, optional
        list of variable names to drop from the dataset before saving.
        Example: ["estimated_phase_quality"]
    apply_solid_earth_corrections : bool
        Apply solid earth tide correction to the data.
        Default is True.
    apply_ionospheric_corrections : bool
        Apply ionospheric delay correction to the data.
        Default is False.
    quality_datasets : Sequence[QualityDataset] | None
        Name of the quality datasets to use as a mask when accumulating
        displacement for re-referencing.
        If None, no masking is performed.
        Must be same length as `quality_thresholds`.
        Default is [QualityDataset.RECOMMENDED_MASK], which uses the built-in
        recommended mask.
    quality_thresholds : Sequence[float]
        Thresholds for the quality datasets to use as a mask.
        Must be same length as `quality_datasets`.
        Default is [0.5].
    reference_method : ReferenceMethod
        Reference method to use.
        Default is ReferenceMethod.NONE.
        Options are:
        - ReferenceMethod.NONE: No reference method.
        - ReferenceMethod.POINT: Reference point.
        - ReferenceMethod.MEDIAN: Full-scene median per date. Excludes water pixels.
        - ReferenceMethod.BORDER: Median of border pixels. Excludes water pixels.
        - ReferenceMethod.HIGH_COHERENCE: Median of high-coherence mask.
    reference_row : int | None
        For ReferenceMethod.POINT, row index for point reference.
    reference_col : int | None
        For ReferenceMethod.POINT, column index for point reference.
    reference_lon : float | None
        For ReferenceMethod.POINT, longitude (in degrees) for point reference.
    reference_lat : float | None
        For ReferenceMethod.POINT, latitude (in degrees) for point reference.
    reference_border_pixels : int
        For ReferenceMethod.BORDER, number of pixels to use for border median.
        Defaults to 3.
    reference_coherence_threshold : float
        For ReferenceMethod.HIGH_COHERENCE, threshold for coherence to use as a mask.
        Defaults to 0.7.
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
    # Multiply the chunk sizes by the shard factors
    out_shard_dict = _to_shard_dict(out_chunks, shard_factors)

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
        ds_minimal.chunk(out_shard_dict).to_zarr(
            output_name,
            encoding=encoding,
            mode="w",
            consolidated=False,
        )
    else:
        encoding = _get_netcdf_encoding(ds_minimal, out_chunks)
        ds_minimal.to_netcdf(
            output_name, engine="h5netcdf", encoding=encoding, mode="w"
        )
    print(f"Wrote minimal dataset: {time.time() - start_time:.1f}s")

    # ################################
    # Write non-displacement variables
    # ################################
    # TODO: we could just read once per ministack, then tile, then write
    ds_remaining = ds[UNIQUE_PER_DATE_DATASETS + SAME_PER_MINISTACK_DATASETS].chunk(
        {
            "time": out_shard_dict["time"],
            "y": process_chunk_dict["y"],
            "x": process_chunk_dict["x"],
        }
    )
    # TODO: make this configurable: currently we take every 15th coherence since, during
    # historical processing, the coherences are the same per ministack
    da_temp_coh = ds.temporal_coherence[::15]
    # Use the harmonic mean to downweight any time periods with low coherence
    avg_coherence = da_temp_coh.shape[0] / (1.0 / da_temp_coh).sum(
        dim="time", skipna=False, min_count=1
    )
    # Save the coherence to the output
    ds_remaining["average_temporal_coherence"] = xr.DataArray(
        avg_coherence, dims=("y", "x"), coords={"y": ds.y, "x": ds.x}
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
        ds_remaining.to_zarr(
            output_name,
            encoding=encoding,
            mode="a",
            consolidated=False,
        )

    else:
        create_virtual_stack(
            input_files=dps.filenames,
            output=output_name,
            dataset_names=[
                str(ds) for ds in UNIQUE_PER_DATE_DATASETS + SAME_PER_MINISTACK_DATASETS
            ],
        )
    print(f"Wrote remaining: {time.time() - start_time:.1f}s")

    if reference_method == ReferenceMethod.HIGH_COHERENCE:
        # Get the average coherence dataset
        good_pixel_mask = avg_coherence > reference_coherence_threshold
        ref_row = ref_col = None
    elif reference_method == ReferenceMethod.POINT:
        transform = _get_transform(ds)
        crs = CRS.from_wkt(ds.spatial_ref.crs_wkt)

        ref_row, ref_col = _get_reference_row_col(
            row=reference_row,
            col=reference_col,
            lon=reference_lon,
            lat=reference_lat,
            crs=crs,
            transform=transform,
        )
    elif reference_method in (ReferenceMethod.BORDER, ReferenceMethod.MEDIAN):
        good_pixel_mask = np.asarray(ds.water_mask) == 1
        ref_row = ref_col = None
    else:
        msg = f"Unknown ReferenceMethod {reference_method}"
        raise ValueError(msg)

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
        out_chunks=out_chunks,
        data_var=DisplacementDataset.DISPLACEMENT,
        reference_method=reference_method,
        reference_row=ref_row,
        reference_col=ref_col,
        border_pixels=reference_border_pixels,
        good_pixel_mask=good_pixel_mask,
        out_format=out_format,
        ds_corrections=ds_corrections,
        quality_datasets=quality_datasets,
        quality_thresholds=quality_thresholds,
        process_chunk_size=process_chunk_size,
        shard_factors=shard_factors,
        do_round=do_round,
    )
    print(f"Wrote displacement at {time.time() - start_time:.1f}s")
    if str(DisplacementDataset.SHORT_WAVELENGTH) in ds.data_vars:
        _write_rebased_stack(
            ds,
            df,
            output_name,
            out_chunks=out_chunks,
            data_var=DisplacementDataset.SHORT_WAVELENGTH,
            out_format=out_format,
            process_chunk_size=process_chunk_size,
            shard_factors=shard_factors,
            do_round=do_round,
        )
        print(f"Wrote short_wavelength_displacement at {time.time() - start_time:.1f}s")


def _get_transform(ds: xr.Dataset) -> Affine:
    return Affine.from_gdal(*map(float, ds.spatial_ref.GeoTransform.split()))


def _to_shard_dict(
    out_chunks: tuple[int, int, int], shard_factors: tuple[int, int, int]
) -> dict[str, int]:
    return dict(
        zip(["time", "y", "x"], [c * f for c, f in zip(out_chunks, shard_factors)])
    )


def _write_rebased_stack(
    ds: xr.Dataset,
    df: pd.DataFrame,
    output_name: Path | str,
    out_chunks: tuple[int, int, int],
    data_var: DisplacementDataset = DisplacementDataset.DISPLACEMENT,
    reference_method: ReferenceMethod = ReferenceMethod.NONE,
    good_pixel_mask: ArrayLike | None = None,
    border_pixels: int = 3,
    reference_row: int | None = None,
    reference_col: int | None = None,
    out_format: str = "zarr",
    ds_corrections: xr.Dataset | None = None,
    quality_datasets: Sequence[QualityDataset] | None = None,
    quality_thresholds: Sequence[float] | None = None,
    do_round: bool = True,
    process_chunk_size: tuple[int, int] = (2048, 2048),
    shard_factors: tuple[int, int, int] = (1, 4, 4),
    nan_policy: str | NaNPolicy = NaNPolicy.propagate,
) -> None:
    da_displacement = ds[str(data_var)]

    # For this, we want to work on the entire time stack at once
    # Otherwise the summation in `create_rebased_displacement` won't work
    process_chunks = {
        "time": -1,
        "y": process_chunk_size[0],
        "x": process_chunk_size[1],
    }
    out_shard_dict = _to_shard_dict(out_chunks, shard_factors)
    process_chunks = _ensure_chunks(process_chunks, da_displacement.shape)
    if ds_corrections:
        ds_corrections = ds_corrections.chunk(process_chunks)
        for var in ds_corrections.data_vars:
            if ds_corrections[var].shape != da_displacement.shape:
                continue
            da_displacement = da_displacement - ds_corrections[var]

    if quality_datasets is not None:
        if quality_thresholds is None:
            msg = "quality_thresholds must be provided if quality_datasets is not None"
            raise ValueError(msg)
        if len(quality_datasets) != len(quality_thresholds):
            msg = "quality_datasets and quality_thresholds must have the same length"
            raise ValueError(msg)
        da_quality_mask = combine_quality_masks(
            [ds[qd].chunk(process_chunks) for qd in quality_datasets],
            quality_thresholds,
        )
        da_displacement = da_displacement.where(da_quality_mask)

    da_disp = disp.create_rebased_displacement(
        da_displacement,
        # Need to strip timezone to match the ds.time coordinates
        reference_datetimes=df.reference_datetime.dt.tz_localize(None),
        process_chunk_size=process_chunk_size,
        nan_policy=nan_policy,
    )
    crs = CRS.from_wkt(ds.spatial_ref.crs_wkt)
    transform = _get_transform(ds)
    if reference_method is not ReferenceMethod.NONE:
        logger.info(f"spatially referencing with {reference_method}")
        ref_values = get_reference_values(
            da_disp,
            method=reference_method,
            row=reference_row,
            col=reference_col,
            crs=crs,
            transform=transform,
            border_pixels=border_pixels,
            good_pixel_mask=good_pixel_mask,
        )
        da_disp_referenced = da_disp - ref_values
    else:
        da_disp_referenced = da_disp

    da_disp_referenced = da_disp_referenced.assign_coords(spatial_ref=ds.spatial_ref)
    if do_round and np.issubdtype(da_disp_referenced.dtype, np.floating):
        da_disp_referenced.data = round_mantissa(da_disp_referenced.data, keep_bits=10)
    ds_disp = da_disp_referenced.to_dataset(name=str(data_var))
    if out_format == "zarr":
        encoding = _get_zarr_encoding(ds_disp, out_chunks, shard_factors=shard_factors)
        ds_disp.chunk(out_shard_dict).to_zarr(
            output_name,
            encoding=encoding,
            mode="a",
            consolidated=False,
        )
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
    shard_factors: tuple[int, int, int] | None = (1, 4, 4),
) -> dict[str, dict]:
    if shard_factors is not None:
        c1, c2, c3 = chunks
        f1, f2, f3 = shard_factors
        shards = (c1 * f1, c2 * f2, c3 * f3)
    else:
        shards = None

    encoding_per_var = {
        "compressors": [BloscCodec(cname=compression_name, clevel=compression_level)],
        "chunks": chunks,
    }
    # Only add shards if they're properly divisible
    if shards is not None:
        # Check if shards are divisible by chunks
        if all(s % c == 0 for s, c in zip(shards, chunks)):
            encoding_per_var["shards"] = shards
        else:
            msg = "Shards are not properly divisible by chunks"
            warnings.warn(msg, stacklevel=2)

    if not data_vars:
        data_vars = list(ds.data_vars)
    encoding = {}
    for var in data_vars:
        if ds[var].ndim < 2:
            continue
        var_encoding = encoding_per_var.copy()
        if ds[var].ndim == 2:
            var_encoding["chunks"] = chunks[-2:]
            if shards is not None:
                var_encoding["shards"] = shards[-2:]
        encoding[var] = var_encoding
    if not add_coords:
        return encoding
    # Handle coordinate compression
    encoding.update({var: {"compressors": []} for var in ds.coords})
    return encoding


def combine_quality_masks(
    quality_datasets: Sequence[xr.DataArray],
    thresholds: Sequence[float],
    reduction_func: Callable = np.logical_or,
) -> xr.DataArray:
    """Create a combined mask from multiple quality datasets.

    This function creates on boolean mask from multiple quality datasets
    by applying `reduction_func` on each thresholded array.

    For example, with `reduction_func=np.logical_or` and
    `thresholds=[0.5, 0.5]`, the mask will be True if any of the quality datasets
    have a value greater than 0.5.
    If you want *all* quality datasets to pass their respective thresholds,
    use `reduction_func=np.logical_and`.

    Parameters
    ----------
    quality_datasets : Sequence[xr.DataArray]
        Sequence of quality datasets.
    thresholds : Sequence[float]
        Thresholds for each quality dataset.
        Must be same length as `quality_datasets`.
    reduction_func : Callable
        Function to use for combining the quality datasets.
        Defaults to `np.logical_or`.

    Returns
    -------
    xr.DataArray
        Combined mask.

    """
    return reduce(
        reduction_func,
        [
            qd > threshold
            for qd, threshold in zip(quality_datasets, thresholds, strict=True)
        ],
    )


if __name__ == "__main__":
    tyro.cli(reformat_stack)
