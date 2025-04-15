#!/usr/bin/env python3
# /// script
# dependencies = ["numpy", "rasterio", "scipy", "tqdm", "tyro"]
# ///
"""Convert a series of OPERA DISP-S1 products to a single-reference stack.

The OPERA L3 InSAR displacement netCDF files have reference dates which
move forward in time. Each displacement is relative between two SAR acquisition dates.

This converts these files into a single continuous displacement time series.
The current format is a stack of geotiff rasters.

Usage:
    python -m opera_utils.disp.rebase_reference single-reference-out/ OPERA_L3_DISP-S1_*.nc
"""

import multiprocessing
from concurrent.futures import FIRST_EXCEPTION, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from itertools import repeat
from pathlib import Path
from typing import Literal, Self, Sequence

import h5py
import numpy as np
import rasterio as rio
import tyro
from numpy.typing import DTypeLike
from scipy import ndimage
from tqdm.auto import trange

from opera_utils.disp._product import DispProductStack
from opera_utils.disp._utils import _last_per_ministack, flatten, round_mantissa


class DisplacementDataset(StrEnum):
    """Enumeration of displacement datasets."""

    DISPLACEMENT = "displacement"
    SHORT_WAVELENGTH = "short_wavelength_displacement"


class CorrectionDataset(StrEnum):
    """Enumeration of correction datasets."""

    SOLID_EARTH_TIDE = "/corrections/solid_earth_tide"
    IONOSPHERIC_DELAY = "/corrections/ionospheric_delay"


class SamePerMinistackDataset(StrEnum):
    """Enumeration of datasets that are same per ministack."""

    TEMPORAL_COHERENCE = "temporal_coherence"
    PHASE_SIMILARITY = "phase_similarity"
    SHP_COUNTS = "shp_counts"


class QualityDataset(StrEnum):
    """Enumeration of quality datasets."""

    TIMESERIES_INVERSION_RESIDUALS = "timeseries_inversion_residuals"
    CONNECTED_COMPONENT_LABELS = "connected_component_labels"
    RECOMMENDED_MASK = "recommended_mask"
    ESTIMATED_PHASE_QUALITY = "estimated_phase_quality"
    SHP_COUNTS = "shp_counts"
    WATER_MASK = "water_mask"


SAME_PER_MINISTACK_DATASETS = [
    SamePerMinistackDataset.TEMPORAL_COHERENCE,
    SamePerMinistackDataset.PHASE_SIMILARITY,
    SamePerMinistackDataset.SHP_COUNTS,
]
UNIQUE_PER_DATE_DATASETS = [
    QualityDataset.TIMESERIES_INVERSION_RESIDUALS,
    QualityDataset.CONNECTED_COMPONENT_LABELS,
    QualityDataset.RECOMMENDED_MASK,
]


def rereference(
    output_dir: Path | str,
    nc_files: Sequence[Path | str],
    dataset: DisplacementDataset = DisplacementDataset.DISPLACEMENT,
    apply_corrections: bool = True,
    reference_point: tuple[int, int] | None = None,
    nodata: float = np.nan,
    keep_bits: int = 9,
    tqdm_position: int = 0,
) -> None:
    """Create a single-reference stack from a list of OPERA displacement files.

    Parameters
    ----------
    output_dir : Path or str
        File path to the output directory.
    nc_files : list[Path | str]
        One or more netCDF files, each containing a 'displacement' dataset
        for a reference_date -> secondary_date interferogram.
    dataset : DisplacementDataset
        Name of HDF5 dataset within product to convert.
    apply_corrections : bool
        Apply corrections to the data.
        Default is True.
    reference_point : tuple[int, int] | None
        Reference point to use when rebasing /displacement.
        If None, finds a point with the highest harmonic mean of temporal coherence.
        Default is None
    nodata : float
        Value to use in translated rasters as nodata value.
        Default is np.nan
    keep_bits : int
        Number of floating point mantissa bits to retain in the output rasters.
        Default is 9.
    tqdm_position : int
        Position of the progress bar. Default is 0.

    """
    products = DispProductStack.from_file_list(nc_files)
    # Flatten all dates, find unique sorted list of SAR epochs
    all_dates = sorted(set(flatten(products.ifg_date_pairs)))

    # Create the main displacement dataset.
    writer = GeotiffStackWriter.from_dates(
        Path(output_dir),
        dataset=dataset,
        date_list=all_dates,
        keep_bits=keep_bits,
        profile=products.get_rasterio_profile(),
    )

    reader = HDF5StackReader(nc_files, dset_name=dataset, nodata=nodata)
    if apply_corrections:
        corrections_readers = [
            HDF5StackReader(nc_files, dset_name=str(correction_dataset), nodata=nodata)
            for correction_dataset in CorrectionDataset
        ]
    else:
        corrections_readers = []

    # Make a "cumulative offset" which adds up the phase each time theres a reference
    # date changeover.
    shape = products[0].shape
    cumulative_offset = np.zeros(shape, dtype=np.float32)
    last_displacement = np.zeros(shape, dtype=np.float32)
    current_displacement = np.zeros(shape, dtype=np.float32)
    latest_reference_date = products[0].reference_datetime

    for idx in trange(len(nc_files), desc="Summing dates", position=tqdm_position):
        current_displacement[:] = reader[idx]

        # Apply corrections if needed
        for r in corrections_readers:
            current_displacement -= r[idx]

        # Apply spaital reference point if needed
        if reference_point is not None:
            current_displacement -= current_displacement[reference_point]

        # Check for the shift in temporal reference date
        cur_ref, _cur_sec = products.ifg_date_pairs[0]
        if cur_ref != latest_reference_date:
            # e.g. we had (1,2), (1,3), now we hit (3,4)
            # So to write out (1,4), we need to add the running total
            # to the current displacement
            latest_reference_date = cur_ref
            cumulative_offset += last_displacement
        last_displacement = current_displacement

        writer[idx] = current_displacement + cumulative_offset


@dataclass
class HDF5StackReader:
    """Reader for HDF5 datasets from multiple files."""

    file_list: Sequence[Path | str]
    dset_name: str
    nodata: float = np.nan

    def __getitem__(self, idx: int | slice) -> np.ndarray:
        if isinstance(idx, slice):
            return np.stack([self[i] for i in range(idx.start, idx.stop, idx.step)])
        with h5py.File(self.file_list[idx], "r") as hf:
            return hf[str(self.dset_name)][()]

    def read_direct(self, idx: int, dest: np.ndarray) -> None:
        with h5py.File(self.file_list[idx], "r") as hf:
            dset = hf[str(self.dset_name)]
            dset.read_direct(dest)


@dataclass
class GeotiffStackWriter:
    """Writer for GeoTIFF files."""

    files: Sequence[Path | str]
    profile: dict | None = None
    like_filename: Path | str | None = None
    nodata: float = np.nan
    keep_bits: int | None = 9
    dtype: DTypeLike | None = None

    def __post_init__(self):
        if self.profile is None:
            self.profile = {}
        if self.like_filename is not None:
            with rio.open(self.like_filename) as src:
                self.profile.update(src.profile)

        self.profile["count"] = 1
        self.profile["driver"] = "GTiff"
        if self.dtype is not None:
            self.profile["dtype"] = np.dtype(self.dtype)
        if self.nodata is not None:
            self.profile["nodata"] = self.nodata

    def __setitem__(self, key, value):
        # Check if we have a floating point raster
        if self.keep_bits is not None and np.issubdtype(value.dtype, np.floating):
            round_mantissa(value, keep_bits=self.keep_bits)

        if isinstance(key, slice):
            keys = list(range(key.start, key.stop, key.step))
        elif isinstance(key, int):
            keys = [key]
        else:
            keys = list(key)

        for idx in keys:
            with rio.open(self.files[idx], "w", **self.profile) as dst:
                dst.write(value, 1)

    def __len__(self):
        return len(self.files)

    @classmethod
    def from_dates(
        cls,
        output_dir: Path,
        dataset: str,
        date_list: Sequence[datetime] | None = None,
        date_pairs: Sequence[tuple[datetime, datetime]] | None = None,
        suffix: str = ".tif",
        **kwargs,
    ) -> Self:
        if date_list:
            ref_date = date_list[0]

            out_paths = [
                output_dir / f"{dataset}_{_format_dates(ref_date, d)}{suffix}"
                for d in date_list[1:]
            ]
        elif date_pairs:
            out_paths = [
                output_dir / f"{dataset}_{_format_dates(*d)}{suffix}"
                for d in date_pairs
            ]
        else:
            raise ValueError("Either date_list or date_pairs must be provided")
        output_dir.mkdir(exist_ok=True, parents=True)
        return cls(files=out_paths, **kwargs)


def _format_dates(*dates: datetime | date, fmt: str = "%Y%m%d", sep: str = "_") -> str:
    """Format a date pair into a string."""
    return sep.join((d.strftime(fmt)) for d in dates)


def extract_quality_layers(
    output_dir: Path,
    products: DispProductStack,
    dataset: str,
    save_mean: bool = True,
    mean_type: Literal["harmonic", "arithmetic"] = "harmonic",
):
    """Extract quality layers from the displacement products and write them to GeoTIFF files."""
    reader = HDF5StackReader(products.filenames, dataset)
    writer = GeotiffStackWriter.from_dates(
        output_dir,
        dataset=dataset,
        date_pairs=products.ifg_date_pairs,
        profile=products.get_rasterio_profile(),
    )
    if save_mean:
        # For harmonic mean, we need to accumulate the reciprocals
        if mean_type == "harmonic":
            # Start with zeros, will replace with sum of reciprocals
            reciprocal_sum = np.zeros(reader[0].shape, dtype=np.float32)
            valid_count = np.zeros(reader[0].shape, dtype=np.int32)
        else:  # arithmetic mean
            cumulative = np.zeros(reader[0].shape, dtype=reader[0].dtype)

        writer_average = GeotiffStackWriter(
            files=[output_dir / f"average_{dataset}.tif"],
            profile=products.get_rasterio_profile(),
        )

    for idx in trange(len(products.filenames)):
        cur_data = reader[idx]
        writer[idx] = cur_data

        if save_mean:
            if mean_type == "harmonic":
                # Handle zeros or negative values to avoid division problems
                valid_mask = cur_data > 0
                reciprocal_sum[valid_mask] += 1.0 / cur_data[valid_mask]
                valid_count[valid_mask] += 1
            else:  # arithmetic mean
                cumulative += cur_data

    if save_mean:
        if mean_type == "harmonic":
            # Avoid division by zero by masking where valid_count is 0
            mask = valid_count > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                average = valid_count / reciprocal_sum
        else:  # arithmetic mean
            mask = cumulative > 0
            average = cumulative / len(products.filenames)

        # Set invalid areas to NaN and write
        writer_average[0] = np.where(mask, average, np.nan)


def find_reference_point(
    average_quality_raster: Path | str,
    max_quality_test: float = 0.95,
    min_quality: float = 0.5,
    step: float = 0.05,
) -> tuple[int, int]:
    """Choose a high quality (row, column) as the reference point.

    This function finds a point with the highest harmonic mean of temporal coherence.
    It steps back from high to low and picks a pixel meeting a threshold that is
    toward the center of mass of the good data pixels.

    Parameters
    ----------
    average_quality_raster : Path | str
        Path to the average quality raster.
    max_quality_test : float, optional
        Maximum quality threshold to test, by default 0.95
    min_quality : float, optional
        Minimum quality threshold, by default 0.5
    step : float, optional
        Step size for quality threshold, by default 0.05

    Returns
    -------
    tuple[int, int]
        Reference point (row, column)
    """
    with rio.open(average_quality_raster) as src:
        quality = src.read(1)

    # Step back from high to low and pick all points meeting a threshold.
    # We don't just want the highest quality, isolated pixel:
    # We want one toward the middle of the good data pixels.
    # TODO: implement median of all candidates as a `reference_mask`
    for threshold in np.arange(max_quality_test, min_quality - step, -step):
        candidates = quality > threshold
        if not np.any(candidates):
            continue
        # Find the pixel closest to the center of mass of the good pixels.
        r, c = ndimage.center_of_mass(candidates)
        # The center may not be a good pixel! So find the candidate closest:
        rows, cols = np.where(candidates)
        dists = np.sqrt((rows - r) ** 2 + (cols - c) ** 2)
        idx = np.argmin(dists)
        return rows[idx], cols[idx]
    raise ValueError(f"No valid candidates found with quality above {min_quality}")


def main(
    nc_files: list[str],
    output_dir: Path | str,
    apply_corrections: bool = True,
    reference_point: tuple[int, int] | None = None,
    num_workers: int = 5,
):
    """Run the rebase reference script and extract all datasets.

    Parameters
    ----------
    nc_files : list[str]
        List of netCDF files to process.
    output_dir : Path or str
        Output directory for the processed files.
    apply_corrections : bool, optional
        Whether to apply corrections to the data, by default True
    reference_point : tuple[int, int] | None, optional
        Reference point to use when rebasing /displacement.
        If None, finds a point with the highest harmonic mean of temporal coherence.
        Default is None.
    num_workers : int, optional
        Number of workers to use, by default 5
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Transfer the water mask
    water_mask_path = output_path / "water_mask.tif"
    if not water_mask_path.exists():
        with rio.open(f"NETCDF:{nc_files[0]}:/water_mask") as src:
            profile = src.profile | {"driver": "GTiff"}
            with rio.open(water_mask_path, "w", **profile) as dst:
                dst.write(src.read(1), 1)

    # Transfer the quality layers (no rebasing needed)
    last_per_ministack_products = DispProductStack.from_file_list(
        _last_per_ministack(nc_files)
    )
    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(
            extract_quality_layers,
            zip(
                repeat(output_dir),
                repeat(last_per_ministack_products),
                SAME_PER_MINISTACK_DATASETS,
                repeat(True),
            ),
        )

    # Find the reference point
    # Load in the coherence dataset
    # TODO: Any way to extract the name, rather than relying on this matching
    # the function `extract_quality_layers`?
    coherence_path = output_path / "average_temporal_coherence.tif"

    # Find the reference point
    reference_point = find_reference_point(coherence_path)

    all_products = DispProductStack.from_file_list(nc_files)
    futures: list[Future] = []
    with ProcessPoolExecutor(num_workers) as pool:
        # Submit time series rebase function
        futures.append(
            pool.submit(
                rereference,
                output_dir,
                nc_files,
                DisplacementDataset.DISPLACEMENT,
                apply_corrections=apply_corrections,
                reference_point=reference_point,
            )
        )
        # Submit the others to extract
        for dataset in UNIQUE_PER_DATE_DATASETS:
            futures.append(
                pool.submit(
                    extract_quality_layers,
                    Path(output_dir),
                    all_products,
                    str(dataset),
                )
            )
        # Wait for all futures to complete
        wait(futures, return_when=FIRST_EXCEPTION)


if __name__ == "__main__":
    tyro.cli(main)
