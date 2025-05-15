# /// script
# dependencies = ["numpy", "rasterio", "scipy", "tqdm", "tyro"]
# ///
"""Convert a series of OPERA DISP-S1 products to a single-reference stack.

The OPERA L3 InSAR displacement values, which represent relative displacement
from a reference SAR acquisition to a secondary, have reference dates which
move forward in time. To get a single, continuous time series, a simple summation
is needed.

This script converts these files into a single continuous displacement time series.
The current format is a stack of geotiff rasters.

Usage:
python -m opera_utils.disp.rebase_reference single-reference-out/ OPERA_L3_DISP-S1_*.nc
"""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from itertools import repeat
from pathlib import Path
from typing import Any, Final, Literal

import h5py
import numpy as np
import rasterio as rio
from numpy.typing import DTypeLike
from rasterio.enums import Resampling
from tqdm.auto import trange
from typing_extensions import Self

from ._product import DispProductStack
from ._rebase import NaNPolicy
from ._utils import _last_per_ministack, flatten, round_mantissa

UINT16_MAX: Final = 65535


class DisplacementDataset(str, Enum):
    """Enumeration of displacement datasets."""

    DISPLACEMENT = "displacement"
    SHORT_WAVELENGTH = "short_wavelength_displacement"

    def __str__(self) -> str:
        return self.value


class CorrectionDataset(str, Enum):
    """Enumeration of correction datasets."""

    SOLID_EARTH_TIDE = "/corrections/solid_earth_tide"
    IONOSPHERIC_DELAY = "/corrections/ionospheric_delay"

    def __str__(self) -> str:
        return self.value


class SamePerMinistackDataset(str, Enum):
    """Enumeration of datasets that are same per ministack."""

    TEMPORAL_COHERENCE = "temporal_coherence"
    PHASE_SIMILARITY = "phase_similarity"
    PERSISTENT_SCATTERER_MASK = "persistent_scatterer_mask"
    SHP_COUNTS = "shp_counts"

    def __str__(self) -> str:
        return self.value


class QualityDataset(str, Enum):
    """Enumeration of quality datasets."""

    TIMESERIES_INVERSION_RESIDUALS = "timeseries_inversion_residuals"
    CONNECTED_COMPONENT_LABELS = "connected_component_labels"
    RECOMMENDED_MASK = "recommended_mask"
    ESTIMATED_PHASE_QUALITY = "estimated_phase_quality"
    SHP_COUNTS = "shp_counts"
    WATER_MASK = "water_mask"

    def __str__(self) -> str:
        return self.value


SAME_PER_MINISTACK_DATASETS = [
    SamePerMinistackDataset.TEMPORAL_COHERENCE,
    SamePerMinistackDataset.PHASE_SIMILARITY,
    SamePerMinistackDataset.PERSISTENT_SCATTERER_MASK,
    SamePerMinistackDataset.SHP_COUNTS,
]
UNIQUE_PER_DATE_DATASETS = [
    QualityDataset.TIMESERIES_INVERSION_RESIDUALS,
    QualityDataset.CONNECTED_COMPONENT_LABELS,
    QualityDataset.RECOMMENDED_MASK,
]

NODATA_VALUES = {
    "shp_counts": 0,
    "persistent_scatterer_mask": 255,
}


def rebase(
    output_dir: Path | str,
    nc_files: Sequence[Path | str],
    dataset: DisplacementDataset = DisplacementDataset.DISPLACEMENT,
    mask_dataset: QualityDataset | None = QualityDataset.RECOMMENDED_MASK,
    apply_solid_earth_corrections: bool = True,
    apply_ionospheric_corrections: bool = True,
    nan_policy: str | NaNPolicy = NaNPolicy.propagate,
    reference_point: tuple[int, int] | None = None,
    reference_lonlat: tuple[float, float] | None = None,
    keep_bits: int = 9,
    make_overviews: bool = True,
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
    mask_dataset : QualityDataset | None
        Name of HDF5 dataset within product to use as a mask.
        If None, no masking is performed.
    apply_solid_earth_corrections : bool
        Apply solid earth tides to the data.
        Default is True.
    apply_ionospheric_corrections : bool
        Apply ionospheric delay to the data.
        Default is True.
    nan_policy : str | NaNPolicy
        Whether to propagate or omit (zero out) NaNs in the data.
        By default "propagate", which means any ministack, or any "reference crossover"
        product, with nan at a pixel causes all subsequent data to be nan.
        If "omit", then any nan causes the pixel to be zeroed out, which is
        equivalent to assuming that 0 displacement occurred during that time.
    reference_point : tuple[int, int] | None
        The (row, column) of the reference point to use when rebasing /displacement.
        If None, finds a point with the highest harmonic mean of temporal coherence.
        Default is None.
    reference_lonlat : tuple[float, float] | None
        Reference point to use when rebasing /displacement, specified as
        (longitude, latitude) in degrees.
        Takes precedence over reference_point if both are provided.
        Default is None.
    keep_bits : int
        Number of floating point mantissa bits to retain in the output rasters.
        Default is 9.
    make_overviews : bool
        Whether to make overviews for the output rasters.
        Default is True.
    tqdm_position : int
        Position of the progress bar. Default is 0.

    """
    nc_files = sorted(nc_files)
    product_stack = DispProductStack.from_file_list(nc_files)
    # Flatten all dates, find unique sorted list of SAR epochs
    all_dates = sorted(set(flatten(product_stack.ifg_date_pairs)))

    # Create the main displacement dataset.
    writer = GeotiffStackWriter.from_dates(
        Path(output_dir),
        dataset=dataset,
        date_list=all_dates,
        keep_bits=keep_bits,
        profile=product_stack.get_rasterio_profile(),
    )
    if all(Path(f).exists() for f in writer.files):
        return

    if reference_lonlat is not None:
        reference_point = product_stack.lonlat_to_rowcol(*reference_lonlat)

    reader = HDF5StackReader(nc_files, dset_name=dataset, nodata=np.nan)
    corrections_readers = []
    if apply_solid_earth_corrections:
        corrections_readers.append(
            HDF5StackReader(
                nc_files,
                dset_name=str(CorrectionDataset.SOLID_EARTH_TIDE),
                nodata=np.nan,
            )
        )
    if apply_ionospheric_corrections:
        corrections_readers.append(
            HDF5StackReader(
                nc_files,
                dset_name=str(CorrectionDataset.IONOSPHERIC_DELAY),
                nodata=np.nan,
            )
        )
    mask_reader = (
        HDF5StackReader(
            nc_files,
            dset_name=str(mask_dataset),
            nodata=255,
        )
        if mask_dataset is not None
        else None
    )

    # Make a "cumulative offset" which adds up the phase each time theres a reference
    # date changeover.
    shape = product_stack[0].shape
    cumulative_offset = np.zeros(shape, dtype=np.float32)
    previous_displacement = np.zeros(shape, dtype=np.float32)
    current_displacement = np.zeros(shape, dtype=np.float32)
    cur_good_mask = np.ones(shape, dtype=bool)
    previous_reference_date = product_stack.products[0].reference_datetime

    for idx in trange(len(nc_files), desc="Summing dates", position=tqdm_position):
        current_displacement = reader[idx]

        # Apply corrections if needed
        for r in corrections_readers:
            current_displacement -= r[idx]

        # Apply mask if needed
        if mask_reader is not None:
            cur_good_mask[:] = mask_reader[idx].astype(bool)
            current_displacement[~cur_good_mask] = np.nan

        # Apply spatial reference point if needed
        if reference_point is not None:
            current_displacement -= current_displacement[reference_point]

        # Check for the shift in temporal reference date
        cur_ref, _cur_sec = product_stack.ifg_date_pairs[idx]
        if cur_ref != previous_reference_date:
            # e.g. we had (1,2), (1,3), now we hit (3,4)
            # So to write out (1,4), we need to add the running total
            # to the current displacement
            if nan_policy == NaNPolicy.omit:
                np.nan_to_num(previous_displacement, copy=False)
            cumulative_offset += previous_displacement
            previous_reference_date = cur_ref

        writer[idx] = current_displacement + cumulative_offset

        previous_displacement = current_displacement

    # Save the reference point(s), if used
    if reference_point is not None:
        writer.save_attr({"reference_point": json.dumps(reference_point, default=str)})
    if make_overviews:
        writer.make_overviews()


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
            mode = "r+" if Path(self.files[idx]).exists() else "w"
            with rio.open(self.files[idx], mode, **self.profile) as dst:
                dst.write(value, 1)

    def __len__(self):
        return len(self.files)

    def save_attr(self, attr: dict[str, Any], namespace: str | None = None) -> None:
        """Save metadata to all geotiff files.

        Parameters
        ----------
        attr
            Dictionary of metadata to save.
        namespace
            Namespace for metadata.

        """
        for f in self.files:
            with rio.open(f, "r+", **self.profile) as dst:
                dst.update_tags(ns=namespace, **attr)

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
            msg = "Either date_list or date_pairs must be provided"
            raise ValueError(msg)
        output_dir.mkdir(exist_ok=True, parents=True)
        return cls(files=out_paths, **kwargs)

    def make_overviews(self, resampling: Resampling = Resampling.average):
        for f in self.files:
            with rio.open(f, "r+") as dst:
                dst.build_overviews([2, 4, 8, 16, 32], resampling)


def _format_dates(*dates: datetime | date, fmt: str = "%Y%m%d", sep: str = "_") -> str:
    """Format a date pair into a string."""
    return sep.join((d.strftime(fmt)) for d in dates)


def extract_quality_layers(
    output_dir: Path,
    products: DispProductStack,
    dataset: str,
    save_mean: bool = True,
    mean_type: Literal["harmonic", "arithmetic"] = "harmonic",
    make_overviews: bool = True,
) -> None:
    """Extract quality layers from the displacement products."""
    reader = HDF5StackReader(products.filenames, dataset)
    writer = GeotiffStackWriter.from_dates(
        output_dir,
        dataset=dataset,
        date_pairs=products.ifg_date_pairs,
        profile=products.get_rasterio_profile(),
        nodata=NODATA_VALUES.get(dataset, np.nan),
    )
    if all(Path(f).exists() for f in writer.files):
        return
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
    if make_overviews:
        writer.make_overviews()

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
    from scipy import ndimage

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
    msg = f"No valid candidates found with quality above {min_quality}"
    raise ValueError(msg)


def main(
    nc_files: Sequence[Path | str],
    output_dir: Path | str,
    apply_solid_earth_corrections: bool = True,
    apply_ionospheric_corrections: bool = True,
    apply_mask: bool = True,
    nan_policy: str | NaNPolicy = NaNPolicy.propagate,
    reference_point: tuple[int, int] | None = None,
    num_workers: int = 5,
):
    """Run the rebase reference script and extract all datasets.

    Parameters
    ----------
    nc_files : list[Path]
        List of netCDF files to process.
    output_dir : Path or str
        Output directory for the processed files.
    apply_solid_earth_corrections : bool, optional
        Whether to apply solid earth corrections to the data, by default True
    apply_ionospheric_corrections : bool, optional
        Whether to apply ionospheric corrections to the data, by default True
    apply_mask : bool, optional
        Whether to apply a mask to the data, by default True
    nan_policy : choices = ["propagate", "omit"]
        Whether to propagate or omit (zero out) NaNs in the data.
        By default "propagate", which means any ministack, or any "reference crossover"
        product, with nan at a pixel causes all subsequent data to be nan.
        If "omit", then any nan causes the pixel to be zeroed out, which is
        equivalent to assuming that 0 displacement occurred during that time.
    reference_point : tuple[int, int] | None, optional
        The (row, column) to use for spatial referencing the `displacement` rasters.
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

    # Load in the coherence dataset
    # TODO: Any way to extract the name, rather than relying on this matching
    # the function `extract_quality_layers`?
    coherence_path = output_path / "average_temporal_coherence.tif"

    # Find the reference point
    if reference_point is None:
        reference_point = find_reference_point(coherence_path)

    all_products = DispProductStack.from_file_list(nc_files)
    futures: set[Future] = set()
    mask_dataset = QualityDataset.RECOMMENDED_MASK if apply_mask else None

    with ProcessPoolExecutor(num_workers) as pool:
        # Submit time series rebase function
        futures.add(
            pool.submit(
                rebase,
                output_dir,
                nc_files,
                dataset=DisplacementDataset.DISPLACEMENT,
                mask_dataset=mask_dataset,
                apply_solid_earth_corrections=apply_solid_earth_corrections,
                apply_ionospheric_corrections=apply_ionospheric_corrections,
                nan_policy=nan_policy,
                reference_point=reference_point,
            )
        )
        futures.add(
            pool.submit(
                rebase,
                output_dir,
                nc_files,
                dataset=DisplacementDataset.SHORT_WAVELENGTH,
                mask_dataset=None,
                apply_solid_earth_corrections=False,
                apply_ionospheric_corrections=False,
                nan_policy=nan_policy,
                reference_point=None,
            )
        )
        # Submit the others to extract
        for dataset in UNIQUE_PER_DATE_DATASETS:
            futures.add(
                pool.submit(
                    extract_quality_layers,
                    Path(output_dir),
                    all_products,
                    str(dataset),
                )
            )
        # Wait for all futures to complete, raising exceptions if they arrive
        while futures:
            done, futures = wait(futures, timeout=1, return_when=FIRST_EXCEPTION)
            for future in done:
                e = future.exception()
                if e is not None:
                    raise e


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
