"""NISAR GSLC product classes for parsing filenames and accessing data."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyproj
from typing_extensions import Self

from opera_utils._cmr import get_download_url
from opera_utils._remote import open_h5
from opera_utils.constants import NISAR_SDS_FILE_REGEX, UrlType

# NISAR GSLC HDF5 dataset paths
NISAR_GSLC_ROOT = "/science/LSAR/GSLC"
NISAR_GSLC_GRIDS = f"{NISAR_GSLC_ROOT}/grids"
NISAR_GSLC_IDENTIFICATION = "/science/LSAR/identification"

# Valid polarizations and frequencies for NISAR GSLC
NISAR_POLARIZATIONS = ("HH", "VV", "HV", "VH", "RH", "RV", "LH", "LV")
NISAR_FREQUENCIES = ("A", "B")

__all__ = [
    "GslcProduct",
    "OrbitDirection",
    "OutOfBoundsError",
    "UrlType",
]


class OutOfBoundsError(ValueError):
    """Exception raised when coordinates are outside the image bounds."""


class OrbitDirection(str, Enum):
    """Choices for the orbit direction of a granule."""

    ASCENDING = "A"
    DESCENDING = "D"

    def __str__(self) -> str:
        return str(self.value)


def _to_datetime(dt: str) -> datetime:
    """Parse NISAR datetime string format (YYYYMMDDThhmmss)."""
    return datetime.strptime(dt, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


@dataclass
class GslcProduct:
    """Class for information from one NISAR GSLC product filename."""

    filename: str | Path
    project: str
    level: str
    mode: str
    product_type: str
    cycle_number: int
    relative_orbit_number: int
    orbit_direction: OrbitDirection
    track_frame_number: int
    subswath_id: str
    polarizations: str
    look_direction: str
    start_datetime: datetime
    end_datetime: datetime
    composite_release_id: str
    processing_level: str
    coverage_indicator: str
    major_version: str
    minor_version: int
    size_in_bytes: int | None = None

    @classmethod
    def from_filename(cls, name: Path | str) -> Self:
        """Parse a filename to create a GslcProduct.

        Parameters
        ----------
        name : str or Path
            Filename to parse for NISAR GSLC information.

        Returns
        -------
        GslcProduct
            Parsed file information.

        Raises
        ------
        ValueError
            If the filename format is invalid.

        """
        if not (match := re.match(NISAR_SDS_FILE_REGEX, Path(name).name)):
            msg = f"Invalid NISAR GSLC filename format: {name}"
            raise ValueError(msg)

        g = match.groupdict()
        data: dict[str, Any] = {
            "project": g["project"],
            "level": g["level"],
            "mode": g["processing_type"],
            "product_type": g["product_type"],
            "cycle_number": int(g["cycle"]),
            "relative_orbit_number": int(g["relative_orbit"]),
            "orbit_direction": OrbitDirection(g["direction"]),
            "track_frame_number": int(g["frame"]),
            "subswath_id": g["scene_id"],
            "polarizations": g["polarization_mode"],
            "look_direction": g["freq_pol"],
            "start_datetime": _to_datetime(g["start_datetime"]),
            "end_datetime": _to_datetime(g["end_datetime"]),
            "composite_release_id": g["crid"],
            "processing_level": g["field1"],
            "coverage_indicator": g["field2"],
            "major_version": g["field3"],
            "minor_version": int(g["counter"]),
        }

        if Path(name).exists():
            data["size_in_bytes"] = Path(name).stat().st_size

        return cls(filename=name, **data)

    @property
    def track_frame_id(self) -> str:
        """Get the combined track and frame identifier.

        Format is ``RRR_D_TTT`` where RRR = relative orbit number,
        D = orbit direction (A/D), TTT = track frame number.
        These three fields are constant across repeat-pass acquisitions,
        so this ID uniquely identifies a geographic footprint.
        """
        return (
            f"{self.relative_orbit_number:03d}_"
            f"{self.orbit_direction}_{self.track_frame_number:03d}"
        )

    @property
    def version(self) -> str:
        """Get the full version string."""
        return f"{self.major_version}.{self.minor_version:03d}"

    @contextmanager
    def _open(self) -> Iterator[h5py.File]:
        """Open the HDF5 file (local or remote) as a context manager."""
        with open_h5(str(self.filename)) as hf:
            yield hf

    def get_dataset_path(self, frequency: str = "A", polarization: str = "HH") -> str:
        """Get the HDF5 dataset path for a specific frequency and polarization.

        Parameters
        ----------
        frequency : str
            Frequency band, either "A" or "B". Default is "A".
        polarization : str
            Polarization, e.g. "HH", "VV", "HV", "VH". Default is "HH".

        Returns
        -------
        str
            The HDF5 dataset path.

        """
        if frequency not in NISAR_FREQUENCIES:
            msg = f"Invalid frequency {frequency}. Choices: {NISAR_FREQUENCIES}"
            raise ValueError(msg)
        if polarization not in NISAR_POLARIZATIONS:
            msg = f"Invalid polarization {polarization}. Choices: {NISAR_POLARIZATIONS}"
            raise ValueError(msg)
        return f"{NISAR_GSLC_GRIDS}/frequency{frequency}/{polarization}"

    def get_available_polarizations(self, frequency: str = "A") -> list[str]:
        """Get available polarizations for a frequency.

        Parameters
        ----------
        frequency : str
            Frequency band, either "A" or "B". Default is "A".

        Returns
        -------
        list[str]
            List of available polarization names.

        """
        freq_group = f"{NISAR_GSLC_GRIDS}/frequency{frequency}"
        with self._open() as hf:
            group = hf.get(freq_group)
            if group is None:
                return []
            return [name for name in NISAR_POLARIZATIONS if name in group]

    def get_available_frequencies(self) -> list[str]:
        """Get available frequencies from the HDF5 file.

        Returns
        -------
        list[str]
            List of available frequency names ("A" and/or "B").

        """
        with self._open() as hf:
            return [
                freq
                for freq in NISAR_FREQUENCIES
                if f"{NISAR_GSLC_GRIDS}/frequency{freq}" in hf
            ]

    @cached_property
    def _identification_cache(self) -> dict[str, Any]:
        """Cache identification metadata from the file."""
        if not Path(self.filename).exists():
            return {}
        with h5py.File(self.filename) as hf:
            if NISAR_GSLC_IDENTIFICATION not in hf:
                return {}
            id_group = hf[NISAR_GSLC_IDENTIFICATION]
            return {key: id_group[key][()] for key in id_group}

    @property
    def bounding_polygon(self) -> str | None:
        """Get the bounding polygon WKT from file metadata."""
        bp = self._identification_cache.get("boundingPolygon")
        if bp is not None and isinstance(bp, bytes):
            return bp.decode("utf-8")
        return bp

    def get_shape(
        self,
        frequency: str = "A",
        polarization: str = "HH",
    ) -> tuple[int, int]:
        """Get the shape of a GSLC dataset.

        Parameters
        ----------
        frequency : str
            Frequency band. Default is "A".
        polarization : str
            Polarization. Default is "HH".

        Returns
        -------
        tuple[int, int]
            Shape as (rows, cols).

        """
        with self._open() as hf:
            dset_path = self.get_dataset_path(frequency, polarization)
            return hf[dset_path].shape  # type: ignore[return-value]

    def read_subset(
        self,
        rows: slice | int | None,
        cols: slice | int | None,
        frequency: str = "A",
        polarization: str = "HH",
    ) -> np.ndarray:
        """Read a subset of the GSLC data.

        Parameters
        ----------
        rows : slice | int | None
            Row slice for subsetting.
        cols : slice | int | None
            Column slice for subsetting.
        frequency : str
            Frequency band. Default is "A".
        polarization : str
            Polarization. Default is "HH".

        Returns
        -------
        np.ndarray
            The subset of complex GSLC data.

        """
        if rows is None:
            rows = slice(None)
        if cols is None:
            cols = slice(None)

        with self._open() as hf:
            dset_path = self.get_dataset_path(frequency, polarization)
            return hf[dset_path][rows, cols]

    def __fspath__(self) -> str:
        return os.fspath(self.filename)

    def get_epsg(self, frequency: str = "A") -> int:
        """Get the EPSG code for the coordinate system.

        Parameters
        ----------
        frequency : str
            Frequency band. Default is "A".

        Returns
        -------
        int
            The EPSG code for the coordinate system.

        """
        freq_path = f"{NISAR_GSLC_GRIDS}/frequency{frequency}"
        with self._open() as hf:
            return int(hf[freq_path]["projection"][()])  # type: ignore[index]

    def get_coordinates(self, frequency: str = "A") -> tuple[np.ndarray, np.ndarray]:
        """Get the x and y coordinate arrays.

        Parameters
        ----------
        frequency : str
            Frequency band. Default is "A".

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The (x_coordinates, y_coordinates) arrays.

        """
        freq_path = f"{NISAR_GSLC_GRIDS}/frequency{frequency}"
        with self._open() as hf:
            x_coords = hf[freq_path]["xCoordinates"][:]  # type: ignore[index]
            y_coords = hf[freq_path]["yCoordinates"][:]  # type: ignore[index]
            return x_coords, y_coords

    def lonlat_to_rowcol(
        self,
        lon: float,
        lat: float,
        frequency: str = "A",
    ) -> tuple[int, int]:
        """Convert lon/lat to row/col indices.

        Parameters
        ----------
        lon : float
            Longitude in degrees.
        lat : float
            Latitude in degrees.
        frequency : str
            Frequency band. Default is "A".

        Returns
        -------
        tuple[int, int]
            Row and column indices.

        Raises
        ------
        OutOfBoundsError
            If the coordinates are outside the image bounds.

        """
        with self._open() as hf:
            freq_path = f"{NISAR_GSLC_GRIDS}/frequency{frequency}"
            epsg = int(hf[freq_path]["projection"][()])  # type: ignore[index]
            x_coords = hf[freq_path]["xCoordinates"][:]  # type: ignore[index]
            y_coords = hf[freq_path]["yCoordinates"][:]  # type: ignore[index]

        # Transform lon/lat to projected coordinates
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", f"EPSG:{epsg}", always_xy=True
        )
        x, y = transformer.transform(lon, lat)

        # Find the nearest column (x_coords are typically increasing)
        col = int(np.searchsorted(x_coords, x))
        # y_coords are typically decreasing (north to south)
        # searchsorted expects increasing, so we search in reverse
        if y_coords[0] > y_coords[-1]:
            row = int(np.searchsorted(-y_coords, -y))
        else:
            row = int(np.searchsorted(y_coords, y))

        # Bounds check
        if col < 0 or col >= len(x_coords) or row < 0 or row >= len(y_coords):
            msg = (
                f"Coordinates ({lon}, {lat}) -> ({row=}, {col=}) are out of bounds. "
                f"Image size: ({len(y_coords)}, {len(x_coords)})"
            )
            raise OutOfBoundsError(msg)

        return row, col

    @classmethod
    def from_umm(
        cls, umm_data: dict[str, Any], url_type: UrlType = UrlType.HTTPS
    ) -> GslcProduct:
        """Construct a GslcProduct instance from a raw CMR UMM dictionary.

        Parameters
        ----------
        umm_data : dict[str, Any]
            The raw granule UMM data from the CMR API.
        url_type : UrlType
            Type of url to use from the Product.
            "s3" for S3 URLs (direct access), "https" for HTTPS URLs.

        Returns
        -------
        GslcProduct
            The parsed GslcProduct instance.

        """
        # For NISAR, prefer .h5 files over .xml or other ancillary files
        url = get_download_url(umm_data, protocol=url_type, filename_suffix=".h5")
        product = GslcProduct.from_filename(url)
        archive_info = umm_data.get("DataGranule", {}).get(
            "ArchiveAndDistributionInformation", []
        )
        size_in_bytes = archive_info[0].get("SizeInBytes", 0) if archive_info else None
        product.size_in_bytes = size_in_bytes
        return product
