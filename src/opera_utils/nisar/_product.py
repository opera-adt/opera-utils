"""NISAR GSLC product classes for parsing filenames and accessing data."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from typing_extensions import Self

from opera_utils.constants import (
    NISAR_FREQUENCIES,
    NISAR_GSLC_FILE_REGEX,
    NISAR_GSLC_GRIDS,
    NISAR_GSLC_IDENTIFICATION,
    NISAR_POLARIZATIONS,
)

__all__ = [
    "GslcProduct",
    "UrlType",
    "OrbitDirection",
]


class UrlType(str, Enum):
    """Choices for the URL type for product access."""

    S3 = "s3"
    HTTPS = "https"

    def __str__(self) -> str:
        return str(self.value)


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
        if not (match := NISAR_GSLC_FILE_REGEX.match(Path(name).name)):
            msg = f"Invalid NISAR GSLC filename format: {name}"
            raise ValueError(msg)

        data: dict[str, Any] = match.groupdict()
        data["start_datetime"] = _to_datetime(data["start_datetime"])
        data["end_datetime"] = _to_datetime(data["end_datetime"])
        data["cycle_number"] = int(data["cycle_number"])
        data["relative_orbit_number"] = int(data["relative_orbit_number"])
        data["track_frame_number"] = int(data["track_frame_number"])
        data["minor_version"] = int(data["minor_version"])
        data["orbit_direction"] = OrbitDirection(data["orbit_direction"])

        if Path(name).exists():
            data["size_in_bytes"] = Path(name).stat().st_size

        return cls(filename=name, **data)

    @property
    def track_frame_id(self) -> str:
        """Get the combined track and frame identifier."""
        return (
            f"{self.cycle_number:03d}_{self.relative_orbit_number:03d}_"
            f"{self.orbit_direction}_{self.track_frame_number:03d}"
        )

    @property
    def version(self) -> str:
        """Get the full version string."""
        return f"{self.major_version}.{self.minor_version:03d}"

    def get_dataset_path(
        self, frequency: str = "A", polarization: str = "HH"
    ) -> str:
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

        Raises
        ------
        ValueError
            If the frequency or polarization is invalid.

        """
        if frequency not in NISAR_FREQUENCIES:
            msg = f"Invalid frequency {frequency}. Choices: {NISAR_FREQUENCIES}"
            raise ValueError(msg)
        if polarization not in NISAR_POLARIZATIONS:
            msg = f"Invalid polarization {polarization}. Choices: {NISAR_POLARIZATIONS}"
            raise ValueError(msg)
        return f"{NISAR_GSLC_GRIDS}/frequency{frequency}/{polarization}"

    def get_available_polarizations(
        self, h5file: h5py.File, frequency: str = "A"
    ) -> list[str]:
        """Get available polarizations for a frequency from an open HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            Open HDF5 file handle.
        frequency : str
            Frequency band, either "A" or "B". Default is "A".

        Returns
        -------
        list[str]
            List of available polarization names.

        """
        freq_group = f"{NISAR_GSLC_GRIDS}/frequency{frequency}"
        if freq_group not in h5file:
            return []
        return [
            name
            for name in h5file[freq_group].keys()
            if name in NISAR_POLARIZATIONS
        ]

    def get_available_frequencies(self, h5file: h5py.File) -> list[str]:
        """Get available frequencies from an open HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            Open HDF5 file handle.

        Returns
        -------
        list[str]
            List of available frequency names ("A" and/or "B").

        """
        return [
            freq
            for freq in NISAR_FREQUENCIES
            if f"{NISAR_GSLC_GRIDS}/frequency{freq}" in h5file
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
            return {key: id_group[key][()] for key in id_group.keys()}

    @property
    def bounding_polygon(self) -> str | None:
        """Get the bounding polygon WKT from file metadata."""
        bp = self._identification_cache.get("boundingPolygon")
        if bp is not None and isinstance(bp, bytes):
            return bp.decode("utf-8")
        return bp

    def get_shape(
        self, h5file: h5py.File, frequency: str = "A", polarization: str = "HH"
    ) -> tuple[int, int]:
        """Get the shape of a GSLC dataset.

        Parameters
        ----------
        h5file : h5py.File
            Open HDF5 file handle.
        frequency : str
            Frequency band. Default is "A".
        polarization : str
            Polarization. Default is "HH".

        Returns
        -------
        tuple[int, int]
            Shape as (rows, cols).

        """
        dset_path = self.get_dataset_path(frequency, polarization)
        return h5file[dset_path].shape

    def read_subset(
        self,
        h5file: h5py.File,
        rows: slice,
        cols: slice,
        frequency: str = "A",
        polarization: str = "HH",
    ) -> np.ndarray:
        """Read a subset of the GSLC data.

        Parameters
        ----------
        h5file : h5py.File
            Open HDF5 file handle.
        rows : slice
            Row slice for subsetting.
        cols : slice
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
        dset_path = self.get_dataset_path(frequency, polarization)
        return h5file[dset_path][rows, cols]

    def __fspath__(self) -> str:
        return os.fspath(self.filename)

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
        url = _get_download_url(umm_data, protocol=url_type)
        product = GslcProduct.from_filename(url)
        archive_info = umm_data.get("DataGranule", {}).get(
            "ArchiveAndDistributionInformation", []
        )
        size_in_bytes = archive_info[0].get("SizeInBytes", 0) if archive_info else None
        product.size_in_bytes = size_in_bytes
        return product


def _get_download_url(
    umm_data: dict[str, Any], protocol: UrlType = UrlType.HTTPS
) -> str:
    """Extract a download URL from the product's UMM metadata.

    Parameters
    ----------
    umm_data : dict[str, Any]
        The product's umm metadata dictionary.
    protocol : UrlType
        The protocol to use for downloading, either "s3" or "https".

    Returns
    -------
    str
        The download URL.

    Raises
    ------
    ValueError
        If no URL with the specified protocol is found.

    """
    if protocol not in ["https", "s3"]:
        msg = f"Unknown protocol {protocol}; must be https or s3"
        raise ValueError(msg)

    for url in umm_data.get("RelatedUrls", []):
        if url["Type"].startswith("GET DATA") and url["URL"].startswith(str(protocol)):
            # For NISAR, we want the .h5 file, not the .xml or other files
            if url["URL"].endswith(".h5"):
                return url["URL"]

    # Fallback: try any URL with the protocol
    for url in umm_data.get("RelatedUrls", []):
        if url["Type"].startswith("GET DATA") and url["URL"].startswith(str(protocol)):
            return url["URL"]

    msg = f"No download URL found for granule {umm_data.get('GranuleUR', 'unknown')}"
    raise ValueError(msg)
