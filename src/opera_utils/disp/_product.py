from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import cached_property
from math import nan
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
from affine import Affine
from typing_extensions import Self

from opera_utils.burst_frame_db import (
    Bbox,
    OrbitPass,
    get_frame_bbox,
    get_frame_geojson,
    get_frame_orbit_pass,
)
from opera_utils.constants import DISP_FILE_REGEX

__all__ = ["DispProduct", "DispProductStack", "UrlType"]


class UrlType(str, Enum):
    """Choices for the orbit direction of a granule."""

    S3 = "s3"
    HTTPS = "https"

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class DispProduct:
    """Class for information from one DISP-S1 production filename."""

    filename: str | Path
    sensor: str
    acquisition_mode: str
    frame_id: int
    polarization: str
    reference_datetime: datetime
    secondary_datetime: datetime
    version: str
    generation_datetime: datetime
    size_in_bytes: int | None = None

    @classmethod
    def from_filename(cls, name: Path | str) -> Self:
        """Parse a filename to create a DispProduct.

        Parameters
        ----------
        name : str or Path
            Filename to parse for OPERA DISP-S1 information.

        Returns
        -------
        DispProduct
            Parsed file information.

        Raises
        ------
        ValueError
            If the filename format is invalid.

        """

        def _to_datetime(dt: str) -> datetime:
            return datetime.strptime(dt, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)

        if not (match := DISP_FILE_REGEX.match(Path(name).name)):
            msg = f"Invalid filename format: {name}"
            raise ValueError(msg)

        data: dict[str, Any] = match.groupdict()
        data["reference_datetime"] = _to_datetime(data["reference_datetime"])
        data["secondary_datetime"] = _to_datetime(data["secondary_datetime"])
        data["generation_datetime"] = _to_datetime(data["generation_datetime"])
        data["frame_id"] = int(data["frame_id"])
        return cls(filename=name, **data)

    @cached_property
    def _frame_bbox_result(self) -> tuple[int, Bbox]:
        return get_frame_bbox(self.frame_id)

    @cached_property
    def frame_geojson(self) -> dict:
        return get_frame_geojson(frame_ids=[self.frame_id])

    @cached_property
    def orbit_pass(self) -> OrbitPass:
        return get_frame_orbit_pass(self.frame_id)[0]

    @property
    def epsg(self) -> int:
        return self._frame_bbox_result[0]

    @property
    def crs(self) -> pyproj.CRS:
        return pyproj.CRS.from_epsg(self.epsg)

    @property
    def shape(self) -> tuple[int, int]:
        left, bottom, right, top = self._frame_bbox_result[1]
        return (round((top - bottom) / 30), round((right - left) / 30))

    @cached_property
    def _coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        from ._utils import get_frame_coordinates

        return get_frame_coordinates(self.frame_id)

    @property
    def x(self) -> np.ndarray:
        return self._coordinates[0]

    @property
    def y(self) -> np.ndarray:
        return self._coordinates[1]

    @property
    def transform(self) -> Affine:
        left, _bottom, _right, top = self._frame_bbox_result[1]
        return Affine(30, 0, left, 0, -30, top)

    def get_rasterio_profile(self, chunks: tuple[int, int] = (256, 256)) -> dict:
        """Generate a `profile` usable by `rasterio.open()`."""
        profile = {
            "driver": "GTiff",
            "interleave": "band",
            "tiled": True,
            "blockysize": chunks[0],
            "blockxsize": chunks[1],
            "compress": "lzw",
            "nodata": nan,
            "dtype": "float32",
            "count": 1,
        }
        # Add frame georeferencing metadata
        left, bottom, right, top = self._frame_bbox_result[1]
        profile["width"] = self.shape[1]
        profile["height"] = self.shape[0]
        profile["transform"] = self.transform
        profile["crs"] = f"EPSG:{self.epsg}"
        return profile

    def __fspath__(self) -> str:
        return os.fspath(self.filename)

    def lonlat_to_rowcol(self: Self, lon: float, lat: float):
        """Convert the longitude and latitude (in degrees) row/column indices."""
        return lonlat_to_rowcol(self, lon, lat)

    @classmethod
    def from_umm(
        cls, umm_data: dict[str, Any], url_type: UrlType = UrlType.HTTPS
    ) -> DispProduct:
        """Construct a DispProduct instance from a raw dictionary.

        Parameters
        ----------
        umm_data : dict[str, Any]
            The raw granule UMM data from the CMR API.
        url_type : UrlType
            Type of url to use from the Product.
            "s3" for S3 URLs (direct access), "https" for HTTPS URLs.

        Returns
        -------
        Granule
            The parsed Granule instance.

        Raises
        ------
        ValueError
            If required temporal extent data is missing.

        """
        url = _get_download_url(umm_data, protocol=url_type)
        product = DispProduct.from_filename(url)
        archive_info = umm_data.get("DataGranule", {}).get(
            "ArchiveAndDistributionInformation", []
        )
        size_in_bytes = archive_info[0].get("SizeInBytes", 0) if archive_info else None
        product.size_in_bytes = size_in_bytes
        return product


@dataclass
class DispProductStack:
    """Class for a stack of DispProducts."""

    products: list[DispProduct]

    def __post_init__(self) -> None:
        if len(self.products) == 0:
            msg = "At least one product is required"
            raise ValueError(msg)
        if len({p.frame_id for p in self.products}) != 1:
            msg = "All products must have the same frame_id"
            raise ValueError(msg)
        # Check for duplicates
        if len(set(self.ifg_date_pairs)) != len(self.products):
            version_count = Counter(p.version for p in self.products)
            msg = "All products must have unique reference and secondary dates."
            msg += f" Got {len(set(self.ifg_date_pairs))} unique pairs: "
            msg += f"but {len(self.products)} products."
            msg += f"Versions: {version_count.most_common()}"
            raise ValueError(msg)
        # TODO: SORT!

    @classmethod
    def from_file_list(cls, file_list: Iterable[Path | str]) -> Self:
        return cls(
            sorted(
                [DispProduct.from_filename(f) for f in file_list],
                key=lambda p: (p.reference_datetime, p.secondary_datetime),
            )
        )

    @property
    def filenames(self) -> list[Path | str]:
        return [p.filename for p in self.products]

    @property
    def reference_dates(self) -> list[datetime]:
        return [p.reference_datetime for p in self.products]

    @property
    def secondary_dates(self) -> list[datetime]:
        return [p.secondary_datetime for p in self.products]

    @property
    def ifg_date_pairs(self) -> list[tuple[datetime, datetime]]:
        return [(p.reference_datetime, p.secondary_datetime) for p in self.products]

    @property
    def frame_id(self) -> int:
        return self.products[0].frame_id

    @property
    def orbit_pass(self) -> OrbitPass:
        return self.products[0].orbit_pass

    @property
    def transform(self) -> Affine:
        return self.products[0].transform

    @property
    def epsg(self) -> int:
        return self.products[0].epsg

    @property
    def crs(self) -> pyproj.CRS:
        return self.products[0].crs

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.products), *self.products[0].shape)

    @property
    def x(self) -> np.ndarray:
        return self.products[0].x

    @property
    def y(self) -> np.ndarray:
        return self.products[0].y

    def get_rasterio_profile(self, chunks: tuple[int, int] = (256, 256)) -> dict:
        """Generate a `profile` usable by `rasterio.open()`."""
        return self.products[0].get_rasterio_profile(chunks)

    def __getitem__(self, idx: int | slice) -> DispProduct | DispProductStack:
        if isinstance(idx, int):
            return self.products[idx]
        return self.__class__(self.products[idx])

    def __iter__(self) -> Iterator[DispProduct]:
        return iter(self.products)

    def lonlat_to_rowcol(self: Self, lon: float, lat: float):
        """Convert the longitude and latitude (in degrees) row/column indices."""
        return lonlat_to_rowcol(self.products[0], lon, lat)


def _get_download_url(
    umm_data: dict[str, Any], protocol: UrlType = UrlType.HTTPS
) -> str:
    """Extract a download URL from the product's UMM metadata.

    Parameters
    ----------
    umm_data : dict[str, Any]
        The product's umm metadata dictionary
    protocol : UrlType
        The protocol to use for downloading, either "s3" or "https"

    Returns
    -------
    str
        The download URL

    Raises
    ------
    ValueError
        If no URL with the specified protocol is found or if the protocol is invalid

    """
    if protocol not in ["https", "s3"]:
        msg = f"Unknown protocol {protocol}; must be https or s3"
        raise ValueError(msg)

    for url in umm_data["RelatedUrls"]:
        if url["Type"].startswith("GET DATA") and url["URL"].startswith(protocol):
            return url["URL"]

    msg = f"No download URL found for granule {umm_data['GranuleUR']}"
    raise ValueError(msg)


class OutOfBoundsError(ValueError):
    """Exception raised when the coordinates are out of bounds."""


def lonlat_to_rowcol(product: DispProduct, lon: float, lat: float) -> tuple[int, int]:
    """Convert lon/lat to row/col in the coordinates of `product`.

    Parameters
    ----------
    product : DispProduct
        DispProduct
    lon : float
        Longitude (in degrees) of point of interest.
    lat : float
        Latitude (in degrees) of point of interest.

    Returns
    -------
    tuple[int, int]
        Row and column indices in the raster

    """
    # Create transformer from WGS84 to the target CRS (always UTM)
    epsg = product.epsg
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True,
    )

    # Transform lon/lat to the raster's UTM coordinates
    x, y = transformer.transform(lon, lat, radians=False)

    # Apply the inverse of the UTM affine transform to get row/col
    col, row = ~product.transform * (x, y)

    # Return to nearest, then check if out of bounds
    row, col = round(row), round(col)
    if col < 0 or col >= product.shape[1] or row < 0 or row >= product.shape[0]:
        msg = (
            f"Coordinates {lon}, {lat} ({row = }, {col = }) are out of bounds for"
            f" {product}"
        )
        raise OutOfBoundsError(msg)
    return row, col
