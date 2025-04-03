from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from math import nan
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from typing_extensions import Self

from opera_utils.burst_frame_db import Bbox, get_frame_bbox
from opera_utils.constants import DISP_FILE_REGEX

__all__ = ["DispProduct", "DispProductStack"]


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
            raise ValueError(f"Invalid filename format: {name}")

        data: dict[str, Any] = match.groupdict()
        data["reference_datetime"] = _to_datetime(data["reference_datetime"])
        data["secondary_datetime"] = _to_datetime(data["secondary_datetime"])
        data["generation_datetime"] = _to_datetime(data["generation_datetime"])
        data["frame_id"] = int(data["frame_id"])

        return cls(filename=name, **data)

    @cached_property
    def _frame_bbox_result(self) -> tuple[int, Bbox]:
        return get_frame_bbox(self.frame_id)

    @property
    def epsg(self) -> int:
        return self._frame_bbox_result[0]

    @property
    def shape(self) -> tuple[int, int]:
        left, bottom, right, top = self._frame_bbox_result[1]
        return (int(round((top - bottom) / 30)), int(round((right - left) / 30)))

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

    def get_rasterio_profile(self, chunks: tuple[int, int] = (256, 256)) -> dict:
        from rasterio import transform

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
        # Add frame georeference metadata
        left, bottom, right, top = self._frame_bbox_result[1]
        profile["width"] = self.shape[1]
        profile["height"] = self.shape[0]
        profile["transform"] = transform.from_bounds(
            left, bottom, right, top, self.shape[1], self.shape[0]
        )
        profile["crs"] = f"EPSG:{self.epsg}"
        return profile


@dataclass
class DispProductStack:
    """Class for a stack of DispProducts."""

    products: list[DispProduct]

    def __post_init__(self) -> None:
        if len(self.products) == 0:
            raise ValueError("At least one product is required")
        if len(set(p.frame_id for p in self.products)) != 1:
            raise ValueError("All products must have the same frame_id")
        # Check for duplicates
        if len(set(self.ifg_date_pairs)) != len(self.products):
            version_count = Counter(p.version for p in self.products)
            msg = "All products must have unique reference and secondary dates."
            msg += f" Got {len(set(self.ifg_date_pairs))} unique pairs: "
            msg += f"but {len(self.products)} products."
            msg += f"Versions: {version_count.most_common()}"
            raise ValueError(msg)

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
    def epsg(self) -> int:
        return self.products[0].epsg

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.products),) + self.products[0].shape

    @property
    def x(self) -> np.ndarray:
        return self.products[0].x

    @property
    def y(self) -> np.ndarray:
        return self.products[0].y

    def get_rasterio_profile(self, chunks: tuple[int, int] = (256, 256)) -> dict:
        return self.products[0].get_rasterio_profile(chunks)

    def __getitem__(self, idx: int) -> DispProduct:
        return self.products[idx]
