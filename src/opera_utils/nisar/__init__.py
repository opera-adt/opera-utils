"""NISAR GSLC utilities for reading and analyzing geocoded SLC data."""

from __future__ import annotations

from ._download import run_download
from ._gunw_search import search_gunw
from ._info import (
    find_intersecting_frames,
    get_frame_latlon_bounds,
    get_nisar_bbox,
    load_gpkg,
    nisar_frame_info,
    plot_frames,
)
from ._product import (
    GslcProduct,
    GunwProduct,
    OrbitDirection,
    OutOfBoundsError,
    UrlType,
)
from ._remote import open_file, open_h5
from ._search import search

__all__ = [
    "GslcProduct",
    "GunwProduct",
    "OrbitDirection",
    "OutOfBoundsError",
    "UrlType",
    "find_intersecting_frames",
    "get_frame_latlon_bounds",
    "get_nisar_bbox",
    "load_gpkg",
    "nisar_frame_info",
    "open_file",
    "open_h5",
    "plot_frames",
    "run_download",
    "search",
    "search_gunw",
]
