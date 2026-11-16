"""Search and download OPERA L3 DISP-S1-STATIC layers (one granule per frame).

Each static granule holds three GeoTIFFs in the frame's UTM grid:

- ``line_of_sight_enu``: 3 bands, east / north / up components of the unit vector
  from the ground to the satellite;
- ``dem``: the DEM used in processing;
- ``layover_shadow_mask``.

The files are tiled, so a bounding box is read as a window over HTTPS (or S3
in-region) instead of downloading whole frames.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import rasterio
import requests
import tyro
from pyproj import Transformer
from rasterio.windows import Window, from_bounds
from shapely import from_wkt

from opera_utils._remote import get_https_fs, get_s3_fs
from opera_utils.constants import UrlType

logger = logging.getLogger("opera_utils")

COLLECTION = "OPERA_L3_DISP-S1-STATIC_V1"
LAYERS = ("line_of_sight_enu", "dem", "layover_shadow_mask")
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"

__all__ = ["StaticGranule", "download_static", "search_static"]


@dataclass
class StaticGranule:
    """One DISP-S1-STATIC granule (a frame) and the URLs of its layers."""

    granule: str
    frame_id: int
    direction: str | None
    urls: dict[str, str] = field(default_factory=dict)  # layer name -> url


def _parse_granule(item: dict, url_type: UrlType) -> StaticGranule:
    umm = item["umm"]
    attrs = {
        a["Name"]: a.get("Values", [None])[0]
        for a in umm.get("AdditionalAttributes", [])
    }
    prefix = "s3://" if url_type == UrlType.S3 else "https://"
    urls: dict[str, str] = {}
    for related in umm.get("RelatedUrls", []):
        url = related.get("URL", "")
        if not url.startswith(prefix) or not url.endswith(".tif"):
            continue
        for layer in LAYERS:
            if url.endswith(f"_{layer}.tif"):
                urls[layer] = url
    direction = attrs.get("ASCENDING_DESCENDING")
    return StaticGranule(
        granule=umm["GranuleUR"],
        frame_id=int(attrs["FRAME_NUMBER"]),
        direction=direction.lower() if isinstance(direction, str) else None,
        urls=urls,
    )


def search_static(
    frame_id: int,
    url_type: UrlType = UrlType.HTTPS,
    session: requests.Session | None = None,
) -> list[StaticGranule]:
    """Find the DISP-S1-STATIC granule(s) of a frame in CMR (normally one)."""
    params: dict[str, str | int] = {
        "short_name": COLLECTION,
        "attribute[]": f"int,FRAME_NUMBER,{frame_id}",
        "page_size": 20,
    }
    response = (session or requests).get(CMR_URL, params=params, timeout=60)
    response.raise_for_status()
    items = response.json().get("items", [])
    granules = [_parse_granule(item, url_type) for item in items]
    return [g for g in granules if g.frame_id == frame_id]


def _bbox_in_crs(
    bbox: tuple[float, float, float, float], crs
) -> tuple[float, float, float, float]:
    west, south, east, north = bbox
    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = to_crs.transform([west, east, east, west], [south, south, north, north])
    return min(xs), min(ys), max(xs), max(ys)


def _window(src, bbox, pad_pixels: int) -> Window:
    full = Window(0, 0, src.width, src.height)
    if bbox is None:
        return full
    window = from_bounds(*_bbox_in_crs(bbox, src.crs), src.transform)
    # Whole pixels covering the bbox (explicit: Window rounding APIs differ by version).
    col0 = math.floor(window.col_off) - pad_pixels
    row0 = math.floor(window.row_off) - pad_pixels
    col1 = math.ceil(window.col_off + window.width) + pad_pixels
    row1 = math.ceil(window.row_off + window.height) + pad_pixels
    return Window(col0, row0, col1 - col0, row1 - row0).intersection(full)


def download_static(
    frame_id: int,
    bbox: tuple[float, float, float, float] | None = None,
    wkt: str | None = None,
    output_dir: Path = Path("./static"),
    layers: tuple[str, ...] = LAYERS,
    url_type: UrlType = UrlType.HTTPS,
    pad_pixels: int = 2,
) -> dict[str, Path]:
    """Download (a window of) the DISP-S1-STATIC layers of a frame.

    Parameters
    ----------
    frame_id : int
        OPERA DISP-S1 frame number.
    bbox : tuple[float, float, float, float] | None
        (west, south, east, north) in degrees; None downloads the whole frame.
    wkt : str | None
        Alternative to `bbox`: the bounds of this WKT geometry are used.
    output_dir : Path
        Where the GeoTIFFs are written (``<granule>_<layer>.tif``).
    layers : tuple[str, ...]
        Any of "line_of_sight_enu", "dem", "layover_shadow_mask".
    url_type : UrlType
        HTTPS (Earthdata login from ~/.netrc or EARTHDATA_* variables) or S3
        (in-region only).
    pad_pixels : int
        Extra pixels around the window, so the subset covers the bbox edges.

    Returns
    -------
    dict[str, Path]
        Layer name -> written file.

    """
    if bbox is not None and wkt is not None:
        msg = "Can't provide both `bbox` and `wkt`"
        raise ValueError(msg)
    if wkt is not None:
        bbox = from_wkt(wkt).bounds
    unknown = set(layers) - set(LAYERS)
    if unknown:
        msg = f"Unknown static layers: {sorted(unknown)}; choose from {LAYERS}"
        raise ValueError(msg)

    granules = search_static(frame_id, url_type=url_type)
    if not granules:
        msg = f"No {COLLECTION} granule found for frame {frame_id}"
        raise ValueError(msg)
    granule = granules[0]
    fs = get_s3_fs() if url_type == UrlType.S3 else get_https_fs()

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for layer in layers:
        url = granule.urls.get(layer)
        if url is None:
            logger.warning("Frame %s: no %s in %s", frame_id, layer, granule.granule)
            continue
        out_path = output_dir / f"{granule.granule}_{layer}.tif"
        with rasterio.open(url, opener=fs.open) as src:
            window = _window(src, bbox, pad_pixels)
            data = src.read(window=window)
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                width=int(window.width),
                height=int(window.height),
                transform=src.window_transform(window),
                compress="deflate",
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
            descriptions = src.descriptions
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
            for i, desc in enumerate(descriptions, start=1):
                if desc:
                    dst.set_band_description(i, desc)
        written[layer] = out_path
        logger.info("Frame %s: wrote %s %s", frame_id, layer, out_path)
    return written


def main() -> None:
    """Command line: ``python -m opera_utils.disp._static --frame-id 8882 ...``."""
    tyro.cli(download_static)


if __name__ == "__main__":
    main()
