"""Crop and OPERA TROPO products for an area of interest."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import rasterio as rio
import xarray as xr
from rasterio.warp import transform_bounds
from tqdm import tqdm

from ._helpers import (
    MissingTropoError,
    _bracket,
    _build_tropo_index,
    _create_total_delay,
    _interp_in_time,
    _open_crop,
)

logger = logging.getLogger(__name__)


def _process_one_datetime(
    dt: datetime,
    tropo_idx_series: pd.Series,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    height_max: float,
    output_dir: Path,
    skip_time_interpolation: bool,
    debug: bool = False,
) -> tuple[datetime, str]:
    """Worker: process one datetime and write output file.

    Parameters
    ----------
    dt : datetime
        Datetime to process.
    tropo_idx_series : pd.Series
        Series of TROPO product URLs/paths.
    lat_bounds : tuple[float, float]
        Latitude bounds as (north, south) in degrees.
    lon_bounds : tuple[float, float]
        Longitude bounds as (west, east) in degrees.
    height_max : float
        Maximum height in meters to include in cropping.
        Higher values with smaller atmospheric delay are ignored.
    output_dir : Path
        Directory to save cropped TROPO products.
    skip_time_interpolation : bool
        Skip time interpolation and use nearest file.
    debug : bool
        Debug mode. If True, write debug info during processing.
        Default: False.

    Returns
    -------
    tuple[datetime, str]
        (datetime, status)
        status is "ok" | "skipped" | "missing" | "error:<msg>"

    """
    try:
        dt_pandas = pd.to_datetime(dt).tz_localize(None)
        time_str = dt_pandas.strftime("%Y%m%dT%H%M%S")
        output_file = output_dir / f"tropo_cropped_{time_str}.nc"

        if output_file.exists():
            return (dt, "skipped")

        if not skip_time_interpolation:
            try:
                early_url, late_url = _bracket(tropo_idx_series, dt_pandas)
            except MissingTropoError:
                return (dt, "missing")

            if debug:
                tqdm.write(f"Cropping {early_url}")
            ds0 = _open_crop(early_url, lat_bounds, lon_bounds, height_max)
            if debug:
                tqdm.write(f"Cropping {late_url}")
            ds1 = _open_crop(late_url, lat_bounds, lon_bounds, height_max)

            td_interp = _interp_in_time(
                ds0,
                ds1,
                ds0.time.to_pandas().item(),
                ds1.time.to_pandas().item(),
                dt_pandas,
            )
        else:
            idx = tropo_idx_series.index.get_indexer([dt_pandas], method="nearest")[0]
            closest_url = tropo_idx_series.values[idx]
            ds = _open_crop(closest_url, lat_bounds, lon_bounds, height_max)
            da_total_delay = _create_total_delay(ds)
            td_interp = ds.copy()
            td_interp["total_delay"] = da_total_delay

        # Keep only total_delay and coord variables for output
        output_ds = xr.Dataset(
            {
                "total_delay": td_interp.total_delay,
                "latitude": td_interp.latitude,
                "longitude": td_interp.longitude,
                "height": td_interp.height,
            }
        )
        output_ds.to_netcdf(output_file, engine="h5netcdf")

    except Exception as e:
        # Return error to main proc; don't raise to avoid stopping all work.
        return (dt, f"error:{type(e).__name__}: {e}")
    else:
        return (dt, "ok")


def crop_tropo(
    tropo_urls_file: Path,
    datetimes: list[datetime],
    aoi_bounds: tuple[float, float, float, float] | None = None,
    file_bounds: Path | str | None = None,
    output_dir: Path = Path("cropped_tropo"),
    skip_time_interpolation: bool = False,
    height_max: float = 10000.0,
    margin_deg: float = 0.3,
    num_workers: int = 2,
) -> None:
    """Crop OPERA TROPO products to AOI and interpolate to specific datetimes.

    Parameters
    ----------
    tropo_urls_file : Path
        File containing list of TROPO product URLs/paths (one per line).
    datetimes : list[datetime]
        List of datetime objects to get corrections for.
    aoi_bounds : tuple[float, float, float, float]
        AOI bounding box as (west, south, east, north) in degrees.
    file_bounds : Path | str | None
        Path to GeoTIFF file containing bounds to crop to (alternative to `aoi_bounds`).
    output_dir : Path
        Directory to save cropped TROPO products.
    skip_time_interpolation : bool
        Skip time interpolation and use nearest file.
    height_max : float
        Maximum height in meters to include in cropping.
    margin_deg : float
        Additional margin in degrees around AOI bounds.
    num_workers : int
        Processes to use. Default: 2

    """
    output_dir.mkdir(exist_ok=True, parents=True)

    if file_bounds is not None:
        if aoi_bounds is not None:
            msg = "Cannot specify both aoi_bounds and file_bounds"
            raise ValueError(msg)

        with rio.open(file_bounds) as src:
            aoi_bounds = src.bounds
            if src.crs != "epsg:4326":
                aoi_bounds = transform_bounds(src.crs, "epsg:4326", *aoi_bounds)

    assert aoi_bounds is not None
    west, south, east, north = aoi_bounds
    # For lat, use north -> south ordering for xarray slicing
    lat_bounds = (north + margin_deg, south - margin_deg)
    lon_bounds = (west - margin_deg, east + margin_deg)

    tropo_urls = Path(tropo_urls_file).read_text(encoding="utf-8").splitlines()
    tropo_idx_series = _build_tropo_index(tropo_urls)

    if num_workers < 1:
        msg = "num_workers must be >= 1"
        raise ValueError(msg)

    logger.info(f"Processing {len(datetimes)} datetime(s) with {num_workers} worker(s)")

    # Submit all tasks up front; tqdm tracks completions.
    futures = []
    status_counts: dict[str, int] = {"ok": 0, "skipped": 0, "missing": 0, "error": 0}

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for dt in datetimes:
            futures.append(
                ex.submit(
                    _process_one_datetime,
                    dt,
                    tropo_idx_series,
                    lat_bounds,
                    lon_bounds,
                    height_max,
                    output_dir,
                    skip_time_interpolation,
                )
            )

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="TROPO crop+interp",
            unit="scene",
        ):
            dt, status = fut.result()
            if status.startswith("error:"):
                status_counts["error"] += 1
                logger.error(f"{dt}: {status}")
            else:
                status_counts[status] = status_counts.get(status, 0) + 1

    logger.info(
        "Done. "
        f"ok={status_counts['ok']}, "
        f"skipped={status_counts['skipped']}, "
        f"missing={status_counts['missing']}, "
        f"errors={status_counts['error']}"
    )
