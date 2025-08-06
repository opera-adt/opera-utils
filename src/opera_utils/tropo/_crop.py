"""Crop and interpolate OPERA TROPO products for AOI."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import tyro
import xarray as xr

from ._helpers import (
    MissingTropoError,
    _bracket,
    _build_tropo_index,
    _interp_in_time,
    _open_crop,
)

logger = logging.getLogger(__name__)


def crop_tropo(
    tropo_urls_file: Path,
    aoi_bounds: tuple[float, float, float, float],
    datetimes: list[datetime],
    output_dir: Path = Path("cropped_tropo"),
    height_max: float = 10000.0,
    height_margin: float = 500.0,
) -> None:
    """Crop OPERA TROPO products to AOI and interpolate to specific datetimes.

    Parameters
    ----------
    tropo_urls_file : Path
        File containing list of TROPO product URLs/paths (one per line).
    aoi_bounds : tuple[float, float, float, float]
        AOI bounding box as (west, south, east, north) in degrees.
    datetimes : list[datetime]
        List of datetime objects to get corrections for.
    output_dir : Path
        Directory to save cropped TROPO products.
    height_max : float
        Maximum height in meters to include in cropping.
    height_margin : float
        Additional height margin in meters above height_max.

    """
    output_dir.mkdir(exist_ok=True, parents=True)

    west, south, east, north = aoi_bounds
    lat_bounds = (north, south)  # north, south for xarray slicing
    lon_bounds = (west, east)  # west, east

    tropo_urls = Path(tropo_urls_file).read_text(encoding="utf-8").splitlines()
    tropo_idx_series = _build_tropo_index(tropo_urls)

    logger.info(f"Processing {len(datetimes)} datetime(s)")

    for dt in datetimes:
        dt_pandas = pd.to_datetime(dt).tz_localize(None)
        logger.info(f"Processing datetime: {dt_pandas}")

        try:
            early_url, late_url = _bracket(tropo_idx_series, dt_pandas)
        except MissingTropoError:
            logger.warning(f"No available tropo files for {dt_pandas}")
            continue

        logger.info(f"Using files: {early_url}, {late_url}")

        # Open and crop the datasets
        ds0 = _open_crop(early_url, lat_bounds, lon_bounds, height_max, height_margin)
        ds1 = _open_crop(late_url, lat_bounds, lon_bounds, height_max, height_margin)

        # Interpolate in time
        td_interp = _interp_in_time(
            ds0,
            ds1,
            ds0.time.to_pandas().item(),
            ds1.time.to_pandas().item(),
            dt_pandas,
        )

        # Save the cropped and interpolated data
        time_str = dt_pandas.strftime("%Y%m%dT%H%M%S")
        output_file = output_dir / f"tropo_cropped_{time_str}.nc"

        # Keep only total_delay for output
        output_ds = xr.Dataset(
            {
                "total_delay": td_interp.total_delay,
                "latitude": td_interp.latitude,
                "longitude": td_interp.longitude,
                "height": td_interp.height,
            }
        )

        output_ds.to_netcdf(output_file, engine="h5netcdf")
        logger.info(f"Saved: {output_file}")


def main() -> None:
    """CLI entry point for tropo-crop command."""
    tyro.cli(crop_tropo)


if __name__ == "__main__":
    main()
