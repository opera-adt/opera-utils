"""Crop and interpolate OPERA TROPO products for AOI."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import rasterio as rio
import tyro
import xarray as xr
from rasterio.warp import transform_bounds

from ._helpers import (
    MissingTropoError,
    _bracket,
    _build_tropo_index,
    _create_total_delay,
    _interp_in_time,
    _open_crop,
)

logger = logging.getLogger(__name__)


def crop_tropo(
    tropo_urls_file: Path,
    datetimes: list[datetime],
    aoi_bounds: tuple[float, float, float, float] | None = None,
    file_bounds: Path | str | None = None,
    output_dir: Path = Path("cropped_tropo"),
    skip_time_interpolation: bool = False,
    height_max: float = 10000.0,
    margin_deg: float = 0.3,
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
        Path to GeoTIFF file containing bounds to crop to.
        Alternative to `aoi_bounds`.
    output_dir : Path
        Directory to save cropped TROPO products.
    skip_time_interpolation : bool
        Skip time interpolation and use nearest file.
    height_max : float
        Maximum height in meters to include in cropping.
    margin_deg : float
        Additional margin in degrees around AOI bounds.

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

    logger.info(f"Processing {len(datetimes)} datetime(s)")

    for dt in datetimes:
        dt_pandas = pd.to_datetime(dt).tz_localize(None)
        time_str = dt_pandas.strftime("%Y%m%dT%H%M%S")
        output_file = output_dir / f"tropo_cropped_{time_str}.nc"
        if output_file.exists():
            logger.info(f"Skipping existing {output_file}")
            continue

        logger.info(f"Processing datetime: {dt_pandas}")

        if not skip_time_interpolation:
            try:
                early_url, late_url = _bracket(tropo_idx_series, dt_pandas)
            except MissingTropoError:
                logger.warning(f"No available tropo files for {dt_pandas}")
                continue

            logger.info(f"Using files: {early_url}, {late_url}")

            # Open and crop the datasets
            ds0 = _open_crop(early_url, lat_bounds, lon_bounds, height_max)
            ds1 = _open_crop(late_url, lat_bounds, lon_bounds, height_max)

            # Interpolate in time
            td_interp = _interp_in_time(
                ds0,
                ds1,
                ds0.time.to_pandas().item(),
                ds1.time.to_pandas().item(),
                dt_pandas,
            )
        else:
            # TODO: could get all these at once really
            idx = tropo_idx_series.index.get_indexer([dt], method="nearest")[0]
            closest_url = tropo_idx_series.values[idx]
            # now we just want the nearest one:
            ds = _open_crop(closest_url, lat_bounds, lon_bounds, height_max)
            da_total_delay = _create_total_delay(ds)
            td_interp = ds.copy()
            td_interp["total_delay"] = da_total_delay

        # Keep only total_delay for output
        output_ds = xr.Dataset(
            {
                "total_delay": td_interp.total_delay,
                "latitude": td_interp.latitude,
                "longitude": td_interp.longitude,
                "height": td_interp.height,
            }
        )

        # Save the cropped and interpolated data
        output_ds.to_netcdf(output_file, engine="h5netcdf")
        logger.info(f"Saved: {output_file}")


def main() -> None:
    """CLI entry point for tropo-crop command."""
    tyro.cli(crop_tropo)


if __name__ == "__main__":
    main()
