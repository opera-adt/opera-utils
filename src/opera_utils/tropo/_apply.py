"""Apply tropospheric corrections to DEM using LOS geometry."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rxr
import tyro
import xarray as xr
from rasterio.enums import Resampling
from scipy.interpolate import RegularGridInterpolator

from ._helpers import (
    MissingTropoError,
    _bracket,
    _build_tropo_index,
    _interp_in_time,
    _open_2d,
    _open_crop,
)

logger = logging.getLogger(__name__)


def _apply_los_correction(
    zenith_delay: xr.DataArray,
    los_enu: xr.DataArray,
) -> xr.DataArray:
    """Apply line-of-sight correction to zenith delay.

    Parameters
    ----------
    zenith_delay : xr.DataArray
        Zenith tropospheric delay in meters.
    los_enu : xr.DataArray
        3-band LOS unit vector with bands [E, N, U].

    Returns
    -------
    xr.DataArray
        LOS tropospheric delay in meters.

    """
    # Extract the up component (band 3, index 2)
    los_up = los_enu.isel(band=2)  # U component

    # Apply cosine correction: LOS_delay = zenith_delay / cos(incidence_angle)
    # where cos(incidence_angle) = los_up (the vertical component)
    # Avoid division by very small numbers
    los_up_safe = los_up.where(np.abs(los_up) > 0.01, 0.01)

    los_delay = zenith_delay / los_up_safe
    return los_delay


def _height_to_dem_surface(
    td_3d: xr.DataArray,
    dem: xr.DataArray,
    method: str = "linear",
) -> xr.DataArray:
    """Interpolate 3D tropospheric delay to DEM surface heights.

    Parameters
    ----------
    td_3d : xr.DataArray
        3D tropospheric delay with dimensions (height, lat/y, lon/x).
    dem : xr.DataArray
        DEM with surface heights.
    method : str
        Interpolation method for RegularGridInterpolator.

    Returns
    -------
    xr.DataArray
        2D tropospheric delay at DEM surface.

    """
    # Ensure consistent coordinate naming
    if "latitude" in td_3d.dims:
        td_3d = td_3d.rename(latitude="y", longitude="x")

    # Reproject to DEM CRS if needed
    if dem.rio.crs and td_3d.rio.crs != dem.rio.crs:
        if td_3d.rio.crs is None:
            td_3d = td_3d.rio.write_crs("epsg:4326")

        td_utm = td_3d.rio.reproject(dem.rio.crs, resampling=Resampling.cubic)
        # Trim edges to avoid interpolation artifacts
        td_utm = td_utm.isel(x=slice(2, -2), y=slice(2, -2))
    else:
        td_utm = td_3d

    # Set up RegularGridInterpolator
    rgi = RegularGridInterpolator(
        (td_utm.height.values, td_utm.y.values, td_utm.x.values),
        td_utm.values,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    # Create coordinate meshes for interpolation
    yy, xx = np.meshgrid(dem.y, dem.x, indexing="ij")

    # Interpolate to DEM surface
    interp_points = np.column_stack(
        [
            dem.values.ravel(),  # height values from DEM
            yy.ravel(),  # y coordinates
            xx.ravel(),  # x coordinates
        ]
    )

    interp_values = rgi(interp_points)

    # Reshape and create output array
    out = dem.copy()
    out.values[:] = interp_values.reshape(dem.shape).astype("float32")

    return out


def apply_tropo(
    tropo_urls_file: Path,
    dem_path: Path,
    los_path: Path,
    datetimes: list[datetime],
    output_dir: Path = Path("tropo_corrections"),
    height_margin: float = 500.0,
) -> None:
    """Apply tropospheric corrections using DEM and LOS geometry.

    Parameters
    ----------
    tropo_urls_file : Path
        File containing list of TROPO product URLs/paths (one per line).
    dem_path : Path
        Path to DEM GeoTIFF (can be UTM or WGS84).
    los_path : Path
        Path to 3-band LOS GeoTIFF with E,N,U components.
    datetimes : list[datetime]
        List of datetime objects to get corrections for.
    output_dir : Path
        Directory to save correction GeoTIFFs.
    height_margin : float
        Additional height margin in meters above DEM max.

    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load DEM and LOS
    logger.info(f"Loading DEM from {dem_path}")
    dem = _open_2d(dem_path)

    logger.info(f"Loading LOS from {los_path}")
    los_enu = rxr.open_rasterio(los_path)

    # Get DEM bounds for cropping TROPO data
    if dem.rio.crs != "epsg:4326":
        dem_bounds_4326 = dem.rio.transform_bounds("epsg:4326")
    else:
        dem_bounds_4326 = dem.rio.bounds()

    # Add margin to bounds
    margin_deg = 0.1  # degrees
    west, south, east, north = dem_bounds_4326
    lat_bounds = (north + margin_deg, south - margin_deg)
    lon_bounds = (west - margin_deg, east + margin_deg)

    h_max = float(dem.max())

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
        ds0 = _open_crop(early_url, lat_bounds, lon_bounds, h_max, height_margin)
        ds1 = _open_crop(late_url, lat_bounds, lon_bounds, h_max, height_margin)

        # Interpolate in time
        td_interp = _interp_in_time(
            ds0,
            ds1,
            ds0.time.to_pandas().item(),
            ds1.time.to_pandas().item(),
            dt_pandas,
        )

        # Interpolate to DEM surface
        logger.info("Interpolating to DEM surface")
        zenith_delay_2d = _height_to_dem_surface(td_interp.total_delay, dem)

        # Apply LOS correction
        logger.info("Applying LOS correction")
        los_correction = _apply_los_correction(zenith_delay_2d, los_enu)

        # Save the correction
        time_str = dt_pandas.strftime("%Y%m%dT%H%M%S")
        output_file = output_dir / f"tropo_correction_{time_str}.tif"

        los_correction.rio.to_raster(output_file, compress="lzw")
        logger.info(f"Saved: {output_file}")


def main() -> None:
    """CLI entry point for tropo-apply command."""
    tyro.cli(apply_tropo)


if __name__ == "__main__":
    main()
