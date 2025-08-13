"""Apply tropospheric corrections to DEM using LOS geometry."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rioxarray as rxr
import tyro
import xarray as xr
from rasterio.enums import Resampling
from scipy.interpolate import RegularGridInterpolator

from opera_utils import get_dates

from ._helpers import (
    _open_2d,
)

logger = logging.getLogger(__name__)


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
    cropped_tropo_list: list[Path],
    dem_path: Path,
    los_path: Path,
    output_dir: Path = Path("tropo_corrections"),
    interp_method: str = "cubic",
    subtract_first_date: bool = True,
) -> None:
    """Apply tropospheric corrections using DEM and LOS geometry.

    Parameters
    ----------
    cropped_tropo_list : list[Path]
        List of cropped TROPO product paths resulting from `tropo-crop`.
    dem_path : Path
        Path to DEM GeoTIFF (can be UTM or WGS84).
    los_path : Path
        Path to 3-band LOS GeoTIFF with E,N,U components.
    output_dir : Path
        Directory to save correction GeoTIFFs.
    interp_method : str
        Interpolation method for RegularGridInterpolator.
        Options are  "linear", "nearest", "slinear", "cubic", "quintic" and "pchip"
        Default is "cubic".
    subtract_first_date : bool
        Whether to subtract the first date from the time series of corrections.
        This is useful for applying to a single-reference displacement time series.
        Default is True.

    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load DEM and LOS
    logger.info(f"Loading DEM from {dem_path}")
    dem = _open_2d(dem_path)

    logger.info(f"Loading LOS from {los_path}")
    los_up = rxr.open_rasterio(los_path).isel(band=2).values

    logger.info(f"Processing {len(cropped_tropo_list)} datetime(s)")

    day1_correction = 0

    for idx, cropped_file in enumerate(cropped_tropo_list):
        logger.info(f"Processing: {cropped_file}")

        # Open and crop the datasets
        ds = xr.open_dataset(cropped_file, engine="h5netcdf")

        # Interpolate to DEM surface
        logger.info("Interpolating to DEM surface")
        zenith_delay_2d = _height_to_dem_surface(
            ds.total_delay, dem, method=interp_method
        )

        # Apply LOS correction
        logger.info("Applying LOS correction")
        los_correction = zenith_delay_2d / los_up
        if idx == 0 and subtract_first_date:
            day1_correction = los_correction

        # subtract first date, or 0 if not using that option
        los_correction = los_correction - day1_correction

        # Save the correction
        fmt = "%Y%m%dT%H%M%S"
        time_str = get_dates(cropped_file, fmt=fmt)[0].strftime(fmt)
        output_file = output_dir / f"tropo_correction_{time_str}.tif"

        los_correction.rio.to_raster(output_file, compress="lzw")
        logger.info(f"Saved: {output_file}")


def main() -> None:
    """CLI entry point for tropo-apply command."""
    tyro.cli(apply_tropo)


if __name__ == "__main__":
    main()
