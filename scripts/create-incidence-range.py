#!/usr/bin/env python
# /// script
# dependencies = ['rasterio', 'tyro']
# ///
"""Create incidence and approximate slant range distance from line-of-sight rasters.

DISP-S1 static layers include the line-of-sight (LOS) east, north, up (ENU) unit
vectors.  From this, we can get the incidence `arccos(up)`, and an approximation of
the slant range distance based on Sentinel-1 orbit altitude.
"""

from pathlib import Path

import numpy as np
import rasterio
import tyro

DEFAULT_RASTERIO_PROFILE = {
    "dtype": rasterio.float32,
    "count": 1,
    "nodata": 0,
    "tiled": True,
    "compress": "deflate",
    "predictor": 2,
    "nbits": 16,
}


def compute_incidence_angle(
    los_enu_path: Path | str,
    out_path: Path | str = Path("incidence_angle.tif"),
    nodata: float = 0,
) -> Path:
    """Compute incidence angles from a LOS ENU raster and save the as a new GeoTIFF.

    Users band 3 ("up") of the input `los_enu` raster, which equals cos(incidence)).
    The incidence angle, in degrees, is `incidence_deg = degrees(arccos(cos_values))`

    Parameters
    ----------
    los_enu_path : Path or str
        Path to the input 'los_enu.tif'.
    out_path : Path or str, optional
        Path to the output incidence angle GeoTIFF. Default is "incidence_angle.tif".
    nodata : float
        Value to use a nodata value in output rasters.
        Default is 0.

    Returns
    -------
    Path
        The path to the output incidence angle GeoTIFF.

    """
    with rasterio.open(los_enu_path) as src:
        # Read the third band (los_up == cos(incidence) )
        cos_values = src.read(3, masked=True)

        # Compute incidence angle in degrees
        incidence_deg = np.degrees(np.arccos(cos_values))

        profile = src.profile.copy()
        profile.update(**DEFAULT_RASTERIO_PROFILE)
        profile["nodata"] = nodata

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(incidence_deg.astype(np.float32).filled(nodata), 1)

    return Path(out_path)


def get_slant_range(incidence_raster: Path | str, subsample: int = 1) -> np.ndarray:
    """Compute approximate slant-range distance from incidence angles.

    The calculation uses the law of sines approach to derive the slant range from:
      - Earth radius (6_371 km)
      - Satellite altitude (~693 km)
      - Incidence angle (in degrees)

    Parameters
    ----------
    incidence_raster : Path or str
        Path to the incidence angle GeoTIFF (in degrees).
    subsample : int, optional
        Factor to subsample the raster (e.g., use every N-th pixel). Default is 100.

    Returns
    -------
    np.ndarray
        A 2D array of slant-range distances, matching the shape of the subsampled raster

    """
    earth_radius = 6_371_000.0  # meters
    sat_altitude = 693_000.0  # meters above Earth's surface
    R = earth_radius + sat_altitude

    with rasterio.open(incidence_raster) as src:
        incidence_deg = src.read(1, masked=True)[::subsample, ::subsample]

    incidence_rad = np.radians(incidence_deg)

    two_times_circ = R / np.sin(incidence_rad)
    look_angle_rad = np.arcsin(earth_radius / two_times_circ)
    range_angle_rad = incidence_rad - look_angle_rad
    slant_range = two_times_circ * np.sin(range_angle_rad)

    return slant_range.filled(0)


def create_inc_range(
    los_enu: Path | str,
    inc_angle_path: Path | str = Path("incidence_angle.tif"),
    slant_range_path: Path | str = Path("slant_range_distance.tif"),
) -> None:
    """Create an incidence-angle and slant range rasters from a LOS ENU raster.

    Parameters
    ----------
    los_enu : Path or str
        Path to the input LOS ENU GeoTIFF.
    inc_angle_path : Path or str
        Path to write the incidence angle GeoTIFF.
        Default is "incidence_angle.tif".
    slant_range_path : Path or str
        Path to write the slant range GeoTIFF.
        Default is "slant_range_distance.tif".

    """
    compute_incidence_angle(los_enu_path=los_enu, out_path=inc_angle_path)
    slant_range = get_slant_range(incidence_raster=inc_angle_path)

    with rasterio.open(inc_angle_path) as src:
        profile = src.profile.copy()
    with rasterio.open(slant_range_path, "w", **profile) as dst:
        dst.write(slant_range.astype(np.float32), 1)
    return


if __name__ == "__main__":
    tyro.cli(create_inc_range)
