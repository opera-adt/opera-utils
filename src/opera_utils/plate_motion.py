# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pyproj",
#     "rioxarray",
#     "tyro",
# ]
# ///
"""Stand-alone plate-motion utilities and CLI.

This module computes rigid plate-motion velocities at geodetic coordinates using
an Euler pole, projecting to the line-of-sight (LOS) given an east/north/up raster.

Conventions
-----------
- Angular velocity:
    * Cartesian: (wx, wy, wz) in milliarcseconds/year (mas/yr)
    * Spherical: (lat [deg], lon [deg], omega [deg/Ma])
- Coordinates:
    * Geodetic latitude/longitude are in degrees (WGS84 ellipsoid).
    * Altitude is meters above ellipsoid (default 0 m).
- ENU frame:
    * East, North, Up (local topocentric) at each (lat, lon).
- LOS projection:
    * The LOS unit vector points **from ground toward the sensor**.

Examples
--------
    uv run --script plate_motion.py \
        --los-enu los_enu.tif
        --plate-name NorthAmerica \
        --out plate_motion_in_los.tif


References
----------
Mintpy Euler Pole and plate_motion scripts.

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pyproj
import rioxarray as rxr
import tyro

MAS_TO_RAD = np.pi / (180.0 * 3_600_000.0)  # 1 mas = this many radians
DEGMA_TO_MASPYR = 1_000_000.0 / 3_600_000.0  # deg/Ma -> mas/yr
WGS84 = pyproj.CRS.from_epsg(4979)  # geodetic 3D
ECEF = pyproj.CRS.from_epsg(4978)  # ECEF


@dataclass(frozen=True)
class PlateITRF2014:
    """ITRF2014 plate angular velocity in mas/yr."""

    abbrev: str
    omega_x: float
    omega_y: float
    omega_z: float


# Table from Altamimi et al. (2017)
ITRF2014_PMM: dict[str, PlateITRF2014] = {
    "antartica": PlateITRF2014("ANTA", -0.248, -0.324, 0.675),
    "arabia": PlateITRF2014("ARAB", 1.154, -0.136, 1.444),
    "australia": PlateITRF2014("AUST", 1.510, 1.182, 1.215),
    "eurasia": PlateITRF2014("EURA", -0.085, -0.531, 0.770),
    "india": PlateITRF2014("INDI", 1.154, -0.005, 1.454),
    "nazca": PlateITRF2014("NAZC", -0.333, -1.544, 1.623),
    "northamerica": PlateITRF2014("NOAM", 0.024, -0.694, -0.063),
    "nubia": PlateITRF2014("NUBI", 0.099, -0.614, 0.733),
    "pacific": PlateITRF2014("PCFC", -0.409, 1.047, -2.169),
    "southamerica": PlateITRF2014("SOAM", -0.270, -0.301, -0.140),
    "somalia": PlateITRF2014("SOMA", -0.121, -0.794, 0.884),
}
PLATE_NAMES = list(ITRF2014_PMM.keys())


PlateName = StrEnum("PlateName", PLATE_NAMES)  # type: ignore[misc]


def cart2sph(
    rx: np.ndarray, ry: np.ndarray, rz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cartesian → spherical.

    Parameters
    ----------
    rx, ry, rz : array_like
        Components in an arbitrary radial unit.

    Returns
    -------
    lat_deg, lon_deg, r : ndarray
        Latitude [deg], longitude [deg], radius [same unit as inputs].

    """
    r = np.sqrt(rx * rx + ry * ry + rz * rz)
    lat = np.rad2deg(np.arcsin(rz / r))
    lon = np.rad2deg(np.arctan2(ry, rx))
    return lat, lon, r


def sph2cart(
    lat_deg: np.ndarray, lon_deg: np.ndarray, r: np.ndarray | float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spherical → Cartesian.

    Parameters
    ----------
    lat_deg, lon_deg : array_like
        Latitude/longitude in degrees.
    r : array_like or float, default=1
        Radius in any unit.

    Returns
    -------
    rx, ry, rz : ndarray
        Cartesian components with same unit as r.

    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    clat = np.cos(lat)
    return (
        r * clat * np.cos(lon),
        r * clat * np.sin(lon),
        r * np.sin(lat),
    )


def geodetic_to_ecef(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m: np.ndarray | float = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """WGS84 geodetic (lat, lon, h) → ECEF (x, y, z), in meters."""
    transformer = pyproj.Transformer.from_crs(WGS84, ECEF, always_xy=True)
    x, y, z = transformer.transform(
        np.radians(lon_deg), np.radians(lat_deg), np.broadcast_to(alt_m, lon_deg.shape)
    )
    return np.asarray(x), np.asarray(y), np.asarray(z)


def ecef_to_enu_components(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate ECEF vectors into ENU at each (lat, lon).

    Parameters
    ----------
    lat_deg, lon_deg : array_like
        Geodetic latitude/longitude [deg].
    x, y, z : array_like
        ECEF vector components (e.g., velocity) with consistent units.

    Returns
    -------
    e, n, u : ndarray
        ENU components in same units as inputs.

    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)

    e = -slon * x + clon * y
    n = -slat * clon * x - slat * slon * y + clat * z
    u = clat * clon * x + clat * slon * y + slat * z
    return e, n, u


@dataclass
class EulerPole:
    """Euler pole / angular velocity.

    Parameters
    ----------
    wx_mas_yr, wy_mas_yr, wz_mas_yr
        Cartesian angular velocity components in mas/yr.

    Notes
    -----
    Internally we store Cartesian mas/yr.

    """

    wx_mas_yr: float
    wy_mas_yr: float
    wz_mas_yr: float

    def velocity_xyz_m_per_yr(
        self,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        alt_m: float | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ECEF velocity induced by Euler rotation at positions.

        Returns
        -------
        vx, vy, vz : ndarray
            ECEF velocity components [m/yr].

        """
        omega_rad_per_yr = (
            np.asarray([self.wx_mas_yr, self.wy_mas_yr, self.wz_mas_yr], dtype=float)
            * MAS_TO_RAD
        )

        x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        xyz = np.stack([x, y, z], axis=0)  # (3, N)
        # Cross product omega x r for each point
        _, rows, cols = xyz.shape
        vx, vy, vz = np.cross(
            omega_rad_per_yr[:, None], xyz.reshape(3, -1), axis=0
        ).reshape(3, rows, cols)
        return vx, vy, vz

    def velocity_enu_m_per_yr(
        self, lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: float | np.ndarray = 0.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ENU velocity induced by Euler rotation at positions.

        Returns
        -------
        ve, vn, vu : ndarray
            East/North/Up velocities [m/yr].

        """
        vx, vy, vz = self.velocity_xyz_m_per_yr(lat_deg, lon_deg, alt_m)
        return ecef_to_enu_components(lat_deg, lon_deg, vx, vy, vz)


def _build_euler_from_inputs(plate_name: PlateName) -> EulerPole:
    """Construct EulerPole from CLI options."""
    plate = ITRF2014_PMM.get(plate_name.value)
    if plate is None:
        msg = f"Unknown ITRF2014 plate: {plate_name}. Options: {list(ITRF2014_PMM)}"
        raise ValueError(msg)
    return EulerPole(
        wx_mas_yr=plate.omega_x, wy_mas_yr=plate.omega_y, wz_mas_yr=plate.omega_z
    )


def run(
    los_enu_path: Path | str,
    plate_name: PlateName,
    out: str | None = "rotation_los_enu.tif",
    subsample: int = 10,
) -> None:
    """Compute plate-motion computation in the radar line-of-sight.

    Parameters
    ----------
    los_enu_path : Path | str
        Path / url to line-of-sight 3-band east, north, up raster
    plate_name : str
        Name of plate in ITRF2014 table
    out : str, optional
        Output LOS velocity GeoTIFF
    subsample : int
        Decimation factor to apply in x and y before computation.
        Default is 10 (Output is 100x smaller than `los_enu_path`)

    """
    da_los_enu = rxr.open_rasterio(los_enu_path, default_name="los_enu")
    # los_enu, proc_enu = _read_tif(los_enu_path)
    da_los_enu = da_los_enu[:, ::subsample, ::subsample].astype("float32")
    da_los_enu_latlon = da_los_enu.rio.reproject("epsg:4326")

    # Create EulerPole if given
    euler = _build_euler_from_inputs(plate_name)

    # Coordinates for DEM grid
    lon, lat = np.meshgrid(
        da_los_enu_latlon.y.values, da_los_enu_latlon.x.values, indexing="ij"
    )
    lon = lon.astype("float32")
    lat = lat.astype("float32")

    # Compute ENU rotation component (m/yr)
    ve_rot, vn_rot, vu_rot = euler.velocity_enu_m_per_yr(lat, lon, alt_m=0.0)
    v_enu = np.stack([ve_rot, vn_rot, vu_rot])

    da_v_los = np.sum(v_enu * da_los_enu_latlon, axis=0)
    da_v_los.attrs["units"] = "meters / year"
    da_v_los.rio.write_nodata(0).rio.to_raster(
        out, dtype="float32", tiled="yes", compress="deflate"
    )


if __name__ == "__main__":
    tyro.cli(run)
