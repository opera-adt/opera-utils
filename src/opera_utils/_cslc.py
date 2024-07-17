from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from pyproj import CRS, Transformer

from ._types import Filename

__all__ = [
    "CslcParseError",
    "parse_filename",
    "get_zero_doppler_time",
    "get_radar_wavelength",
    "get_orbit_arrays",
    "get_xy_coords",
    "get_lonlat_grid",
    "get_cslc_orbit",
]


CSLC_S1_FILE_REGEX = (
    r"(?P<project>OPERA)_"
    r"(?P<level>L2)_"
    r"(?P<product_type>CSLC-S1)_"
    r"(?P<burst_id>T\d{3}-\d+-IW\d)_"
    r"(?P<start_datetime>\d{8}T\d{6}Z)_"
    r"(?P<end_datetime>\d{8}T\d{6}Z)_"
    r"(?P<sensor>S1[AB])_"
    r"(?P<polarization>VV|HH)_"
    r"v(?P<product_version>\d+\.\d+)"
)


class CslcParseError(ValueError):
    """Error raised for non-matching filename."""

    pass


def parse_filename(h5_filename: Filename) -> dict[str, str | datetime]:
    """Get the complex valued dataset from the CSLC HDF5 file.

    ...
    """
    name = Path(h5_filename).name
    match = re.match(CSLC_S1_FILE_REGEX, name)
    if match is None:
        raise CslcParseError(f"Unable to parse {h5_filename}")

    result = match.groupdict()
    # Normalize to lowercase / underscore
    result["burst_id"] = result["burst_id"].lower().replace("-", "_")
    result["start_datetime"] = datetime.fromisoformat(result["start_datetime"])
    result["end_datetime"] = datetime.fromisoformat(result["end_datetime"])
    return result


def get_zero_doppler_time(filename: Filename, type_: str = "start") -> datetime:
    """Get the full acquisition time from the CSLC product.

    Uses `/identification/zero_doppler_{type_}_time` from the CSLC product.

    Parameters
    ----------
    filename : Filename
        Path to the CSLC product.
    type_ : str, optional
        Either "start" or "stop", by default "start".

    Returns
    -------
    str
        Full acquisition time.
    """
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

    def get_dt(in_str):
        return datetime.strptime(in_str.decode("utf-8"), DATETIME_FORMAT)

    dset = f"/identification/zero_doppler_{type_}_time"
    value = _get_dset_and_attrs(filename, dset, parse_func=get_dt)[0]
    return value


def _get_dset_and_attrs(
    filename: Filename,
    dset_name: str,
    parse_func: Callable = lambda x: x,
) -> tuple[Any, dict[str, Any]]:
    """Get one dataset's value and attributes from the CSLC product.

    Parameters
    ----------
    filename : Filename
        Path to the CSLC product.
    dset_name : str
        Name of the dataset.
    parse_func : Callable, optional
        Function to parse the dataset value, by default lambda x: x
        For example, could be parse_func=lambda x: x.decode("utf-8") to decode,
        or getting a datetime object from a string.

    Returns
    -------
    dset : Any
        The value of the scalar
    attrs : dict
        Attributes.
    """
    with h5py.File(filename, "r") as hf:
        dset = hf[dset_name]
        attrs = dict(dset.attrs)
        value = parse_func(dset[()])
        return value, attrs


def get_radar_wavelength(filename: Filename):
    """Get the radar wavelength from the CSLC product.

    Parameters
    ----------
    filename : Filename
        Path to the CSLC product.

    Returns
    -------
    wavelength : float
        Radar wavelength in meters.
    """
    dset = "/metadata/processing_information/input_burst_metadata/wavelength"
    value = _get_dset_and_attrs(filename, dset)[0]
    return value


def get_orbit_arrays(
    h5file: Filename,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, datetime]:
    """Parse orbit info from OPERA S1 CSLC HDF5 file into python types.

    Parameters
    ----------
    h5file : Filename
        Path to OPERA S1 CSLC HDF5 file.

    Returns
    -------
    times : np.ndarray
        Array of times in seconds since reference epoch.
    positions : np.ndarray
        Array of positions in meters.
    velocities : np.ndarray
        Array of velocities in meters per second.
    reference_epoch : datetime
        Reference epoch of orbit.

    """
    with h5py.File(h5file) as hf:
        orbit_group = hf["/metadata/orbit"]
        times = orbit_group["time"][:]
        positions = np.stack([orbit_group[f"position_{p}"] for p in ["x", "y", "z"]]).T
        velocities = np.stack([orbit_group[f"velocity_{p}"] for p in ["x", "y", "z"]]).T
        reference_epoch = datetime.fromisoformat(
            orbit_group["reference_epoch"][()].decode()
        )

    return times, positions, velocities, reference_epoch


def get_cslc_orbit(h5file: Filename):
    """Parse orbit info from OPERA S1 CSLC HDF5 file into an isce3.core.Orbit.

    `isce3` must be installed to use this function.

    Parameters
    ----------
    h5file : Filename
        Path to OPERA S1 CSLC HDF5 file.

    Returns
    -------
    orbit : isce3.core.Orbit
        Orbit object.

    """
    from isce3.core import DateTime, Orbit, StateVector

    times, positions, velocities, reference_epoch = get_orbit_arrays(h5file)
    orbit_svs = []

    for t, x, v in zip(times, positions, velocities):
        orbit_svs.append(
            StateVector(
                DateTime(reference_epoch + timedelta(seconds=t)),
                x,
                v,
            )
        )

    return Orbit(orbit_svs)


def get_xy_coords(h5file: Filename, subsample: int = 100) -> tuple:
    """Get x and y grid from OPERA S1 CSLC HDF5 file.

    Parameters
    ----------
    h5file : Filename
        Path to OPERA S1 CSLC HDF5 file.
    subsample : int, optional
        Subsampling factor, by default 100

    Returns
    -------
    x : np.ndarray
        Array of x coordinates in meters.
    y : np.ndarray
        Array of y coordinates in meters.
    projection : int
        EPSG code of projection.

    """
    with h5py.File(h5file) as hf:
        x = hf["/data/x_coordinates"][:]
        y = hf["/data/y_coordinates"][:]
        projection = hf["/data/projection"][()]

    return x[::subsample], y[::subsample], projection


def get_lonlat_grid(
    h5file: Filename, subsample: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Get 2D latitude and longitude grid from OPERA S1 CSLC HDF5 file.

    Parameters
    ----------
    h5file : Filename
        Path to OPERA S1 CSLC HDF5 file.
    subsample : int, optional
        Subsampling factor, by default 100

    Returns
    -------
    lat : np.ndarray
        2D Array of latitude coordinates in degrees.
    lon : np.ndarray
        2D Array of longitude coordinates in degrees.
    projection : int
        EPSG code of projection.

    """
    x, y, projection = get_xy_coords(h5file, subsample)
    X, Y = np.meshgrid(x, y)
    xx = X.flatten()
    yy = Y.flatten()
    crs = CRS.from_epsg(projection)
    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(xx=xx, yy=yy, radians=False)
    lon = lon.reshape(X.shape)
    lat = lat.reshape(Y.shape)
    return lon, lat
