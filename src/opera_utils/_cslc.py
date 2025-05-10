from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from os import fspath
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from pyproj import CRS, Transformer
from shapely import geometry, ops, wkt

from ._types import Filename
from .bursts import normalize_burst_id
from .constants import (
    COMPASS_FILE_REGEX,
    CSLC_S1_FILE_REGEX,
    NISAR_BOUNDING_POLYGON,
    NISAR_FILE_REGEX,
    OPERA_IDENTIFICATION,
)

__all__ = [
    "CslcParseError",
    "create_nodata_mask",
    "get_cslc_orbit",
    "get_cslc_polygon",
    "get_lonlat_grid",
    "get_orbit_arrays",
    "get_radar_wavelength",
    "get_union_polygon",
    "get_xy_coords",
    "get_zero_doppler_time",
    "make_nodata_mask",  # TODO: deprecate
    "parse_filename",
]
logger = logging.getLogger(__name__)


class CslcParseError(ValueError):
    """Error raised for non-matching filename."""


def parse_filename(h5_filename: Filename) -> dict[str, str | datetime]:
    """Parse the filename of a CSLC HDF5 file.

    Parameters
    ----------
    h5_filename : Filename
        The path or name of the CSLC HDF5 file.

    Returns
    -------
    dict[str, str | datetime]
        A dictionary containing parsed components of the filename:
        - project: str
        - level: str
        - product_type: str
        - burst_id: str (normalized to lowercase with underscores)
        - start_datetime: datetime
        - generation_datetime: datetime
        - sensor: str
        - polarization: str
        - product_version: str

    Or, if the filename is a COMPASS-generated file,
        - burst_id: str (lowercase with underscores)
        - start_datetime: datetime (but no hour/minute/second info)

    Raises
    ------
    CslcParseError
        If the filename does not match the expected pattern.

    """
    name = Path(h5_filename).name
    match: re.Match | None = None

    if match := re.match(CSLC_S1_FILE_REGEX, name):
        return _parse_cslc_product(match)
    elif match := re.match(NISAR_FILE_REGEX, name):
        return _parse_gslc_product(match)
    elif match := re.match(COMPASS_FILE_REGEX, name):
        return _parse_compass(match)
    else:
        msg = f"Unable to parse {h5_filename}"
        raise CslcParseError(msg)


def _parse_compass(match: re.Match):
    result = match.groupdict()
    result["start_datetime"] = datetime.strptime(
        result["start_datetime"], "%Y%m%d"
    ).replace(tzinfo=timezone.utc)
    return result


def _parse_gslc_product(match: re.Match):
    result = match.groupdict()
    result["frame_id"] = result["frame_id"]
    fmt = "%Y%m%dT%H%M%SZ"
    result["start_datetime"] = datetime.strptime(result["start_datetime"], fmt).replace(
        tzinfo=timezone.utc
    )
    result["generation_datetime"] = datetime.strptime(
        result["generation_datetime"], fmt
    ).replace(tzinfo=timezone.utc)
    return result


def _parse_cslc_product(match: re.Match):
    result = match.groupdict()
    # Normalize to lowercase / underscore
    result["burst_id"] = normalize_burst_id(result["burst_id"])
    fmt = "%Y%m%dT%H%M%SZ"
    result["start_datetime"] = datetime.strptime(result["start_datetime"], fmt).replace(
        tzinfo=timezone.utc
    )
    result["generation_datetime"] = datetime.strptime(
        result["generation_datetime"], fmt
    ).replace(tzinfo=timezone.utc)
    return result


def get_dataset_name(h5_filename: Filename) -> str:
    """Get the complex valued dataset from the CSLC HDF5 file.

    Parameters
    ----------
    h5_filename : Filename
        The path or name of the CSLC HDF5 file.

    Returns
    -------
    str
        The name of the complex dataset in the format "/data/{polarization}".

    Raises
    ------
    CslcParseError
        If the filename cannot be parsed.

    """
    name = Path(h5_filename).name
    parsed = parse_filename(name)
    if "polarization" in parsed:
        return f"/data/{parsed['polarization']}"
    else:
        # For compass, no polarization is given, so we have to check the file
        with h5py.File(h5_filename) as hf:
            if "VV" in hf["/data"]:
                return "/data/VV"
            else:
                return "/data/HH"


def get_zero_doppler_time(
    filename: Filename,
    dataset: str | None = None,
    type_: str = "start",
    datetime_format: str = "%Y-%m-%d %H:%M:%S.%f",
) -> datetime:
    """Get the full acquisition time from the CSLC product.

    Uses `/identification/zero_doppler_{type_}_time` from the CSLC product.

    Parameters
    ----------
    filename : Filename
        Path to the CSLC product.
    dataset : str, optional
        The subdataset to read zero doppler time from
    type_ : str, optional
        Either "start" or "stop", by default "start".
    datetime_format : str, optional
        The format of the datetime

    Returns
    -------
    str
        Full acquisition time.

    """

    def get_dt(in_str):
        # Sentinel-1: datetime_format = "%Y-%m-%d %H:%M:%S.%f"
        # NISAR: datetime_format = "%Y-%m-%dT%H:%M:%S.%f" + some extra digits
        # The index 0:26 gets rid of those extra digits
        return datetime.strptime(in_str.decode("utf-8")[:26], datetime_format)

    # This has to be the default but since we have to change in disp-s1
    # product.py and maybe other use cases, I am leaving it as is
    # because it depends on the type_
    dset = f"/identification/zero_doppler_{type_}_time"
    if dataset:
        dset = dataset
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


def get_radar_wavelength(filename: Filename) -> float:
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
    """Retrieve orbit arrays and timestamps from an HDF5 file.

    This function parses the filename to determine the sensor type
    (e.g., "S1" or "NISAR") and calls either `get_nisar_orbit` or
    `get_s1_orbit` accordingly.

    Parameters
    ----------
    h5file : Filename
        Path to the HDF5 file containing orbit data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, datetime]
        - Position array (np.ndarray)
        - Velocity array (np.ndarray)
        - Time array (np.ndarray)
        - Reference datetime (datetime)

    """
    # Parse the filename to figure out if this is S1 vs NISAR
    parsed = parse_filename(h5file)
    project = str(parsed.get("project", ""))
    if project.lower().startswith("nisar"):
        return get_nisar_orbit(h5file)
    else:
        return get_s1_orbit(h5file)


def get_s1_orbit(
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


def get_nisar_orbit(
    h5file: Filename,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, datetime]:
    """Parse orbit info from NISAR HDF5 file into python types.

    Parameters
    ----------
    h5file : Filename
        Path to NISAR GSLC HDF5 file.

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
        orbit_group = hf["/science/LSAR/GSLC//metadata/orbit"]
        times = orbit_group["time"][:]
        positions = orbit_group["position"][()]
        velocities = orbit_group["velocity"][()]
        units = orbit_group["time"].attrs["units"].decode("utf-8")
        reference_epoch_str = units.split("since")[-1].strip()
        reference_epoch = datetime.fromisoformat(reference_epoch_str)
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


def get_xy_coords(
    h5file: Filename, subsample: int = 100
) -> tuple[np.ndarray, np.ndarray, int]:
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
    epsg_code : int
        EPSG code of projection.

    """
    with h5py.File(h5file) as hf:
        x = hf["/data/x_coordinates"][:]
        y = hf["/data/y_coordinates"][:]
        projection_dset = hf["/data/projection"]
        crs_string = ""
        # https://github.com/corteva/rioxarray/blob/5783693895b4b055909c5758a72a5d40a365ef11/rioxarray/rioxarray.py#L34
        for attr_name in "spatial_ref", "crs_wkt":
            if attr_name in projection_dset.attrs:
                crs_string = projection_dset.attrs[attr_name]
        if not crs_string:
            msg = f"Failed to parse CRS for {h5file}"
            raise ValueError(msg)
        if isinstance(crs_string, bytes):
            crs_string = crs_string.decode("utf-8")
        crs = CRS.from_user_input(crs_string)

    return x[::subsample], y[::subsample], crs.to_epsg()


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

    """
    x, y, epsg = get_xy_coords(h5file, subsample)
    X, Y = np.meshgrid(x, y)
    xx = X.flatten()
    yy = Y.flatten()
    crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(xx=xx, yy=yy, radians=False)
    lon = lon.reshape(X.shape)
    lat = lat.reshape(Y.shape)
    return lon, lat


def get_cslc_polygon(
    opera_file: Filename, buffer_degrees: float = 0.0
) -> geometry.Polygon | None:
    """Get the union of the bounding polygons of the given files.

    Parameters
    ----------
    opera_file : list[Filename]
        list of COMPASS SLC filenames.
    buffer_degrees : float, optional
        Buffer the polygons by this many degrees, by default 0.0

    """
    if "NISAR" in str(opera_file):
        dset_name = NISAR_BOUNDING_POLYGON
    else:
        dset_name = f"{OPERA_IDENTIFICATION}/bounding_polygon"
    with h5py.File(opera_file) as hf:
        if dset_name not in hf:
            logger.debug(f"Could not find {dset_name} in {opera_file}")
            return None
        wkt_str = hf[dset_name][()].decode("utf-8")
    return wkt.loads(wkt_str).buffer(buffer_degrees)


def get_union_polygon(
    opera_file_list: Sequence[Filename], buffer_degrees: float = 0.0
) -> geometry.Polygon:
    """Get the union of the bounding polygons of the given files.

    Parameters
    ----------
    opera_file_list : list[Filename]
        list of COMPASS SLC filenames.
    buffer_degrees : float, optional
        Buffer the polygons by this many degrees, by default 0.0

    """
    polygons = [get_cslc_polygon(f, buffer_degrees) for f in opera_file_list]
    polygons = [p for p in polygons if p is not None]

    if len(polygons) == 0:
        msg = "No polygons found in the given file list."
        raise ValueError(msg)
    # Union all the polygons
    return ops.unary_union(polygons)


def create_nodata_mask(
    opera_file_list: Sequence[Filename],
    out_file: Filename,
    dset_name: str | None = None,
    buffer_pixels: int = 400,
    overwrite: bool = False,
):
    """Create a boolean raster mask from the union of nodata polygons using GDAL.

    The output datatype is UInt8, where 1 means valid data in the polygon and
    0 is invalid (outside the polygon).

    Parameters
    ----------
    opera_file_list : list[Filename]
        list of COMPASS SLC filenames.
    out_file : Filename
        Output filename.
    dset_name : str, optional
        The name of the dataset in opera files
    buffer_pixels : int, optional
        Number of pixels to buffer the union polygon by, by default 0.
        Note that buffering will *decrease* the numba of pixels marked as nodata.
        This is to be more conservative to not mask possible valid pixels.
    overwrite : bool, optional
        Overwrite the output file if it already exists, by default False

    """
    from osgeo import gdal

    gdal.UseExceptions()
    if Path(out_file).exists():
        if not overwrite:
            logger.debug(f"Skipping {out_file} since it already exists.")
            return
        else:
            logger.info(f"Overwriting {out_file} since overwrite=True.")
            Path(out_file).unlink()

    # Check these are the right format to get nodata polygons
    if dset_name:
        dataset_name = dset_name
    else:
        try:
            dataset_name = get_dataset_name(opera_file_list[-1])
        except CslcParseError as e:
            msg = f"{opera_file_list[-1]} is not a CSLC file"
            raise ValueError(msg) from e

    try:
        test_f = f"NETCDF:{opera_file_list[-1]}:{dataset_name}"
        # convert pixels to degrees lat/lon
        gt = _get_raster_gt(test_f)
    except RuntimeError as e:
        msg = f"Unable to get geotransform from {test_f}"
        raise ValueError(msg) from e
    # TODO: more robust way to get the pixel size... this is a hack
    # maybe just use pyproj to warp lat/lon to meters and back?
    dx_meters = gt[1]
    dx_degrees = dx_meters / 111000
    buffer_degrees = buffer_pixels * dx_degrees

    # Get the union of all the polygons and convert to a temp geojson
    union_poly = get_union_polygon(opera_file_list, buffer_degrees=buffer_degrees)
    # convert shapely polygon to geojson

    # Make a dummy raster from the first file with all 0s
    # This will get filled in with the polygon rasterization
    cmd = (
        f"gdal_calc.py --quiet --outfile {out_file} --type Byte  -A"
        f" NETCDF:{opera_file_list[-1]}:{dataset_name} --calc 'numpy.nan_to_num(A)"
        " * 0' --creation-option COMPRESS=LZW --creation-option TILED=YES"
        " --creation-option BLOCKXSIZE=256 --creation-option BLOCKYSIZE=256"
    )
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_vector_file = Path(tmpdir) / "temp.geojson"
        with open(temp_vector_file, "w") as f:
            f.write(json.dumps(geometry.mapping(union_poly)))

        # Open the input vector file
        src_ds = gdal.OpenEx(fspath(temp_vector_file), gdal.OF_VECTOR)
        dst_ds = gdal.Open(fspath(out_file), gdal.GA_Update)

        # Now burn in the union of all polygons
        gdal.Rasterize(dst_ds, src_ds, burnValues=[1])


make_nodata_mask = create_nodata_mask


def _get_raster_gt(filename: Filename) -> list[float]:
    """Get the geotransform from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.

    Returns
    -------
    List[float]
        6 floats representing a GDAL Geotransform.

    """
    from osgeo import gdal

    ds = gdal.Open(fspath(filename))
    gt = ds.GetGeoTransform()
    return gt
