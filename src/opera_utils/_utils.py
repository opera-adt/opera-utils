from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike

from ._helpers import reproject_bounds, reproject_coordinates
from ._types import Bbox, PathOrStr

__all__ = [
    "create_yx_arrays",
    "format_nc_filename",
    "get_snwe",
    "scratch_directory",
    "transform_xy_to_latlon",
]


def format_nc_filename(filename: PathOrStr, ds_name: str | None = None) -> str:
    """Format an HDF5/NetCDF filename with dataset for reading using GDAL.

    If `filename` is already formatted, or if `filename` is not an HDF5/NetCDF
    file (based on the file extension), it is returned unchanged.

    Parameters
    ----------
    filename : str or PathLike
        Filename to format.
    ds_name : str, optional
        Dataset name to use. If not provided for a .h5 or .nc file, an error is raised.

    Returns
    -------
    str
        Formatted filename like
        NETCDF:"filename.nc":"//ds_name"

    Raises
    ------
    ValueError
        If `ds_name` is not provided for a .h5 or .nc file.

    """
    # If we've already formatted the filename, return it
    if str(filename).startswith("NETCDF:") or str(filename).startswith("HDF5:"):
        return str(filename)

    if not (os.fspath(filename).endswith(".nc") or os.fspath(filename).endswith(".h5")):
        return os.fspath(filename)

    # Now we're definitely dealing with an HDF5/NetCDF file
    if ds_name is None:
        msg = "Must provide dataset name for HDF5/NetCDF files"
        raise ValueError(msg)

    return f'NETCDF:"{filename}":"//{ds_name.lstrip("/")}"'


def _get_path_from_gdal_str(name: PathOrStr) -> Path:
    s = str(name)
    if s.upper().startswith("DERIVED_SUBDATASET"):
        # like DERIVED_SUBDATASET:AMPLITUDE:slc_filepath.tif
        p = s.split(":")[-1].strip('"').strip("'")
    elif ":" in s and (s.upper().startswith("NETCDF") or s.upper().startswith("HDF")):
        # like NETCDF:"slc_filepath.nc":subdataset
        p = s.split(":")[1].strip('"').strip("'")
    else:
        # Whole thing is the path
        p = str(name)
    return Path(p)


def numpy_to_gdal_type(np_dtype: DTypeLike) -> int:
    """Convert numpy dtype to gdal type.

    Parameters
    ----------
    np_dtype : DTypeLike
        Numpy dtype to convert.

    Returns
    -------
    int
        GDAL type code corresponding to `np_dtype`.

    Raises
    ------
    TypeError
        If `np_dtype` is not a numpy dtype, or if the provided dtype is not
        supported by GDAL (for example, `np.dtype('>i4')`)

    """
    from osgeo import gdal_array, gdalconst

    np_dtype = np.dtype(np_dtype)

    if np.issubdtype(bool, np_dtype):
        return gdalconst.GDT_Byte
    gdal_code = gdal_array.NumericTypeCodeToGDALTypeCode(np_dtype)
    if gdal_code is None:
        msg = f"dtype {np_dtype} not supported by GDAL."
        raise TypeError(msg)
    return gdal_code


def gdal_to_numpy_type(gdal_type: str | int) -> np.dtype:
    """Convert gdal type to numpy type."""
    from osgeo import gdal, gdal_array

    if isinstance(gdal_type, str):
        gdal_type = gdal.GetDataTypeByName(gdal_type)
    return np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type))


@contextmanager
def scratch_directory(
    dir_: PathOrStr | None = None, *, delete: bool = True
) -> Generator[Path, None, None]:
    """Context manager that creates a (possibly temporary) file system directory.

    If `dir_` is a path-like object, a directory will be created at the specified
    file system path if it did not already exist. Otherwise, if `dir_` is None, a
    temporary directory will instead be created as though by
    ``tempfile.TemporaryDirectory()``.

    The directory may be automatically removed from the file system upon exiting the
    context manager.

    Parameters
    ----------
    dir_ : PathOrStr or None, optional
        Scratch directory path. If None, a temporary directory will be created. Defaults
        to None.
    delete : bool, optional
        If True and `dir_` didn't exist, the directory and its contents are
        recursively removed from the file system upon exiting the context manager.
        Note that if `dir_` already exists, this option is ignored.
        Defaults to True.

    Yields
    ------
    pathlib.Path
        Scratch directory path. If `delete` was True, the directory will be removed from
        the file system upon exiting the context manager scope.

    """
    if dir_ is None:
        scratchdir = Path(tempfile.mkdtemp())
        dir_already_existed = False
    else:
        scratchdir = Path(dir_)
        dir_already_existed = scratchdir.exists()
        scratchdir.mkdir(parents=True, exist_ok=True)

    yield scratchdir

    if delete and not dir_already_existed:
        shutil.rmtree(scratchdir)


def get_snwe(epsg: int, bounds: Bbox) -> tuple[float, float, float, float]:
    """Convert bounds from (west, south, east, north) (WSEN) to SNWE in lat/lon.

    This box format is used by the `RAiDER` library for its area of interest tracking:
    https://github.com/dbekaert/RAiDER/blob/38eab3969ac762bc59502d7eb482fd73d6a0deef/tools/RAiDER/llreader.py#L30

    Parameters
    ----------
    epsg : int
        EPSG code of the input coordinates in `bounds`.
    bounds : tuple[float, float, float, float]
        Bounds in WSEN format.

    Returns
    -------
    tuple[float, float, float, float]
        Bounds in SNWE (lat/lon) format.

    """
    if epsg != 4326:
        bounds = reproject_bounds(bounds, epsg, 4326)

    snwe = (bounds[1], bounds[3], bounds[0], bounds[2])

    return snwe


def create_yx_arrays(
    gt: list[float], shape: tuple[int, int], step_size: float = 500
) -> tuple[np.ndarray, np.ndarray]:
    """Create the x and y coordinate datasets.

    Assumes that the `y` output coordinates will be north-up, so that the
    `y` array is in decreasing order.

    Parameters
    ----------
    gt : List[float]
        Geotransform list.
    shape : tuple[int, int]
        Shape of the dataset (ysize, xsize).
    step_size : float
        Pixel spacing, in units matching the projection of `gt`
        (e.g. meters for a UTM geotransform)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        x and y coordinate arrays.

    """
    ysize, xsize = shape
    # Parse the geotransform
    x_origin, x_res, _, y_origin, _, y_res = gt
    y_end = y_origin + y_res * ysize
    x_end = x_origin + x_res * xsize

    # Make the x/y arrays
    y = np.arange(y_origin, y_end - step_size, -1 * step_size)
    x = np.arange(x_origin, x_end + step_size, step_size)

    return y, x


def transform_xy_to_latlon(
    epsg: int, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the x, y coordinates in the source projection to WGS84 lat/lon.

    Parameters
    ----------
    epsg : int
        EPSG code.
    x : np.ndarray
        x coordinates.
    y : np.ndarray
        y coordinates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Latitude and longitude arrays.

    """
    if epsg != 4326:
        longitude, latitude = reproject_coordinates(
            x.flatten(), y.flatten(), epsg, 4326
        )

        # # reshape Lat lon of data cube
        latitude = np.array(latitude).reshape(x.shape)
        longitude = np.array(longitude).reshape(x.shape)
    else:
        latitude = y.copy()
        longitude = x.copy()

    return latitude, longitude
