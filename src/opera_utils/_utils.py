from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import numpy as np
from numpy.typing import DTypeLike

from ._types import PathOrStr

__all__ = [
    "format_nc_filename",
    "scratch_directory",
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
        raise ValueError("Must provide dataset name for HDF5/NetCDF files")

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
        raise TypeError(f"dtype {np_dtype} not supported by GDAL.")
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
