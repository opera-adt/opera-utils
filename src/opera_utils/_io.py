from __future__ import annotations

import logging
from os import fspath
from typing import Any

import numpy as np
import rasterio as rio
from affine import Affine
from numpy.typing import ArrayLike
from osgeo import gdal, gdal_array
from pyproj import CRS

from ._types import Bbox, Filename, PathOrStr

__all__ = [
    "load_gdal",
]


def load_gdal(
    filename: Filename,
    band: int | None = None,
    subsample_factor: int | tuple[int, int] = 1,
    overview: int | None = None,
    rows: slice | None = None,
    cols: slice | None = None,
    masked: bool = False,
) -> ArrayLike:
    """Load a gdal file into a numpy array.

    Parameters
    ----------
    filename : str or Path
        Path to the file to load.
    band : int, optional
        Band to load. If None, load all bands as 3D array.
    subsample_factor : int or tuple[int, int], optional
        Subsample the data by this factor. Default is 1 (no subsampling).
        Uses nearest neighbor resampling.
    overview: int, optional
        If passed, will load an overview of the file.
        Raster must have existing overviews, or ValueError is raised.
    rows : slice, optional
        Rows to load. Default is None (load all rows).
    cols : slice, optional
        Columns to load. Default is None (load all columns).
    masked : bool, optional
        If True, return a masked array using the raster's `nodata` value.
        Default is False.

    Returns
    -------
    arr : np.ndarray
        Array of shape (bands, y, x) or (y, x) if `band` is specified,
        where y = height // subsample_factor and x = width // subsample_factor.

    """
    ds = gdal.Open(fspath(filename))
    nrows, ncols = ds.RasterYSize, ds.RasterXSize

    if overview is not None:
        # We can handle the overviews most easily
        bnd = ds.GetRasterBand(band or 1)
        ovr_count = bnd.GetOverviewCount()
        if ovr_count > 0:
            idx = ovr_count + overview if overview < 0 else overview
            out = bnd.GetOverview(idx).ReadAsArray()
            bnd = ds = None
            return out
        else:
            logging.warning(f"Requested {overview = }, but none found for {filename}")

    # if rows or cols are not specified, load all rows/cols
    rows = slice(0, nrows) if rows in (None, slice(None)) else rows
    cols = slice(0, ncols) if cols in (None, slice(None)) else cols
    # Help out mypy:
    assert rows is not None
    assert cols is not None

    dt = gdal_to_numpy_type(ds.GetRasterBand(1).DataType)

    if isinstance(subsample_factor, int):
        subsample_factor = (subsample_factor, subsample_factor)

    xoff, yoff = int(cols.start), int(rows.start)
    row_stop = min(rows.stop, nrows)
    col_stop = min(cols.stop, ncols)
    xsize, ysize = int(col_stop - cols.start), int(row_stop - rows.start)
    if xsize <= 0 or ysize <= 0:
        msg = (
            f"Invalid row/col slices: {rows}, {cols} for file {filename} of size"
            f" {nrows}x{ncols}"
        )
        raise IndexError(msg)
    nrows_out, ncols_out = (
        ysize // subsample_factor[0],
        xsize // subsample_factor[1],
    )

    # Read the data, and decimate if specified
    resamp = gdal.GRA_NearestNeighbour
    if band is None:
        count = ds.RasterCount
        out = np.empty((count, nrows_out, ncols_out), dtype=dt)
        ds.ReadAsArray(xoff, yoff, xsize, ysize, buf_obj=out, resample_alg=resamp)
        if count == 1:
            out = out[0]
    else:
        out = np.empty((nrows_out, ncols_out), dtype=dt)
        bnd = ds.GetRasterBand(band)
        bnd.ReadAsArray(xoff, yoff, xsize, ysize, buf_obj=out, resample_alg=resamp)

    if not masked:
        return out
    # Get the nodata value
    nd = get_raster_nodata(filename)
    if nd is not None and np.isnan(nd):
        return np.ma.masked_invalid(out)
    else:
        return np.ma.masked_equal(out, nd)


def gdal_to_numpy_type(gdal_type: str | int) -> np.dtype:
    """Convert gdal type to numpy type."""
    if isinstance(gdal_type, str):
        gdal_type = gdal.GetDataTypeByName(gdal_type)
    return np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type))


def get_raster_nodata(filename: PathOrStr, band: int = 1) -> float | None:
    """Get the nodata value from a file.

    Parameters
    ----------
    filename : PathOrStr
        Path to the file to load.
    band : int, optional
        Band to get nodata value for, by default 1.

    Returns
    -------
    Optional[float]
        Nodata value, or None if not found.

    """
    nodatas = _get_dataset_attr(filename, "nodatavals")
    return nodatas[band - 1]


def get_raster_crs(filename: PathOrStr) -> CRS:
    """Get the CRS from a file.

    Parameters
    ----------
    filename : PathOrStr
        Path to the file to load.

    Returns
    -------
    CRS
        pyproj CRS for `filename`

    """
    return _get_dataset_attr(filename, "crs")


def get_raster_transform(filename: PathOrStr) -> Affine:
    """Get the rasterio `Affine` transform from a file.

    Parameters
    ----------
    filename : PathOrStr
        Path to the file to load.

    Returns
    -------
    List[float]
        6 floats representing a GDAL Geotransform.

    """
    return _get_dataset_attr(filename, "transform")


def get_raster_gt(filename: PathOrStr) -> list[float]:
    """Get the gdal geotransform from a file.

    Parameters
    ----------
    filename : PathOrStr
        Path to the file to load.

    Returns
    -------
    Affine
        Two dimensional affine transform for 2D linear mapping.

    """
    return get_raster_transform(filename).to_gdal()


def get_raster_dtype(filename: PathOrStr, band: int = 1) -> np.dtype:
    """Get the numpy data type from a raster file.

    Parameters
    ----------
    filename : PathOrStr
        Path to the file to load.
    band : int, optional
        Band to get nodata value for, by default 1.

    Returns
    -------
    np.dtype
        Data type.

    """
    dtype_per_band = _get_dataset_attr(filename, "dtypes")
    return np.dtype(dtype_per_band[band - 1])


def get_raster_driver(filename: PathOrStr) -> str:
    """Get the GDAL driver `ShortName` from a file.

    Parameters
    ----------
    filename : PathOrStr
        Path to the file to load.

    Returns
    -------
    str
        Driver name.

    """
    return _get_dataset_attr(filename, "driver")


def get_raster_bounds(filename: PathOrStr) -> Bbox:
    """Get the (left, bottom, right, top) bounds of the image."""
    return _get_dataset_attr(filename, "bounds")


def _get_dataset_attr(filename: PathOrStr, attr_name: str) -> Any:
    with rio.open(filename) as src:
        return getattr(src, attr_name)
