from __future__ import annotations

import logging
from os import fspath
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from osgeo import gdal, gdal_array
from pyproj import CRS

from ._types import Bbox, Filename

__all__ = [
    "load_gdal",
]


def load_gdal(
    filename: Filename,
    band: Optional[int] = None,
    subsample_factor: Union[int, tuple[int, int]] = 1,
    overview: Optional[int] = None,
    rows: Optional[slice] = None,
    cols: Optional[slice] = None,
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
        raise IndexError(
            f"Invalid row/col slices: {rows}, {cols} for file {filename} of size"
            f" {nrows}x{ncols}"
        )
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


def get_raster_bounds(
    filename: Optional[Filename] = None, ds: Optional[gdal.Dataset] = None
) -> Bbox:
    """Get the (left, bottom, right, top) bounds of the image."""
    if ds is None:
        if filename is None:
            raise ValueError("Must provide either `filename` or `ds`")
        ds = gdal.Open(fspath(filename))

    gt = ds.GetGeoTransform()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize

    x = 0
    y = 0
    left = gt[0] + x * gt[1] + y * gt[2]
    top = gt[3] + x * gt[4] + y * gt[5]

    x = xsize
    y = ysize

    right = gt[0] + x * gt[1] + y * gt[2]
    bottom = gt[3] + x * gt[4] + y * gt[5]

    return (left, bottom, right, top)


def get_raster_crs(filename: Filename) -> CRS:
    """Get the CRS from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.

    Returns
    -------
    CRS
        CRS.
    """
    ds = gdal.Open(fspath(filename))
    crs = CRS.from_wkt(ds.GetProjection())
    return crs


def gdal_to_numpy_type(gdal_type: Union[str, int]) -> np.dtype:
    """Convert gdal type to numpy type."""
    if isinstance(gdal_type, str):
        gdal_type = gdal.GetDataTypeByName(gdal_type)
    return np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type))


def get_raster_nodata(filename: Filename, band: int = 1) -> Optional[float]:
    """Get the nodata value from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.
    band : int, optional
        Band to get nodata value for, by default 1.

    Returns
    -------
    Optional[float]
        Nodata value, or None if not found.
    """
    ds = gdal.Open(fspath(filename))
    nodata = ds.GetRasterBand(band).GetNoDataValue()
    return nodata
