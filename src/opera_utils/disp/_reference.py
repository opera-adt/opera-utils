from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from affine import Affine
from numpy.typing import ArrayLike
from pyproj import CRS, Transformer

from ._enums import ReferenceMethod

logger = logging.getLogger("opera_utils")


def get_reference_values(
    da: xr.DataArray,
    *,
    method: str | ReferenceMethod,
    row: int | None = None,
    col: int | None = None,
    lon: float | None = None,
    lat: float | None = None,
    crs: CRS | None = None,
    transform: Affine | None = None,
    border_pixels: int = 3,
    good_pixel_mask: ArrayLike | None = None,
) -> xr.DataArray:
    """Compute the reference values for each epoch.

    Returns a 1-D DataArray of shape (time,) holding the reference value to
    subtract from `da` for each epoch.

    Parameters
    ----------
    da : xr.DataArray
        Displacement stack after rebasing, shape (time, y, x).
    method : str | ReferenceMethod
        Reference method to use.
    row : int, optional
        Row index for point reference.
    col : int, optional
        Column index for point reference.
    lon : float, optional
        Longitude (in degrees) for point reference.
    lat : float, optional
        Latitude (in degrees) for point reference.
    crs : CRS, optional
        Coordinate Reference System of the frame.
    transform : Affine, optional
        Affine transform of the frame's UTM grid.
    border_pixels : int, optional
        For ReferenceMethod.BORDER, number of pixels to use for border median.
        Defaults to 3.
    good_pixel_mask : ArrayLike, optional
        For ReferenceMethod.BORDER, a DataArray mask to use to hide certain
        pixels from the border median calculation.
        For example, the `water_mask` could be used.
        Should be a boolean array where `True` indicates good pixels.

    """
    method = ReferenceMethod(method)
    # ----  Point -- row/col OR lon/lat  ----
    if method is ReferenceMethod.POINT:
        row, col = _get_reference_row_col(
            row, col, lon, lat, crs=crs, transform=transform
        )
        ref_vals = da[:, row, col]
        return ref_vals

    if method is ReferenceMethod.MEDIAN:
        return da.median(dim=("y", "x"), skipna=True)

    if method is ReferenceMethod.BORDER:
        da_masked = da.where(good_pixel_mask) if good_pixel_mask is not None else da
        border = _get_border_pixels(da_masked, border_pixels)
        return border.median(dim="pixels", skipna=True)

    if method is ReferenceMethod.HIGH_COHERENCE:
        da_masked = da.where(good_pixel_mask)
        return da_masked.median(dim=("y", "x"), skipna=True)

    msg = f"Unknown ReferenceMethod {method}"
    raise ValueError(msg)


def _get_reference_row_col(
    row: int | None,
    col: int | None,
    lon: float | None,
    lat: float | None,
    crs: CRS | None,
    transform: Affine | None,
) -> tuple[int, int]:
    if row is None or col is None:
        if lon is None or lat is None or crs is None or transform is None:
            msg = "Need (row, col) or (lon, lat & crs & transform)"
            raise ValueError(msg)
        # Convert lon/lat to row/col once, using any product's grid
        row, col = _convert_lonlat_to_rowcol(lon, lat, crs, transform)
    return row, col


def _convert_lonlat_to_rowcol(
    lon: float,
    lat: float,
    crs: CRS,
    transform: Affine,
) -> tuple[int, int]:
    """Convert longitude/latitude to row/column indices.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with x/y coordinates.
    lon : float
        Longitude in degrees.
    lat : float
        Latitude in degrees.
    crs : CRS
        UTM Coordinate Reference System of the frame.
    transform : Affine
        Affine transform of the frame's UTM grid.

    Returns
    -------
    tuple[int, int]
        Row and column indices.

    """
    transformer_from_latlon = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True,
    )
    xx, yy = transformer_from_latlon.transform(lon, lat, radians=False)
    # Now transform from the grid x, y to row, col using the inverse of the transform
    col, row = ~transform * (xx, yy)
    return round(row), round(col)


def _get_border_pixels(
    da: xr.DataArray,
    num_border_pixels: int,
) -> xr.DataArray:
    """Extract border pixels from a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray.
    num_border_pixels : int
        Number of pixels to use for border.

    Returns
    -------
    xr.DataArray
        Concatenated border pixels.

    """
    mask = np.zeros(da.shape[-2:], dtype=bool)
    mask[:num_border_pixels, :] = True  # top
    mask[-num_border_pixels:, :] = True  # bottom
    mask[:, :num_border_pixels] = True  # left
    mask[:, -num_border_pixels:] = True  # right
    # Stack the border regions together
    return da.where(mask).stack(pixels=("y", "x")).dropna("pixels")
