from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from pyproj import CRS, Transformer

from ._enums import ReferenceMethod


def _convert_lonlat_to_rowcol(
    da: xr.DataArray,
    lon: float,
    lat: float,
    crs_wkt: str,
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
    crs_wkt : str
        Well-Known Text representation of the coordinate reference system.

    Returns
    -------
    tuple[int, int]
        Row and column indices.

    """
    crs = CRS.from_wkt(crs_wkt)
    transformer_from_latlon = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True,
    )
    xx, yy = transformer_from_latlon.transform(lon, lat, radians=False)
    r = int((np.abs(da.y.values - yy)).argmin())
    c = int((np.abs(da.x.values - xx)).argmin())
    return r, c


def _get_border_pixels(
    da: xr.DataArray,
    border_pixels: int,
) -> xr.DataArray:
    """Extract border pixels from a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray.
    border_pixels : int
        Number of pixels to use for border.

    Returns
    -------
    xr.DataArray
        Concatenated border pixels.

    """
    bb = border_pixels
    # Stack the border regions together
    top = da.isel(y=slice(0, bb))
    bottom = da.isel(y=slice(-bb, None))
    left = da.isel(x=slice(0, bb))
    right = da.isel(x=slice(-bb, None))

    # Flatten spatial dimensions for each border region
    top_flat = top.stack(pixels=("y", "x"))
    bottom_flat = bottom.stack(pixels=("y", "x"))
    left_flat = left.stack(pixels=("y", "x"))
    right_flat = right.stack(pixels=("y", "x"))

    # Concatenate all border pixels
    border = xr.concat(
        [top_flat, bottom_flat, left_flat, right_flat],
        dim="pixels",
    )
    return border


def _compute_coherence_harmonic_mean(
    coherence: ArrayLike,
    coherence_thresh: float,
) -> np.ndarray:
    """Compute harmonic mean of coherence along time axis.

    Parameters
    ----------
    coherence : ArrayLike
        Coherence dataset.
    coherence_thresh : float
        Coherence threshold for masking.

    Returns
    -------
    np.ndarray
        2D coherence mask (True where coherence > threshold).

    """
    # Harmonic mean along time
    if np.ndim(coherence) > 2:
        arr_coherence = np.asarray(coherence)
        with np.errstate(divide="ignore", invalid="ignore"):
            coh_2d = np.nanmedian(len(arr_coherence) / (1.0 / arr_coherence), axis=0)
    else:
        coh_2d = np.asarray(coherence)

    return coh_2d > coherence_thresh


def get_reference_values(
    da: xr.DataArray,
    *,
    method: str | ReferenceMethod,
    row: int | None = None,
    col: int | None = None,
    lon: float | None = None,
    lat: float | None = None,
    crs_wkt: str | None = None,
    border_pixels: int = 1,
    coherence: ArrayLike | None = None,
    coherence_thresh: float = 0.7,
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
    crs_wkt : str, optional
        Well-Known Text representation of the coordinate reference system.
    border_pixels : int, optional
        Number of pixels to use for border median.
    coherence : ArrayLike, optional
        Coherence dataset.
    coherence_thresh : float, optional
        Coherence threshold.

    """
    method = ReferenceMethod(method)
    # ----  Point -- row/col OR lon/lat  ----
    if method is ReferenceMethod.POINT:
        if row is None or col is None:
            # Convert lon/lat to row/col once, using any product's grid
            if lon is None or lat is None or crs_wkt is None:
                msg = "Need (row, col) or (lon, lat & crs_wkt)"
                raise ValueError(msg)
            row, col = _convert_lonlat_to_rowcol(da, lon, lat, crs_wkt)
        ref_vals = da[:, row, col]
        return ref_vals

    # ----  Median of whole scene  ----
    if method is ReferenceMethod.MEDIAN:
        return da.median(dim=("y", "x"), skipna=True)

    # ----  Border median  ----
    if method is ReferenceMethod.BORDER:
        border = _get_border_pixels(da, border_pixels)
        return border.median(dim="pixels", skipna=True)

    # ----  High-coherence mask  ----
    if method is ReferenceMethod.HIGH_COHERENCE:
        if coherence is None:
            msg = "Need coherence dataset"
            raise ValueError(msg)
        mask = _compute_coherence_harmonic_mean(coherence, coherence_thresh)
        masked = da.where(mask)
        return masked.median(dim=("y", "x"), skipna=True)

    msg = f"Unknown ReferenceMethod {method}"
    raise ValueError(msg)
