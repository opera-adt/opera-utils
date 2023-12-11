from __future__ import annotations

from typing import Any, cast

import numpy as np
import rasterio as rio
from affine import Affine
from pyproj import CRS

from ._types import Bbox, PathOrStr


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
    nodatas = cast(tuple[float | None], _get_dataset_attr(filename, "nodatavals"))
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
    return cast(CRS, _get_dataset_attr(filename, "crs"))


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
    return cast(Affine, _get_dataset_attr(filename, "transform"))


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
    dtype_per_band = cast(tuple[str], _get_dataset_attr(filename, "dtypes"))
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
    return cast(str, _get_dataset_attr(filename, "driver"))


def get_raster_bounds(filename: PathOrStr) -> Bbox:
    """Get the (left, bottom, right, top) bounds of the image."""
    return _get_dataset_attr(filename, "bounds")


def _get_dataset_attr(filename: PathOrStr, attr_name: str) -> Any:
    with rio.open(filename) as src:
        return getattr(src, attr_name)
