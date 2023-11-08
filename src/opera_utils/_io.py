from os import fspath

from osgeo import gdal

from ._types import Filename

__all__ = [
    "get_raster_gt",
]


def get_raster_gt(filename: Filename) -> list[float]:
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
    ds = gdal.Open(fspath(filename))
    gt = ds.GetGeoTransform()
    return gt
