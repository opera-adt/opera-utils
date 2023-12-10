import numpy as np
import pytest
from pyproj import CRS

try:
    import rasterio as rio
    import rasterio.transform

    RASTERIO_MISSING = False
except ImportError:
    RASTERIO_MISSING = True


if RASTERIO_MISSING:
    pytest.skip(reason="Rasterio not installed")

from opera_utils import _io


@pytest.fixture(scope="module")
def bounds():
    return (-104, 30, -103, 33)


@pytest.fixture(scope="module")
def crs():
    return CRS.from_epsg(32611)


@pytest.fixture(scope="module")
def shape():
    return 30, 20


@pytest.fixture(scope="module")
def transform(bounds, shape):
    height, width = shape
    return rasterio.transform.from_bounds(*bounds, width=width, height=height)


@pytest.fixture(scope="module")
def raster_file(tmp_path_factory, crs, transform, shape):
    height, width = shape
    img = np.random.rand(height, width).astype("float32")

    filename = tmp_path_factory.mktemp("data") / "raster.tif"
    with rio.open(
        filename,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        dtype="float32",
        count=1,
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as src:
        src.write(img, 1)

    return filename


def test_get_raster_bounds(raster_file, bounds):
    assert _io.get_raster_bounds(raster_file) == bounds


def test_get_raster_crs(raster_file, crs):
    assert _io.get_raster_crs(raster_file) == crs


def test_get_raster_nodata(raster_file):
    assert np.isnan(_io.get_raster_nodata(raster_file))


def test_get_raster_transform(raster_file, transform):
    assert _io.get_raster_transform(raster_file) == transform


def test_get_raster_get(raster_file, transform):
    assert _io.get_raster_gt(raster_file) == transform.to_gdal()


def test_get_raster_dtype(
    raster_file,
):
    assert _io.get_raster_dtype(raster_file) == np.float32


def test_get_raster_driver(raster_file):
    assert _io.get_raster_driver(raster_file) == "GTiff"
