from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from opera_utils.disp._enums import ReferenceMethod
from opera_utils.disp._reference import (
    _convert_lonlat_to_rowcol,
    _get_border_pixels,
    get_reference_values,
)


@pytest.fixture
def sample_da():
    """Create a sample displacement DataArray for testing."""
    np.random.seed(42)
    data = np.random.randn(5, 10, 15)  # time, y, x
    time = np.arange(5)
    y = np.linspace(40.0, 41.0, 10)  # lat-like
    x = np.linspace(-120.0, -119.0, 15)  # lon-like

    da = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": time, "y": y, "x": x},
    )
    return da


@pytest.fixture
def sample_coherence():
    """Create sample coherence data."""
    np.random.seed(42)
    return np.random.rand(5, 10, 15) * 0.9 + 0.1  # values between 0.1 and 1.0


@pytest.fixture
def sample_crs():
    """Sample CRS for testing coordinate conversion."""
    from pyproj import CRS

    return CRS.from_epsg(4326)


@pytest.fixture
def sample_transform():
    """Sample transform for testing coordinate conversion."""
    from affine import Affine

    # Simple transform for geographic coordinates
    return Affine.from_gdal(-120.0, 0.0666667, 0.0, 41.0, 0.0, -0.1)


class TestConvertLonLatToRowCol:
    def test_convert_lonlat_to_rowcol(self, sample_crs, sample_transform):
        """Test coordinate conversion from lon/lat to row/col."""
        lon, lat = -119.5, 40.5
        row, col = _convert_lonlat_to_rowcol(lon, lat, sample_crs, sample_transform)

        assert isinstance(row, (int, float))
        assert isinstance(col, (int, float))

    def test_convert_edge_coordinates(self, sample_crs, sample_transform):
        """Test coordinate conversion at edges."""
        # Test corner coordinates
        lon_min, lat_min = -120.0, 40.0
        row, col = _convert_lonlat_to_rowcol(
            lon_min, lat_min, sample_crs, sample_transform
        )

        assert isinstance(row, (int, float))
        assert isinstance(col, (int, float))


class TestGetBorderPixels:
    def test_get_border_pixels_basic(self, sample_da):
        """Test basic border pixel extraction."""
        border_pixels = 2
        border = _get_border_pixels(sample_da, border_pixels)

        assert "pixels" in border.dims
        assert border.sizes["time"] == sample_da.sizes["time"]

    def test_get_border_pixels_single(self, sample_da):
        """Test border pixel extraction with single pixel border."""
        border_pixels = 1
        border = _get_border_pixels(sample_da, border_pixels)

        # The border includes overlapping corners, so we calculate actual count
        # by creating the same mask used in the implementation
        mask = np.zeros(sample_da.shape[-2:], dtype=bool)
        mask[:border_pixels, :] = True  # top
        mask[-border_pixels:, :] = True  # bottom
        mask[:, :border_pixels] = True  # left
        mask[:, -border_pixels:] = True  # right
        expected_pixels = mask.sum()

        assert border.sizes["pixels"] == expected_pixels

    def test_get_border_pixels_large(self, sample_da):
        """Test border pixel extraction with large border."""
        border_pixels = 3
        border = _get_border_pixels(sample_da, border_pixels)

        assert border.sizes["time"] == sample_da.sizes["time"]
        assert "pixels" in border.dims


class TestGetReferenceValues:
    def test_point_method_rowcol(self, sample_da):
        """Test POINT method with row/col specification."""
        row, col = 5, 7
        ref_vals = get_reference_values(
            sample_da,
            method=ReferenceMethod.POINT,
            row=row,
            col=col,
        )

        expected = sample_da[:, row, col]
        xr.testing.assert_equal(ref_vals, expected)

    def test_point_method_lonlat(self, sample_da, sample_crs, sample_transform):
        """Test POINT method with lon/lat specification."""
        lon, lat = -119.5, 40.5
        ref_vals = get_reference_values(
            sample_da,
            method=ReferenceMethod.POINT,
            lon=lon,
            lat=lat,
            crs=sample_crs,
            transform=sample_transform,
        )

        assert ref_vals.sizes["time"] == sample_da.sizes["time"]
        assert ref_vals.ndim == 1

    def test_point_method_missing_params(self, sample_da):
        """Test POINT method with missing parameters."""
        with pytest.raises(
            ValueError, match="Need \\(row, col\\) or \\(lon, lat & crs & transform\\)"
        ):
            get_reference_values(sample_da, method=ReferenceMethod.POINT)

    def test_median_method(self, sample_da):
        """Test MEDIAN method."""
        ref_vals = get_reference_values(sample_da, method=ReferenceMethod.MEDIAN)

        expected = sample_da.median(dim=("y", "x"), skipna=True)
        xr.testing.assert_equal(ref_vals, expected)

    def test_border_method(self, sample_da):
        """Test BORDER method."""
        border_pixels = 2
        ref_vals = get_reference_values(
            sample_da,
            method=ReferenceMethod.BORDER,
            border_pixels=border_pixels,
        )

        assert ref_vals.sizes["time"] == sample_da.sizes["time"]
        assert ref_vals.ndim == 1

    def test_high_coherence_method(self, sample_da, sample_coherence):
        """Test HIGH_COHERENCE method."""
        # Create a mask from the coherence data
        mask = sample_coherence[0] > 0.5  # Use first time slice
        ref_vals = get_reference_values(
            sample_da,
            method=ReferenceMethod.HIGH_COHERENCE,
            good_pixel_mask=mask,
        )

        assert ref_vals.sizes["time"] == sample_da.sizes["time"]
        assert ref_vals.ndim == 1

    def test_high_coherence_method_missing_mask(self, sample_da):
        """Test HIGH_COHERENCE method without good pixel mask."""
        # Should work without mask (uses all pixels)
        ref_vals = get_reference_values(
            sample_da,
            method=ReferenceMethod.HIGH_COHERENCE,
        )
        assert ref_vals.sizes["time"] == sample_da.sizes["time"]
        assert ref_vals.ndim == 1

    def test_unknown_method(self, sample_da):
        """Test with unknown reference method."""
        with pytest.raises(ValueError, match="is not a valid ReferenceMethod"):
            get_reference_values(sample_da, method="unknown_method")

    def test_string_method_input(self, sample_da):
        """Test that string method inputs work."""
        row, col = 5, 7
        ref_vals = get_reference_values(
            sample_da,
            method="point",  # string instead of enum
            row=row,
            col=col,
        )

        expected = sample_da[:, row, col]
        xr.testing.assert_equal(ref_vals, expected)
