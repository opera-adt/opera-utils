from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from opera_utils.disp._enums import ReferenceMethod
from opera_utils.disp._reference import (
    _compute_coherence_harmonic_mean,
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

        # Should have top, bottom, left, right borders
        expected_pixels = (
            2 * 1 * sample_da.sizes["x"]  # top + bottom
            + 2 * 1 * sample_da.sizes["y"]  # left + right
        )
        assert border.sizes["pixels"] == expected_pixels

    def test_get_border_pixels_large(self, sample_da):
        """Test border pixel extraction with large border."""
        border_pixels = 3
        border = _get_border_pixels(sample_da, border_pixels)

        assert border.sizes["time"] == sample_da.sizes["time"]
        assert "pixels" in border.dims


class TestComputeCoherenceHarmonicMean:
    def test_compute_coherence_3d(self, sample_coherence):
        """Test coherence harmonic mean with 3D coherence."""
        result = _compute_coherence_harmonic_mean(sample_coherence)

        assert result.shape == sample_coherence.shape[1:]  # Should be 2D
        assert result.dtype == np.float64

    def test_compute_coherence_2d(self):
        """Test coherence harmonic mean with 2D coherence."""
        coherence_2d = np.random.rand(10, 15) * 0.9 + 0.1
        result = _compute_coherence_harmonic_mean(coherence_2d)

        assert result.shape == coherence_2d.shape
        assert result.dtype == np.float64

    def test_coherence_threshold(self):
        """Test coherence computation with known values."""
        # Create coherence with known values
        coherence = np.array([[0.3, 0.7], [0.9, 0.2]])
        result = _compute_coherence_harmonic_mean(coherence)

        # Should return the input for 2D arrays
        np.testing.assert_array_equal(result, coherence)


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
