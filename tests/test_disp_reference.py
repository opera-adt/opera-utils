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
def sample_crs_wkt():
    """Sample CRS WKT string for testing coordinate conversion."""
    # Use a simple projection for testing
    return (
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS'
        ' 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
    )


class TestConvertLonLatToRowCol:
    def test_convert_lonlat_to_rowcol(self, sample_da, sample_crs_wkt):
        """Test coordinate conversion from lon/lat to row/col."""
        lon, lat = -119.5, 40.5
        row, col = _convert_lonlat_to_rowcol(sample_da, lon, lat, sample_crs_wkt)

        assert isinstance(row, int)
        assert isinstance(col, int)
        assert 0 <= row < sample_da.sizes["y"]
        assert 0 <= col < sample_da.sizes["x"]

    def test_convert_edge_coordinates(self, sample_da, sample_crs_wkt):
        """Test coordinate conversion at edges."""
        # Test corner coordinates
        lon_min, lat_min = float(sample_da.x.min()), float(sample_da.y.min())
        row, col = _convert_lonlat_to_rowcol(
            sample_da, lon_min, lat_min, sample_crs_wkt
        )

        assert row >= 0
        assert col >= 0


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
        thresh = 0.5
        mask = _compute_coherence_harmonic_mean(sample_coherence, thresh)

        assert mask.shape == sample_coherence.shape[1:]  # Should be 2D
        assert mask.dtype == bool

    def test_compute_coherence_2d(self):
        """Test coherence harmonic mean with 2D coherence."""
        coherence_2d = np.random.rand(10, 15) * 0.9 + 0.1
        thresh = 0.5
        mask = _compute_coherence_harmonic_mean(coherence_2d, thresh)

        assert mask.shape == coherence_2d.shape
        assert mask.dtype == bool

    def test_coherence_threshold(self):
        """Test coherence thresholding."""
        # Create coherence with known values
        coherence = np.array([[0.3, 0.7], [0.9, 0.2]])
        thresh = 0.5
        mask = _compute_coherence_harmonic_mean(coherence, thresh)

        expected = np.array([[False, True], [True, False]])
        np.testing.assert_array_equal(mask, expected)


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

    def test_point_method_lonlat(self, sample_da, sample_crs_wkt):
        """Test POINT method with lon/lat specification."""
        lon, lat = -119.5, 40.5
        ref_vals = get_reference_values(
            sample_da,
            method=ReferenceMethod.POINT,
            lon=lon,
            lat=lat,
            crs_wkt=sample_crs_wkt,
        )

        assert ref_vals.sizes["time"] == sample_da.sizes["time"]
        assert ref_vals.ndim == 1

    def test_point_method_missing_params(self, sample_da):
        """Test POINT method with missing parameters."""
        with pytest.raises(
            ValueError, match="Need \\(row, col\\) or \\(lon, lat & crs_wkt\\)"
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
        coherence_thresh = 0.5
        ref_vals = get_reference_values(
            sample_da,
            method=ReferenceMethod.HIGH_COHERENCE,
            coherence=sample_coherence,
            coherence_thresh=coherence_thresh,
        )

        assert ref_vals.sizes["time"] == sample_da.sizes["time"]
        assert ref_vals.ndim == 1

    def test_high_coherence_method_missing_coherence(self, sample_da):
        """Test HIGH_COHERENCE method without coherence data."""
        with pytest.raises(ValueError, match="Need coherence dataset"):
            get_reference_values(
                sample_da,
                method=ReferenceMethod.HIGH_COHERENCE,
            )

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
