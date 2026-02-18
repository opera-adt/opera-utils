"""Tests for opera_utils.nisar._product module."""

from __future__ import annotations

import os
from dataclasses import is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pyproj
import pytest

from opera_utils._cmr import get_download_url as _get_download_url
from opera_utils.constants import UrlType
from opera_utils.nisar._product import (
    NISAR_GSLC_GRIDS,
    GslcProduct,
    OrbitDirection,
    OutOfBoundsError,
    _to_datetime,
)

# Example filename from the NISAR naming convention (see constants.py)
FILE_1 = "NISAR_L2_PR_GSLC_004_076_A_022_2005_QPDH_A_20251103T110514_20251103T110549_X05007_N_F_J_001.h5"
FILE_2 = "NISAR_L2_PR_GSLC_005_076_A_022_2005_QPDH_A_20251115T110514_20251115T110549_X05008_N_F_J_001.h5"
FILE_DESCENDING = "NISAR_L2_PR_GSLC_004_076_D_022_2005_QPDH_A_20251103T110514_20251103T110549_X05007_N_F_J_001.h5"

# Expected parsed values for FILE_1
EXPECTED = {
    "project": "NISAR",
    "level": "L2",
    "mode": "PR",
    "product_type": "GSLC",
    "cycle_number": 4,
    "relative_orbit_number": 76,
    "orbit_direction": OrbitDirection.ASCENDING,
    "track_frame_number": 22,
    "subswath_id": "2005",
    "polarizations": "QPDH",
    "look_direction": "A",
    "start_datetime": datetime(2025, 11, 3, 11, 5, 14, tzinfo=timezone.utc),
    "end_datetime": datetime(2025, 11, 3, 11, 5, 49, tzinfo=timezone.utc),
    "composite_release_id": "X05007",
    "processing_level": "N",
    "coverage_indicator": "F",
    "major_version": "J",
    "minor_version": 1,
}

# Mock EPSG and coordinate arrays for h5 tests
MOCK_EPSG = 32637  # UTM zone 37N
MOCK_NROWS = 100
MOCK_NCOLS = 200
MOCK_X_COORDS = np.linspace(400000, 420000, MOCK_NCOLS)
MOCK_Y_COORDS = np.linspace(1510000, 1500000, MOCK_NROWS)  # decreasing (N to S)


@pytest.fixture
def gslc_h5(tmp_path):
    """Create a minimal GSLC HDF5 file for testing."""
    h5path = tmp_path / "test_gslc.h5"
    with h5py.File(h5path, "w") as f:
        freq_path = f"{NISAR_GSLC_GRIDS}/frequencyA"
        freq_group = f.create_group(freq_path)

        # Coordinate datasets
        freq_group.create_dataset("xCoordinates", data=MOCK_X_COORDS)
        freq_group.create_dataset("yCoordinates", data=MOCK_Y_COORDS)
        freq_group.create_dataset("xCoordinateSpacing", data=100.0)
        freq_group.create_dataset("yCoordinateSpacing", data=-100.0)
        freq_group.create_dataset("projection", data=MOCK_EPSG)

        # Polarization datasets (small complex data)
        rng = np.random.default_rng(42)
        data = (
            rng.standard_normal((MOCK_NROWS, MOCK_NCOLS))
            + 1j * rng.standard_normal((MOCK_NROWS, MOCK_NCOLS))
        ).astype(np.complex64)
        freq_group.create_dataset("HH", data=data)
        freq_group.create_dataset("VV", data=data * 0.5)

    return h5path


class TestToDatetime:
    def test_parses_nisar_format(self):
        result = _to_datetime("20251103T110514")
        assert result == datetime(2025, 11, 3, 11, 5, 14, tzinfo=timezone.utc)

    def test_has_utc_timezone(self):
        result = _to_datetime("20240101T000000")
        assert result.tzinfo == timezone.utc


class TestOrbitDirection:
    def test_ascending(self):
        assert OrbitDirection("A") == OrbitDirection.ASCENDING
        assert str(OrbitDirection.ASCENDING) == "A"

    def test_descending(self):
        assert OrbitDirection("D") == OrbitDirection.DESCENDING
        assert str(OrbitDirection.DESCENDING) == "D"


class TestGslcProduct:
    def test_is_dataclass(self):
        assert is_dataclass(GslcProduct)

    def test_from_filename_valid(self):
        product = GslcProduct.from_filename(FILE_1)
        for key, expected_val in EXPECTED.items():
            assert getattr(product, key) == expected_val, f"Mismatch for {key}"
        assert product.filename == FILE_1

    def test_from_filename_path(self):
        product = GslcProduct.from_filename(Path(FILE_1))
        assert product.cycle_number == 4

    def test_from_filename_with_directory(self):
        product = GslcProduct.from_filename(f"/some/path/{FILE_1}")
        assert product.cycle_number == 4
        assert product.filename == f"/some/path/{FILE_1}"

    def test_from_filename_descending(self):
        product = GslcProduct.from_filename(FILE_DESCENDING)
        assert product.orbit_direction == OrbitDirection.DESCENDING

    def test_from_filename_invalid(self):
        with pytest.raises(ValueError, match="Invalid NISAR GSLC filename format"):
            GslcProduct.from_filename("not_a_valid_file.h5")

    def test_track_frame_id(self):
        product = GslcProduct.from_filename(FILE_1)
        assert product.track_frame_id == "004_076_A_022"

    def test_track_frame_id_descending(self):
        product = GslcProduct.from_filename(FILE_DESCENDING)
        assert product.track_frame_id == "004_076_D_022"

    def test_version(self):
        product = GslcProduct.from_filename(FILE_1)
        assert product.version == "J.001"

    def test_fspath(self):
        product = GslcProduct.from_filename(FILE_1)
        assert os.fspath(product) == FILE_1

    def test_size_in_bytes_none_for_nonexistent(self):
        product = GslcProduct.from_filename(FILE_1)
        assert product.size_in_bytes is None

    # --- get_dataset_path ---

    def test_get_dataset_path_defaults(self):
        product = GslcProduct.from_filename(FILE_1)
        assert product.get_dataset_path() == f"{NISAR_GSLC_GRIDS}/frequencyA/HH"

    def test_get_dataset_path_custom(self):
        product = GslcProduct.from_filename(FILE_1)
        path = product.get_dataset_path(frequency="B", polarization="VV")
        assert path == f"{NISAR_GSLC_GRIDS}/frequencyB/VV"

    def test_get_dataset_path_invalid_frequency(self):
        product = GslcProduct.from_filename(FILE_1)
        with pytest.raises(ValueError, match="Invalid frequency"):
            product.get_dataset_path(frequency="C")

    def test_get_dataset_path_invalid_polarization(self):
        product = GslcProduct.from_filename(FILE_1)
        with pytest.raises(ValueError, match="Invalid polarization"):
            product.get_dataset_path(polarization="XX")

    # --- HDF5 access methods ---

    def test_get_available_polarizations(self, gslc_h5):
        product = GslcProduct.from_filename(FILE_1)
        with h5py.File(gslc_h5) as f:
            pols = product.get_available_polarizations(f)
        assert set(pols) == {"HH", "VV"}

    def test_get_available_polarizations_missing_frequency(self, gslc_h5):
        product = GslcProduct.from_filename(FILE_1)
        with h5py.File(gslc_h5) as f:
            pols = product.get_available_polarizations(f, frequency="B")
        assert pols == []

    def test_get_available_frequencies(self, gslc_h5):
        product = GslcProduct.from_filename(FILE_1)
        with h5py.File(gslc_h5) as f:
            freqs = product.get_available_frequencies(f)
        assert freqs == ["A"]

    def test_get_shape(self, gslc_h5):
        product = GslcProduct.from_filename(FILE_1)
        with h5py.File(gslc_h5) as f:
            shape = product.get_shape(f)
        assert shape == (MOCK_NROWS, MOCK_NCOLS)

    def test_read_subset(self, gslc_h5):
        product = GslcProduct.from_filename(FILE_1)
        with h5py.File(gslc_h5) as f:
            subset = product.read_subset(f, rows=slice(10, 30), cols=slice(50, 80))
        assert subset.shape == (20, 30)
        assert np.iscomplexobj(subset)

    def test_get_epsg(self, gslc_h5):
        product = GslcProduct.from_filename(FILE_1)
        with h5py.File(gslc_h5) as f:
            epsg = product.get_epsg(f)
        assert epsg == MOCK_EPSG

    def test_get_epsg_missing_projection(self, gslc_h5):
        # Remove projection dataset
        with h5py.File(gslc_h5, "a") as f:
            del f[f"{NISAR_GSLC_GRIDS}/frequencyA/projection"]

        product = GslcProduct.from_filename(FILE_1)
        with (
            h5py.File(gslc_h5) as f,
            pytest.raises(ValueError, match="No projection dataset found"),
        ):
            product.get_epsg(f)

    def test_get_coordinates(self, gslc_h5):
        product = GslcProduct.from_filename(FILE_1)
        with h5py.File(gslc_h5) as f:
            x, y = product.get_coordinates(f)
        np.testing.assert_array_equal(x, MOCK_X_COORDS)
        np.testing.assert_array_equal(y, MOCK_Y_COORDS)

    def test_lonlat_to_rowcol(self, gslc_h5):
        product = GslcProduct.from_filename(FILE_1)
        # Convert center of grid to lon/lat, then back to row/col
        t = pyproj.Transformer.from_crs(
            f"EPSG:{MOCK_EPSG}", "EPSG:4326", always_xy=True
        )
        center_x = MOCK_X_COORDS[MOCK_NCOLS // 2]
        center_y = MOCK_Y_COORDS[MOCK_NROWS // 2]
        lon, lat = t.transform(center_x, center_y)

        with h5py.File(gslc_h5) as f:
            row, col = product.lonlat_to_rowcol(f, lon, lat)

        # Should be close to the center indices
        assert abs(row - MOCK_NROWS // 2) <= 1
        assert abs(col - MOCK_NCOLS // 2) <= 1

    def test_lonlat_to_rowcol_out_of_bounds(self, gslc_h5):
        product = GslcProduct.from_filename(FILE_1)
        # Convert a point far past the grid boundaries to lon/lat
        t = pyproj.Transformer.from_crs(
            f"EPSG:{MOCK_EPSG}", "EPSG:4326", always_xy=True
        )
        lon_oob, lat_oob = t.transform(
            MOCK_X_COORDS[-1] + 50000,  # way past east edge
            MOCK_Y_COORDS[-1] - 50000,  # way past south edge
        )
        with h5py.File(gslc_h5) as f, pytest.raises(OutOfBoundsError):
            product.lonlat_to_rowcol(f, lon_oob, lat_oob)


class TestFromUmm:
    @pytest.fixture
    def umm_data(self):
        return {
            "RelatedUrls": [
                {
                    "URL": f"https://datapool.asf.alaska.edu/GSLC/NISAR/{FILE_1}",
                    "Type": "GET DATA",
                },
                {
                    "URL": f"s3://nisar-gslc-bucket/{FILE_1}",
                    "Type": "GET DATA VIA DIRECT ACCESS",
                },
            ],
            "DataGranule": {
                "ArchiveAndDistributionInformation": [{"SizeInBytes": 1_000_000_000}]
            },
        }

    def test_from_umm_https(self, umm_data):
        product = GslcProduct.from_umm(umm_data, url_type=UrlType.HTTPS)
        assert product.filename.startswith("https://")
        assert product.size_in_bytes == 1_000_000_000
        assert product.cycle_number == 4

    def test_from_umm_s3(self, umm_data):
        product = GslcProduct.from_umm(umm_data, url_type=UrlType.S3)
        assert product.filename.startswith("s3://")

    def test_from_umm_no_size_info(self, umm_data):
        del umm_data["DataGranule"]
        product = GslcProduct.from_umm(umm_data)
        assert product.size_in_bytes is None


class TestGetDownloadUrl:
    def test_prefers_h5_url(self):
        umm = {
            "RelatedUrls": [
                {"URL": "https://example.com/file.xml", "Type": "GET DATA"},
                {"URL": "https://example.com/file.h5", "Type": "GET DATA"},
            ]
        }
        assert (
            _get_download_url(umm, UrlType.HTTPS, filename_suffix=".h5")
            == "https://example.com/file.h5"
        )

    def test_falls_back_to_non_h5(self):
        umm = {
            "RelatedUrls": [
                {"URL": "https://example.com/file.nc", "Type": "GET DATA"},
            ]
        }
        assert (
            _get_download_url(umm, UrlType.HTTPS, filename_suffix=".h5")
            == "https://example.com/file.nc"
        )

    def test_s3_url(self):
        umm = {
            "RelatedUrls": [
                {"URL": "s3://bucket/file.h5", "Type": "GET DATA VIA DIRECT ACCESS"},
            ]
        }
        assert _get_download_url(umm, UrlType.S3) == "s3://bucket/file.h5"

    def test_no_matching_url_raises(self):
        umm = {
            "RelatedUrls": [
                {"URL": "ftp://example.com/file.h5", "Type": "GET DATA"},
            ],
            "GranuleUR": "test_granule",
        }
        with pytest.raises(ValueError, match="No download URL found"):
            _get_download_url(umm, UrlType.HTTPS)

    def test_invalid_protocol_raises(self):
        with pytest.raises(ValueError, match="Unknown protocol"):
            _get_download_url({}, protocol="ftp")
