# test_disp_reader.py
from __future__ import annotations

import multiprocessing as mp
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Assuming the classes are in opera_utils.disp
from opera_utils.disp import DispProduct, DispProductStack
from opera_utils.disp._product import OutOfBoundsError
from opera_utils.disp._reader import (
    ReferenceMethod,
    _get_border,
    _get_rows_cols,
    read_lonlat,
    read_stack_lonlat,
)

FILE_1 = "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v1.0_20250318T222753Z.nc"
FILE_2 = "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160810T140756Z_v1.0_20250318T222753Z.nc"
FILE_3 = "OPERA_L3_DISP-S1_IW_F11116_VV_20160717T140755Z_20160729T140756Z_v1.0_20250318T222753Z.nc"
VALID_FILES = [FILE_1, FILE_2, FILE_3]
REF_DATES = [
    datetime(2016, 7, 5, 14, 7, 55, tzinfo=timezone.utc),
    datetime(2016, 7, 5, 14, 7, 55, tzinfo=timezone.utc),
    datetime(2016, 7, 17, 14, 7, 55, tzinfo=timezone.utc),
]
MOCK_DATA_SHAPE = (5, 6)  # Small mock data shape (rows, cols)
MOCK_HDF5_DATA_1 = (
    np.arange(MOCK_DATA_SHAPE[0] * MOCK_DATA_SHAPE[1]).reshape(MOCK_DATA_SHAPE) * 1.1
)
MOCK_HDF5_DATA_2 = (
    np.arange(MOCK_DATA_SHAPE[0] * MOCK_DATA_SHAPE[1]).reshape(MOCK_DATA_SHAPE) * 1.2
)
MOCK_HDF5_DATA_3 = (
    np.arange(MOCK_DATA_SHAPE[0] * MOCK_DATA_SHAPE[1]).reshape(MOCK_DATA_SHAPE) * 1.3
)
MOCK_HDF5_DATA_LIST = [MOCK_HDF5_DATA_1, MOCK_HDF5_DATA_2, MOCK_HDF5_DATA_3]

MOCK_HDF5_ATTRS = {"units": "m", "description": "Mock displacement"}


@pytest.fixture
def mock_product():
    """Provides a DispProduct instance."""
    # We need a real product instance for methods like lonlat_to_rowcol
    # Mock external dependencies for the product itself if needed
    with patch(
        "opera_utils.burst_frame_db.get_frame_bbox", return_value=(32610, MagicMock())
    ):
        # Mock the bounds to avoid needing real geometry
        p = DispProduct.from_filename(FILE_1)
    return p


@pytest.fixture
def mock_stack(mock_product):
    """Provides a DispProductStack instance with mocked products."""
    # Use real product parsing but potentially replace products with mocks if needed
    # For simplicity, we'll use DispProduct instances based on filenames
    # Assume test_disp_product ensures these are valid and sortable
    products = [DispProduct.from_filename(f) for f in VALID_FILES]
    # Mock the frame bbox lookup used during stack init
    with patch(
        "opera_utils.burst_frame_db.get_frame_bbox", return_value=(32610, MagicMock())
    ):
        stack = DispProductStack(products=products)
    return stack


@pytest.fixture
def mock_open_h5(monkeypatch):
    """Mocks the open_h5 function to simulate reading from HDF5."""
    mock_hf_dict = {}  # Store mock data per product filename

    def _mock_open_h5(product: DispProduct):
        # Use filename as key to return specific mock data
        filename = product.filename
        if filename not in mock_hf_dict:
            # Assign mock data sequentially if not already assigned
            data_index = len(mock_hf_dict) % len(MOCK_HDF5_DATA_LIST)
            mock_data = MOCK_HDF5_DATA_LIST[data_index]
            mock_hf_dict[filename] = mock_data

        mock_data_to_return = mock_hf_dict[filename]
        mock_dset = MagicMock()
        mock_dset.__getitem__.side_effect = lambda slices: mock_data_to_return[slices]
        mock_dset.attrs = MOCK_HDF5_ATTRS.copy()  # Return a copy

        mock_hf = MagicMock()
        mock_hf.__enter__.return_value = mock_hf  # Return self for context manager
        mock_hf.__exit__.return_value = None
        mock_hf.__getitem__.return_value = mock_dset  # hf[dset] returns mock_dset

        return mock_hf

    monkeypatch.setattr("opera_utils.disp._reader.open_h5", _mock_open_h5)
    return _mock_open_h5


@pytest.fixture
def mock_lonlat_to_rowcol(monkeypatch):
    """Mocks DispProduct.lonlat_to_rowcol."""

    # Define specific mock behaviors for different lon/lat inputs
    def _mock_transform(self, lon, lat):
        # Simple mock: return fixed row/col for specific inputs used in tests
        # These values should correspond to slices of MOCK_DATA_SHAPE
        if lon == -122.0 and lat == 37.0:
            return (0, 0)  # Top-left for slice start
        if lon == -121.8 and lat == 36.8:
            return (
                MOCK_DATA_SHAPE[0],
                MOCK_DATA_SHAPE[1],
            )  # Bottom-right for slice end
        if lon == -121.9 and lat == 36.9:
            return (2, 3)  # Single point row/col
        if lon == -121.85 and lat == 36.85:
            return (3, 4)  # Ref point row/col for point referencing
        # Raise OutOfBoundsError for values outside the mock "valid" range
        if lat < 36.7 or lat > 37.1 or lon < -122.1 or lon > -121.7:
            msg = "Mocked coordinates out of bounds"
            raise OutOfBoundsError(msg)

        # Return something plausible based on MOCK_DATA_SHAPE
        row = int(MOCK_DATA_SHAPE[0] * (37.0 - lat) / 0.2)
        col = int(MOCK_DATA_SHAPE[1] * (lon - (-122.0)) / 0.2)
        return (
            max(0, min(row, MOCK_DATA_SHAPE[0])),
            max(0, min(col, MOCK_DATA_SHAPE[1])),
        )

    monkeypatch.setattr(DispProduct, "lonlat_to_rowcol", _mock_transform)
    return _mock_transform


@pytest.fixture
def mock_rebase_timeseries(monkeypatch):
    """Mocks rebase_timeseries to just return the input."""
    mock_rebase = MagicMock(side_effect=lambda data, dates: data)  # noqa: ARG005
    monkeypatch.setattr("opera_utils.disp._reader.rebase_timeseries", mock_rebase)
    return mock_rebase


@pytest.fixture
def mock_pool(monkeypatch):
    """Mocks multiprocessing Pool and imap."""
    mock_imap_results = {}  # Store results keyed by the function being mapped

    class MockPool:
        def __init__(self, processes=None):
            self.processes = processes

        def imap(self, func, iterable):
            # Execute function sequentially and store results
            results = [func(item) for item in iterable]
            # Store results based on the function's partial args if possible
            # This helps verify which read operation was performed
            args_key = getattr(func, "args", None)
            if args_key:
                mock_imap_results[args_key] = results
            else:  # Fallback key
                mock_imap_results["unknown"] = results
            return iter(results)  # imap returns an iterator

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    # Patch mp.get_context("spawn").Pool
    mock_context = MagicMock()
    mock_context.Pool.side_effect = MockPool
    monkeypatch.setattr(mp, "get_context", lambda s: mock_context)  # noqa: ARG005

    # Return the dictionary to allow tests to inspect results
    return mock_imap_results


# This function is implicitly tested via read_lonlat, but direct tests can be added.
def test_get_rows_cols_basic(mock_product, mock_lonlat_to_rowcol):
    """Test _get_rows_cols with standard slices."""
    lon_slice = slice(-122.0, -121.8)
    lat_slice = slice(37.0, 36.8)  # Normal: high -> low
    rows, cols = _get_rows_cols(lon_slice, lat_slice, mock_product)
    assert rows == slice(0, MOCK_DATA_SHAPE[0])
    assert cols == slice(0, MOCK_DATA_SHAPE[1])


def test_get_rows_cols_out_of_bounds(mock_product):
    """Test _get_rows_cols when lonlat_to_rowcol raises OutOfBoundsError."""
    with pytest.raises(OutOfBoundsError):
        _get_rows_cols(-130.0, 40.0, mock_product)  # Outside mock valid range


@pytest.mark.usefixtures("mock_open_h5", "mock_lonlat_to_rowcol")
class TestReadLonLat:
    def test_read_lonlat_slice(self, mock_product):
        """Test reading a slice from a single product."""
        lon_slice = slice(-122.0, -121.8)
        lat_slice = slice(37.0, 36.8)
        data = read_lonlat(mock_product, lon_slice, lat_slice)

        # Based on mock_lonlat_to_rowcol returning (0,0) and (rows, cols)
        # and mock_open_h5 returning MOCK_HDF5_DATA_1 for the first product
        expected_data = MOCK_HDF5_DATA_1[0 : MOCK_DATA_SHAPE[0], 0 : MOCK_DATA_SHAPE[1]]
        np.testing.assert_array_equal(data, expected_data)

    def test_read_lonlat_point(self, mock_product):
        """Test reading a single point (float inputs) from a product."""
        lon = -121.9
        lat = 36.9
        data = read_lonlat(mock_product, lon, lat)

        # Based on mock_lonlat_to_rowcol returning (2, 3) for this point
        # Result should be 1x1 array
        expected_data = MOCK_HDF5_DATA_1[2:3, 3:4]
        assert data.shape == (1, 1)
        np.testing.assert_array_equal(data, expected_data)

    def test_read_lonlat_different_dset(self, mock_product):
        """Test reading from a different dataset name."""
        lon_slice = slice(-122.0, -121.8)
        lat_slice = slice(37.0, 36.8)
        # The mock h5 returns the same data regardless of dset name,
        # but we can verify the name was used in the mock call if needed.
        data = read_lonlat(mock_product, lon_slice, lat_slice, dset="velocity")
        expected_data = MOCK_HDF5_DATA_1[0 : MOCK_DATA_SHAPE[0], 0 : MOCK_DATA_SHAPE[1]]
        np.testing.assert_array_equal(data, expected_data)


def test_get_border():
    """Test the _get_border helper function."""
    a = np.arange(24, dtype=float).reshape(2, 3, 4)
    # Add a NaN to test nanmedian
    a[0, 0, 0] = np.nan
    expected = np.array([[[7.5]], [[17.5]]])
    np.testing.assert_allclose(_get_border(a), expected)


@pytest.mark.usefixtures(
    "mock_open_h5", "mock_lonlat_to_rowcol", "mock_rebase_timeseries", "mock_pool"
)
class TestReadStackLonLat:
    lon_slice = slice(-122.0, -121.8)
    lat_slice = slice(37.0, 36.8)
    expected_slice = (slice(0, MOCK_DATA_SHAPE[0]), slice(0, MOCK_DATA_SHAPE[1]))

    ref_lon = -121.85
    ref_lat = 36.85
    # Based on mock_lonlat_to_rowcol
    expected_ref_slice = (slice(3, 4), slice(4, 5))

    @pytest.fixture(autouse=True)
    def setup_mock_data(self, mock_open_h5):
        """Ensure mock_open_h5 uses consistent data for the stack."""
        # Pre-populate the mock data store used by mock_open_h5
        mock_open_h5._mock_hf_dict = {
            FILE_1: MOCK_HDF5_DATA_1,
            FILE_2: MOCK_HDF5_DATA_2,
            FILE_3: MOCK_HDF5_DATA_3,
        }

    def test_read_stack_none_reference(self, mock_stack, mock_rebase_timeseries):
        """Test reading stack with no referencing."""
        data, attrs = read_stack_lonlat(mock_stack, self.lon_slice, self.lat_slice)

        # Check shape: (num_products, height, width)
        assert data.shape == (
            len(mock_stack.products),
            MOCK_DATA_SHAPE[0],
            MOCK_DATA_SHAPE[1],
        )

        # Check data (should be rebased - which our mock just passes through)
        expected_data = np.stack([d[self.expected_slice] for d in MOCK_HDF5_DATA_LIST])
        np.testing.assert_array_equal(data, expected_data)
        mock_rebase_timeseries.assert_called_once()  # Ensure rebase was called

        # Check attributes
        assert attrs["units"] == "m"
        assert attrs["reference_method"] == ReferenceMethod.none.value
        assert attrs["reference_datetime"] == REF_DATES[0].isoformat()  # First ref date
        assert attrs["reference_lon"] == "None"
        assert attrs["reference_lat"] == "None"
        # Check internal attrs removed
        assert "_Netcdf4Coordinates" not in attrs
        assert "_FillValue" not in attrs

    def test_read_stack_point_reference(self, mock_stack):
        """Test reading stack with point referencing."""
        data, attrs = read_stack_lonlat(
            mock_stack,
            self.lon_slice,
            self.lat_slice,
            ref_lon=self.ref_lon,
            ref_lat=self.ref_lat,
            # reference_method implicitly set by ref_lon/lat
        )

        assert data.shape == (
            len(mock_stack.products),
            MOCK_DATA_SHAPE[0],
            MOCK_DATA_SHAPE[1],
        )

        # Verify data calculation
        # Grab the "none" version:
        unreffed_data = read_stack_lonlat(
            mock_stack,
            self.lon_slice,
            self.lat_slice,
        )[0]
        # data = "unreffed - reference", so reference = "unreffed - data"
        diff = unreffed_data - data
        assert np.all(diff[0] == MOCK_HDF5_DATA_1[self.expected_ref_slice])
        assert np.all(diff[1] == MOCK_HDF5_DATA_2[self.expected_ref_slice])
        assert np.all(diff[2] == MOCK_HDF5_DATA_3[self.expected_ref_slice])

        # Check attributes
        assert attrs["reference_method"] == ReferenceMethod.point.value
        assert attrs["reference_lon"] == self.ref_lon
        assert attrs["reference_lat"] == self.ref_lat

    def test_read_stack_point_reference_missing_coords(self, mock_stack):
        """Test point reference error if lon/lat are missing."""
        with pytest.raises(ValueError, match="ref_lon and ref_lat must be provided"):
            read_stack_lonlat(
                mock_stack,
                self.lon_slice,
                self.lat_slice,
                reference_method=ReferenceMethod.point,
            )

    def test_read_stack_median_reference(self, mock_stack):
        """Test reading stack with median referencing."""
        data, attrs = read_stack_lonlat(
            mock_stack,
            self.lon_slice,
            self.lat_slice,
            reference_method=ReferenceMethod.median,
        )
        # Verify data calculation
        unreffed_data = np.stack([d[self.expected_slice] for d in MOCK_HDF5_DATA_LIST])
        rebased_data = unreffed_data
        ref_values = np.nanmedian(rebased_data, axis=(1, 2), keepdims=True)
        expected_referenced_data = rebased_data - ref_values
        np.testing.assert_allclose(data, expected_referenced_data)

        # Check attributes
        assert attrs["reference_method"] == ReferenceMethod.median.value
        assert attrs["reference_lon"] == "None"
        assert attrs["reference_lat"] == "None"

    def test_read_stack_border_reference(self, mock_stack):
        """Test reading stack with border referencing."""
        data, attrs = read_stack_lonlat(
            mock_stack,
            self.lon_slice,
            self.lat_slice,
            reference_method=ReferenceMethod.border,
        )
        # Verify data calculation
        unreffed_data = np.stack([d[self.expected_slice] for d in MOCK_HDF5_DATA_LIST])
        rebased_data = unreffed_data  # Mock rebase passes through
        ref_values = _get_border(rebased_data)  # Calculate expected ref values
        expected_referenced_data = rebased_data - ref_values
        np.testing.assert_allclose(data, expected_referenced_data)

        # Check attributes
        assert attrs["reference_method"] == ReferenceMethod.border.value
        assert attrs["reference_lon"] == "None"
        assert attrs["reference_lat"] == "None"

    def test_read_stack_invalid_reference(self, mock_stack):
        """Test reading stack with an invalid reference method."""
        with pytest.raises(ValueError, match="Unknown reference_method"):
            read_stack_lonlat(
                mock_stack,
                self.lon_slice,
                self.lat_slice,
                reference_method="invalid_method",
            )
