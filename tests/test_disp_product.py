import math
from dataclasses import is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from opera_utils.burst_frame_db import Bbox

# Assuming the classes are in opera_utils.disp
from opera_utils.disp import DispProduct, DispProductStack
from opera_utils.disp._product import OutOfBoundsError

# Create more filenames for stack testing
FILE_1 = "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v1.0_20250318T222753Z.nc"
FILE_2 = "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160810T140756Z_v1.0_20250318T222753Z.nc"  # Same ref, diff sec
FILE_3 = "OPERA_L3_DISP-S1_IW_F11116_VV_20160717T140755Z_20160729T140756Z_v1.0_20250318T222753Z.nc"  # Diff ref, same sec
FILE_4 = "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v1.1_20250319T222753Z.nc"  # Same dates, diff version
FILE_DIFF_FRAME = "OPERA_L3_DISP-S1_IW_F11117_VV_20160705T140755Z_20160729T140756Z_v1.0_20250318T222753Z.nc"

VALID_FILES = [FILE_3, FILE_1, FILE_2]  # Intentionally unsorted

# Expected sorted order based on (ref_date, sec_date)
EXPECTED_SORTED_FILENAMES = [FILE_1, FILE_2, FILE_3]
EXPECTED_REF_DATES = [
    datetime(2016, 7, 5, 14, 7, 55, tzinfo=timezone.utc),
    datetime(2016, 7, 5, 14, 7, 55, tzinfo=timezone.utc),
    datetime(2016, 7, 17, 14, 7, 55, tzinfo=timezone.utc),
]
EXPECTED_SEC_DATES = [
    datetime(2016, 7, 29, 14, 7, 56, tzinfo=timezone.utc),
    datetime(2016, 8, 10, 14, 7, 56, tzinfo=timezone.utc),
    datetime(2016, 7, 29, 14, 7, 56, tzinfo=timezone.utc),
]
EXPECTED_IFG_PAIRS = list(zip(EXPECTED_REF_DATES, EXPECTED_SEC_DATES))

# Sample return data for mocked functions
FRAME_ID = 11116
EPSG = 32610
BOUNDS = Bbox(left=499800.0, bottom=3907560.0, right=788340.0, top=4114500.0)
# Mock external functions for property tests
# This avoids the need to download the dataset
MOCK_FRAME_BBOX_RESULT = (EPSG, BOUNDS)
MOCK_X_COORDS = np.arange(499815.0, 788325.0 + 30, 30)
MOCK_Y_COORDS = np.arange(4114485.0, 3907575.0 - 30, -30)

# Expected shape from BOUNDS
EXPECTED_HEIGHT = round((BOUNDS.top - BOUNDS.bottom) / 30)
EXPECTED_WIDTH = int(round(BOUNDS.right - BOUNDS.left) / 30)
EXPECTED_SHAPE = (EXPECTED_HEIGHT, EXPECTED_WIDTH)


class TestDispProduct:
    def test_dataclass_definition(self):
        """Check if DispProduct is a dataclass."""
        assert is_dataclass(DispProduct)

    def test_from_filename_valid_str(self):
        """Test parsing a valid filename string."""
        product = DispProduct.from_filename(FILE_1)
        assert product.sensor == "S1"
        assert product.acquisition_mode == "IW"
        assert product.frame_id == FRAME_ID
        assert product.polarization == "VV"
        assert product.reference_datetime == datetime(
            2016, 7, 5, 14, 7, 55, tzinfo=timezone.utc
        )
        assert product.secondary_datetime == datetime(
            2016, 7, 29, 14, 7, 56, tzinfo=timezone.utc
        )
        assert product.version == "1.0"
        assert product.generation_datetime == datetime(
            2025, 3, 18, 22, 27, 53, tzinfo=timezone.utc
        )

    def test_from_filename_valid_path(self):
        """Test parsing a valid filename Path object."""
        product = DispProduct.from_filename(Path(FILE_1))
        assert product.frame_id == FRAME_ID  # Check one attribute is enough

    def test_from_filename_invalid_format(self):
        """Test parsing an invalid filename."""
        invalid_filename = "invalid_filename_format.txt"
        with pytest.raises(ValueError, match="Invalid filename format"):
            DispProduct.from_filename(invalid_filename)

    @pytest.fixture(autouse=True)
    def mock_get_frame_bbox(self, monkeypatch):
        monkeypatch.setattr(
            "opera_utils.burst_frame_db.get_frame_bbox",
            lambda frame_id: MOCK_FRAME_BBOX_RESULT,  # noqa: ARG005
        )

    def test_epsg_property(self):
        """Test the epsg property."""
        product = DispProduct.from_filename(FILE_1)
        assert product.epsg == EPSG

    def test_shape_property(self):
        """Test the shape property."""
        product = DispProduct.from_filename(FILE_1)
        assert product.shape == EXPECTED_SHAPE
        assert isinstance(product.shape[0], int)
        assert isinstance(product.shape[1], int)

    def test_xy_coords(self):
        """Test the x, y coordinate property."""
        product = DispProduct.from_filename(FILE_1)
        np.testing.assert_array_equal(product.x, MOCK_X_COORDS)
        np.testing.assert_array_equal(product.y, MOCK_Y_COORDS)

    def test_get_rasterio_profile_default_chunks(self):
        """Test get_rasterio_profile with default chunks."""
        product = DispProduct.from_filename(FILE_1)
        profile = product.get_rasterio_profile()

        assert profile["driver"] == "GTiff"
        assert profile["interleave"] == "band"
        assert profile["tiled"] is True
        assert profile["blockysize"] == 256
        assert profile["blockxsize"] == 256
        assert profile["compress"] == "lzw"
        assert math.isnan(profile["nodata"])
        assert profile["dtype"] == "float32"
        assert profile["count"] == 1
        assert profile["width"] == EXPECTED_WIDTH
        assert profile["height"] == EXPECTED_HEIGHT
        assert profile["crs"] == f"EPSG:{EPSG}"

        # Check transform calculation
        from rasterio import transform

        expected_transform = transform.from_bounds(
            BOUNDS.left,
            BOUNDS.bottom,
            BOUNDS.right,
            BOUNDS.top,
            EXPECTED_WIDTH,
            EXPECTED_HEIGHT,
        )
        assert profile["transform"] == expected_transform

    def test_get_rasterio_profile_custom_chunks(self):
        """Test get_rasterio_profile with custom chunks."""
        product = DispProduct.from_filename(FILE_1)
        custom_chunks = (512, 128)
        profile = product.get_rasterio_profile(chunks=custom_chunks)

        assert profile["blockysize"] == custom_chunks[0]
        assert profile["blockxsize"] == custom_chunks[1]
        # Check other essential keys are still present
        assert profile["width"] == EXPECTED_WIDTH
        assert profile["height"] == EXPECTED_HEIGHT
        assert profile["crs"] == f"EPSG:{EPSG}"

    def test_lonlat_to_rowcol(self):
        product = DispProduct.from_filename(FILE_1)
        assert product.lonlat_to_rowcol(-122.0, 36.0) == (4336, 3011)
        with pytest.raises(OutOfBoundsError):
            product.lonlat_to_rowcol(-122.0, 30.0)
        with pytest.raises(OutOfBoundsError):
            product.lonlat_to_rowcol(-102.0, 36.0)


class TestDispProductStack:
    def test_dataclass_definition(self):
        """Check if DispProductStack is a dataclass."""
        # It uses __post_init__, so it should be a dataclass
        assert is_dataclass(DispProductStack)

    def test_from_file_list_valid_sorting(self):
        """Test creating a stack from a list of files and check sorting."""
        stack = DispProductStack.from_file_list(VALID_FILES)
        assert len(stack.products) == len(VALID_FILES)
        # Check if the products are sorted correctly by date pairs
        assert stack.ifg_date_pairs == EXPECTED_IFG_PAIRS
        # Verify filenames correspond to the sorted order
        assert stack.products[0].reference_datetime == EXPECTED_REF_DATES[0]
        assert stack.products[0].secondary_datetime == EXPECTED_SEC_DATES[0]
        assert stack.products[1].reference_datetime == EXPECTED_REF_DATES[1]
        assert stack.products[1].secondary_datetime == EXPECTED_SEC_DATES[1]
        assert stack.products[2].reference_datetime == EXPECTED_REF_DATES[2]
        assert stack.products[2].secondary_datetime == EXPECTED_SEC_DATES[2]

    def test_from_file_list_empty(self):
        """Test creating a stack from an empty list."""
        with pytest.raises(ValueError, match="At least one product is required"):
            DispProductStack.from_file_list([])

    def test_init_different_frame_ids_error(self):
        """Test ValueError on init if frame_ids differ."""
        prod1 = DispProduct.from_filename(FILE_1)
        prod_diff_frame = DispProduct.from_filename(FILE_DIFF_FRAME)
        with pytest.raises(
            ValueError, match="All products must have the same frame_id"
        ):
            DispProductStack(products=[prod1, prod_diff_frame])

    def test_init_duplicate_date_pairs_error(self):
        """Test ValueError on init if date pairs are duplicated."""
        prod1 = DispProduct.from_filename(FILE_1)
        prod4_dup = DispProduct.from_filename(FILE_4)  # Same dates, diff version
        with pytest.raises(
            ValueError,
            match="All products must have unique reference and secondary dates",
        ):
            DispProductStack(products=[prod1, prod4_dup])

        with pytest.raises(
            ValueError,
            match="All products must have unique reference and secondary dates",
        ):
            DispProductStack.from_file_list([FILE_1, FILE_4])

    def test_stack_properties(self):
        """Test properties that delegate to the first product."""
        stack = DispProductStack.from_file_list(VALID_FILES)

        # Basic properties
        assert stack.frame_id == FRAME_ID
        assert stack.reference_dates == EXPECTED_REF_DATES
        assert stack.secondary_dates == EXPECTED_SEC_DATES
        assert stack.ifg_date_pairs == EXPECTED_IFG_PAIRS

        # Properties requiring mocked external calls (delegated to product[0])
        assert stack.epsg == EPSG
        # Check shape includes the number of products
        assert stack.shape == (len(VALID_FILES), EXPECTED_HEIGHT, EXPECTED_WIDTH)
        np.testing.assert_array_equal(stack.x, MOCK_X_COORDS)
        np.testing.assert_array_equal(stack.y, MOCK_Y_COORDS)

    def test_getitem(self):
        """Test accessing products by index."""
        stack = DispProductStack.from_file_list(VALID_FILES)
        # Check first item corresponds to the first in sorted list
        assert isinstance(stack[0], DispProduct)
        assert stack[0] == DispProduct.from_filename(FILE_1)

    def test_getitem_stack(self):
        """Test accessing products by a slice, to create a new DispProductStack."""
        stack = DispProductStack.from_file_list(VALID_FILES)
        assert stack[:2] == DispProductStack.from_file_list(sorted(VALID_FILES)[:2])

    def test_get_rasterio_profile_delegation(self):
        """Test that stack's profile matches the first product's profile."""
        stack = DispProductStack.from_file_list(VALID_FILES)
        assert (
            stack.get_rasterio_profile()
            == DispProduct.from_filename(VALID_FILES[0]).get_rasterio_profile()
        )

    def test_to_dataframe(self):
        """Test that to_dataframe returns a DataFrame with the correct data."""
        import pandas as pd

        stack = DispProductStack.from_file_list(VALID_FILES)
        df = stack.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(VALID_FILES)
        assert all(df["frame_id"] == FRAME_ID)
        assert all(df["reference_datetime"] == EXPECTED_REF_DATES)
        assert all(df["secondary_datetime"] == EXPECTED_SEC_DATES)
        assert all(df["version"] == "1.0")
        assert all(
            df["generation_datetime"]
            == datetime(2025, 3, 18, 22, 27, 53, tzinfo=timezone.utc)
        )
        assert all(df["polarization"] == "VV")
        assert all(df["sensor"] == "S1")
        assert all(df["acquisition_mode"] == "IW")
        assert "size_in_bytes" in df.columns
