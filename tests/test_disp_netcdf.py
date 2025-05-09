import tempfile
from datetime import datetime, timezone

import h5netcdf
import numpy as np
import pytest

from opera_utils.disp import DispProductStack
from opera_utils.disp._netcdf import (
    _create_time_array,
    save_data,
)

FILE_1 = "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v1.0_20250318T222753Z.nc"
FILE_2 = "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160810T140756Z_v1.0_20250318T222753Z.nc"
FILE_3 = "OPERA_L3_DISP-S1_IW_F11116_VV_20160717T140755Z_20160729T140756Z_v1.0_20250318T222753Z.nc"
FILE_LIST = [FILE_1, FILE_2, FILE_3]


@pytest.fixture
def test_stack():
    return DispProductStack.from_file_list(FILE_LIST)


@pytest.fixture
def test_data():
    # Create sample 3D array [time, height, width]
    return np.random.randn(len(FILE_LIST), 10, 15).astype(np.float32)


def test_create_time_array():
    # Test _create_time_array with a sequence of datetime objects
    dates = [
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 15, tzinfo=timezone.utc),
    ]
    time_array, calendar, units = _create_time_array(dates)

    # Check time array is correctly calculated
    assert len(time_array) == 2
    assert (
        time_array[1] - time_array[0]
    ) / 86400 == 14  # 14 days difference in seconds
    assert calendar == "standard"
    assert "seconds since 2010-01-01" in units


def test_save_data(test_stack, test_data):
    # Test save_data function end-to-end
    with tempfile.NamedTemporaryFile(suffix=".nc") as temp_file:
        # Create subset slices
        rows = slice(0, 10)
        cols = slice(0, 15)

        # Save data
        save_data(
            data=test_data,
            product_stack=test_stack,
            output=temp_file.name,
            dataset_name="displacement",
            rows=rows,
            cols=cols,
            attrs={
                "units": "meters",
                "long_name": "Displacement",
                "description": "Test displacement data",
            },
        )

        # Verify file was created and is readable
        with h5netcdf.File(temp_file.name, "r") as f:
            # Check dataset exists and has correct shape
            assert "displacement" in f
            assert f["displacement"].shape == test_data.shape
            assert "x" in f
            assert "y" in f
            assert "time" in f
            assert "spatial_ref" in f

            # Check attributes were set
            assert f["displacement"].attrs["units"] == "meters"
            assert f["displacement"].attrs["long_name"] == "Displacement"
            assert f["displacement"].attrs["description"] == "Test displacement data"

            # Verify coordinates are correct
            assert len(f["x"]) == 15
            assert len(f["y"]) == 10
            assert len(f["time"]) == len(test_stack.products)

            # Check data was written correctly
            np.testing.assert_array_equal(f["displacement"][:], test_data)

            # Check grid mapping attributes were set
            dset = f["spatial_ref"]
            assert "GeoTransform" in dset.attrs
            assert "crs_wkt" in dset.attrs or "spatial_ref" in dset.attrs
            assert dset.attrs["units"] == "unitless"

            # Verify the transform was saved correctly
            assert dset.attrs["GeoTransform"] == "499800.0 30.0 0.0 4114500.0 0.0 -30.0"
