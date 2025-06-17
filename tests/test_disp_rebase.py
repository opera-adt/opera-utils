from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from opera_utils.disp._rebase import (
    NaNPolicy,
    create_rebased_displacement,
    rebase_timeseries,
)


def test_rebase_single_reference():
    """Test identity when there's a single reference date."""
    # Create test data with same reference date
    raw_data = np.ones((3, 2, 2), dtype=np.float32)
    # Set different values for each time step
    raw_data[0] *= 1
    raw_data[1] *= 2
    raw_data[2] *= 3

    reference_dates = [
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc),
    ]

    # Since there's only one reference date, the output should be the same as input
    result = rebase_timeseries(raw_data, reference_dates)

    # Check output shape and values
    assert result.shape == raw_data.shape
    assert np.array_equal(result, raw_data)
    assert id(result) != id(raw_data)  # Should be a copy, not the same object


def test_rebase_multiple_references():
    """Test linear ramp with multiple reference dates."""
    reference_dates = [
        datetime(2020, 1, 1, tzinfo=timezone.utc),  # Ref date 1
        datetime(2020, 1, 1, tzinfo=timezone.utc),  # Ref date 1 (same)
        datetime(2020, 2, 1, tzinfo=timezone.utc),  # Ref date 2
        datetime(2020, 2, 1, tzinfo=timezone.utc),  # Ref date 2 (same)
    ]

    # Create a 3D array with shape (4, 2, 2)
    raw_data = np.zeros((4, 2, 2), dtype=np.float32)

    # Fill raw_data with test values: each pixel goes [1,2,3,4]
    # First two acquisitions use ref date 1
    raw_data[0, :, :] = 1.0  # Reference date 1 to secondary date 1
    raw_data[1, :, :] = 2.0  # Reference date 1 to secondary date 2
    # Next two acquisitions use ref date 2 (which is the same as secondary date 2)
    raw_data[2, :, :] = 1.0  # Reference date 2 to secondary date 3
    raw_data[3, :, :] = 2.0  # Reference date 2 to secondary date 4

    # Call the function we're testing
    result = rebase_timeseries(raw_data, reference_dates)

    # Expected result should be:
    # Time 0: 1.0 (unchanged)
    # Time 1: 2.0 (unchanged)
    # Time 1 is now the reference crossover point
    # Time 2: 2.0 + 1.0 = 3.0 (added offset)
    # Time 3: 2.0 + 2.0 = 4.0 (added offset)

    expected = np.zeros_like(raw_data)
    expected[0, :, :] = 1.0
    expected[1, :, :] = 2.0
    expected[2, :, :] = 3.0
    expected[3, :, :] = 4.0

    np.testing.assert_array_equal(result, expected)


def test_rebase_with_three_references():
    """Test with three reference dates to ensure multiple shifts work correctly."""
    # Expected result with accumulated offsets
    shape = (5, 2, 2)
    expected = np.zeros(shape, dtype=np.float32)
    expected[0, :, :] = 1.0
    expected[1, :, :] = 2.0
    expected[2, :, :] = 3.0
    expected[3, :, :] = 4.0
    expected[4, :, :] = 5.0

    raw_data = np.zeros(shape, dtype=np.float32)
    # First two acquisitions use ref date 1
    raw_data[0, :, :] = 1.0  # Ref1 -> Sec1
    raw_data[1, :, :] = 2.0  # Ref1 -> Sec2
    # Next two use ref date 2
    raw_data[2, :, :] = 1.0  # Ref2 -> Sec3
    raw_data[3, :, :] = 2.0  # Ref2 -> Sec4
    # Last uses ref date 3
    raw_data[4, :, :] = 1.0  # Ref3 -> Sec5

    # Define reference dates with two shifts
    reference_dates = [
        datetime(2020, 1, 1, tzinfo=timezone.utc),  # Ref date 1
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 2, 1, tzinfo=timezone.utc),  # Ref date 2
        datetime(2020, 2, 1, tzinfo=timezone.utc),
        datetime(2020, 3, 1, tzinfo=timezone.utc),  # Ref date 3
    ]

    result = rebase_timeseries(raw_data, reference_dates)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("nan_policy", ["propagate", "omit"])
def test_rebase_with_nan_values(nan_policy: str):
    """Test with NaN values to ensure they're handled correctly."""
    # Create 3D array with some NaN values at one pixel
    raw_data = np.ones((3, 2, 2), dtype=np.float32)
    raw_data[1, :, :] *= 2  # Second time step values = 2
    raw_data[
        2, :, :
    ] *= 4  # Third time step values = 4 (6 in the single-reference output)

    raw_data[0, 0, 0] = np.nan  # Add NaN at first time step: recoverable
    raw_data[1, 1, 0] = np.nan  # Add NaN at second time step: propagates
    # Different reference dates for each time
    reference_dates = [
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 3, 1, tzinfo=timezone.utc),  # Reference date changes here
    ]

    result = rebase_timeseries(raw_data, reference_dates, nan_policy=nan_policy)

    # Second column had no nans, should be all fine:
    np.testing.assert_array_equal(result[:, 0, 1], np.array([1, 2, 6]))
    np.testing.assert_array_equal(result[:, 1, 1], np.array([1, 2, 6]))

    # Top left pixel had a nan *only* during the first time step, not during
    # a reference change
    np.testing.assert_array_equal(result[:, 0, 0], np.array([np.nan, 2, 6]))

    # Step 2 (same reference date as step 1) has the same cumulative offset
    if nan_policy == "omit":
        # We assumed here was zero deformation, so we add the offset to the previous value
        np.testing.assert_array_equal(result[:, 1, 0], np.array([1, np.nan, 4]))
    else:
        np.testing.assert_array_equal(result[:, 1, 0], np.array([1, np.nan, np.nan]))


class TestCreateRebasedDisplacement:
    """Test suite for create_rebased_displacement function."""

    def test_create_rebased_displacement_single_reference(self):
        """Test create_rebased_displacement with single reference date."""
        # Create test displacement DataArray
        np.random.seed(42)
        data = np.random.randn(3, 4, 5).astype(np.float32)
        times = [datetime(2020, 1, i + 1, tzinfo=timezone.utc) for i in range(3)]
        da = xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": times, "y": np.arange(4), "x": np.arange(5)},
        )

        # All same reference date
        reference_dates = [datetime(2020, 1, 1, tzinfo=timezone.utc)] * 3

        result = create_rebased_displacement(da, reference_dates)

        # Should be unchanged when reference date is constant
        xr.testing.assert_allclose(result, da)
        assert result.dims == da.dims
        assert result.shape == da.shape

    def test_create_rebased_displacement_multiple_references(self):
        """Test create_rebased_displacement with multiple reference dates."""
        # Create test data with known pattern
        data = np.zeros((4, 2, 2), dtype=np.float32)
        data[0] = 1.0  # First image: 1.0 everywhere
        data[1] = 2.0  # Second image: 2.0 everywhere
        data[2] = 1.0  # Third image: 1.0 everywhere (new reference)
        data[3] = 2.0  # Fourth image: 2.0 everywhere

        times = [datetime(2020, 1, i + 1, tzinfo=timezone.utc) for i in range(4)]
        da = xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": times, "y": np.arange(2), "x": np.arange(2)},
        )

        # Reference dates: first two same, last two same but different
        reference_dates = [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 3, tzinfo=timezone.utc),  # Reference changes
            datetime(2020, 1, 3, tzinfo=timezone.utc),
        ]

        result = create_rebased_displacement(da, reference_dates)

        # Expected: [1, 2, 3, 4] (accumulated at reference change)
        expected_data = np.array([1.0, 2.0, 3.0, 4.0])

        np.testing.assert_array_equal(result.values[:, 0, 0], expected_data)
        np.testing.assert_array_equal(result.values[:, 0, 1], expected_data)
        np.testing.assert_array_equal(result.values[:, 1, 0], expected_data)
        np.testing.assert_array_equal(result.values[:, 1, 1], expected_data)

    def test_create_rebased_displacement_with_add_reference_time(self):
        """Test create_rebased_displacement with add_reference_time=True."""
        data = np.ones((2, 2, 2), dtype=np.float32)
        times = [datetime(2020, 1, i + 1, tzinfo=timezone.utc) for i in range(2)]
        da = xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": times, "y": np.arange(2), "x": np.arange(2)},
        )

        reference_dates = [datetime(2020, 1, 1, tzinfo=timezone.utc)] * 2

        result = create_rebased_displacement(
            da, reference_dates, add_reference_time=True
        )

        # Should have one extra time dimension at the beginning with zeros
        assert result.shape == (3, 2, 2)
        np.testing.assert_array_equal(result.values[0], np.zeros((2, 2)))
        np.testing.assert_array_equal(result.values[1:], data)

    @pytest.mark.parametrize("nan_policy", [NaNPolicy.propagate, NaNPolicy.omit])
    def test_create_rebased_displacement_with_nans(self, nan_policy):
        """Test create_rebased_displacement with NaN values."""
        data = np.ones((3, 2, 2), dtype=np.float32)
        data[1, 0, 0] = np.nan  # Add NaN at one location

        times = [datetime(2020, 1, i + 1, tzinfo=timezone.utc) for i in range(3)]
        da = xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": times, "y": np.arange(2), "x": np.arange(2)},
        )

        # Reference date changes at third time step
        reference_dates = [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 3, tzinfo=timezone.utc),  # Reference changes
        ]

        result = create_rebased_displacement(da, reference_dates, nan_policy=nan_policy)

        # Check that NaN handling follows the policy
        if nan_policy == NaNPolicy.propagate:
            # NaN should propagate through the reference change
            assert np.isnan(result.values[2, 0, 0])
        else:  # omit
            # NaN should be treated as zero, so result should be finite
            assert np.isfinite(result.values[2, 0, 0])

        # Other locations should not be affected
        assert np.isfinite(result.values[:, 0, 1]).all()
        assert np.isfinite(result.values[:, 1, :]).all()

    def test_create_rebased_displacement_custom_chunk_size(self):
        """Test create_rebased_displacement with custom chunk size."""
        data = np.ones((2, 4, 4), dtype=np.float32)
        times = [datetime(2020, 1, i + 1, tzinfo=timezone.utc) for i in range(2)]
        da = xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": times, "y": np.arange(4), "x": np.arange(4)},
        )

        reference_dates = [datetime(2020, 1, 1, tzinfo=timezone.utc)] * 2

        result = create_rebased_displacement(
            da, reference_dates, process_chunk_size=(2, 2)
        )

        # Should work the same regardless of chunk size
        xr.testing.assert_allclose(result, da)

    def test_create_rebased_displacement_preserves_coordinates(self):
        """Test that create_rebased_displacement preserves coordinates."""
        data = np.ones((2, 3, 4), dtype=np.float32)
        times = [datetime(2020, 1, i + 1, tzinfo=timezone.utc) for i in range(2)]
        y_coords = np.array([10.0, 20.0, 30.0])
        x_coords = np.array([100.0, 200.0, 300.0, 400.0])

        da = xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": times, "y": y_coords, "x": x_coords},
        )

        reference_dates = [datetime(2020, 1, 1, tzinfo=timezone.utc)] * 2

        result = create_rebased_displacement(da, reference_dates)

        # Check that coordinates are preserved
        np.testing.assert_array_equal(result.coords["time"], da.coords["time"])
        np.testing.assert_array_equal(result.coords["y"], da.coords["y"])
        np.testing.assert_array_equal(result.coords["x"], da.coords["x"])

    def test_create_rebased_displacement_dimension_order(self):
        """Test that create_rebased_displacement maintains proper dimension order."""
        data = np.ones((2, 3, 4), dtype=np.float32)
        times = [datetime(2020, 1, i + 1, tzinfo=timezone.utc) for i in range(2)]
        da = xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": times, "y": np.arange(3), "x": np.arange(4)},
        )

        reference_dates = [datetime(2020, 1, 1, tzinfo=timezone.utc)] * 2

        result = create_rebased_displacement(
            da, reference_dates, add_reference_time=True
        )

        # Check dimension order is maintained as (time, y, x)
        assert result.dims == ("time", "y", "x")
        assert result.shape == (3, 3, 4)  # +1 time dimension
