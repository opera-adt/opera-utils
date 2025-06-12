from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from opera_utils.disp._rebase import combine_quality_masks, rebase_timeseries


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


def test_combine_quality_masks_logical_or():
    """Test combine_quality_masks with default logical_or reduction."""
    # Create test quality datasets
    quality1 = xr.DataArray(
        [[0.3, 0.7], [0.2, 0.8]], dims=["y", "x"], coords={"y": [0, 1], "x": [0, 1]}
    )
    quality2 = xr.DataArray(
        [[0.6, 0.4], [0.9, 0.1]], dims=["y", "x"], coords={"y": [0, 1], "x": [0, 1]}
    )

    quality_datasets = [quality1, quality2]
    thresholds = [0.5, 0.5]

    result = combine_quality_masks(quality_datasets, thresholds)

    # Expected: True where any dataset > threshold
    # quality1 > 0.5: [[False, True], [False, True]]
    # quality2 > 0.5: [[True, False], [True, False]]
    # logical_or:     [[True, True], [True, True]]
    expected = xr.DataArray(
        [[True, True], [True, True]], dims=["y", "x"], coords={"y": [0, 1], "x": [0, 1]}
    )

    xr.testing.assert_equal(result, expected)


def test_combine_quality_masks_logical_and():
    """Test combine_quality_masks with logical_and reduction."""
    # Create test quality datasets
    quality1 = xr.DataArray(
        [[0.3, 0.7], [0.6, 0.8]], dims=["y", "x"], coords={"y": [0, 1], "x": [0, 1]}
    )
    quality2 = xr.DataArray(
        [[0.6, 0.4], [0.7, 0.9]], dims=["y", "x"], coords={"y": [0, 1], "x": [0, 1]}
    )

    quality_datasets = [quality1, quality2]
    thresholds = [0.5, 0.5]

    result = combine_quality_masks(
        quality_datasets, thresholds, reduction_func=np.logical_and
    )

    # Expected: True where all datasets > threshold
    # quality1 > 0.5: [[False, True], [True, True]]
    # quality2 > 0.5: [[True, False], [True, True]]
    # logical_and:    [[False, False], [True, True]]
    expected = xr.DataArray(
        [[False, False], [True, True]],
        dims=["y", "x"],
        coords={"y": [0, 1], "x": [0, 1]},
    )

    xr.testing.assert_equal(result, expected)


def test_combine_quality_masks_different_thresholds():
    """Test combine_quality_masks with different thresholds for each dataset."""
    # Create test quality datasets
    quality1 = xr.DataArray(
        [[0.2, 0.4], [0.6, 0.8]], dims=["y", "x"], coords={"y": [0, 1], "x": [0, 1]}
    )
    quality2 = xr.DataArray(
        [[0.5, 0.7], [0.3, 0.9]], dims=["y", "x"], coords={"y": [0, 1], "x": [0, 1]}
    )

    quality_datasets = [quality1, quality2]
    thresholds = [0.3, 0.6]  # Different thresholds

    result = combine_quality_masks(quality_datasets, thresholds)

    # Expected: True where any dataset > its threshold
    # quality1 > 0.3: [[False, True], [True, True]]
    # quality2 > 0.6: [[False, True], [False, True]]
    # logical_or:     [[False, True], [True, True]]
    expected = xr.DataArray(
        [[False, True], [True, True]],
        dims=["y", "x"],
        coords={"y": [0, 1], "x": [0, 1]},
    )

    xr.testing.assert_equal(result, expected)


def test_combine_quality_masks_single_dataset():
    """Test combine_quality_masks with a single quality dataset."""
    quality1 = xr.DataArray(
        [[0.3, 0.7], [0.2, 0.8]], dims=["y", "x"], coords={"y": [0, 1], "x": [0, 1]}
    )

    quality_datasets = [quality1]
    thresholds = [0.5]

    result = combine_quality_masks(quality_datasets, thresholds)

    # Expected: True where dataset > threshold
    expected = xr.DataArray(
        [[False, True], [False, True]],
        dims=["y", "x"],
        coords={"y": [0, 1], "x": [0, 1]},
    )

    xr.testing.assert_equal(result, expected)


def test_combine_quality_masks_three_datasets():
    """Test combine_quality_masks with three quality datasets."""
    quality1 = xr.DataArray(
        [[0.3, 0.7]], dims=["y", "x"], coords={"y": [0], "x": [0, 1]}
    )
    quality2 = xr.DataArray(
        [[0.4, 0.2]], dims=["y", "x"], coords={"y": [0], "x": [0, 1]}
    )
    quality3 = xr.DataArray(
        [[0.8, 0.1]], dims=["y", "x"], coords={"y": [0], "x": [0, 1]}
    )

    quality_datasets = [quality1, quality2, quality3]
    thresholds = [0.5, 0.5, 0.5]

    result = combine_quality_masks(quality_datasets, thresholds)

    # Expected: True where any dataset > threshold
    # quality1 > 0.5: [[False, True]]
    # quality2 > 0.5: [[False, False]]
    # quality3 > 0.5: [[True, False]]
    # logical_or:     [[True, True]]
    expected = xr.DataArray(
        [[True, True]], dims=["y", "x"], coords={"y": [0], "x": [0, 1]}
    )

    xr.testing.assert_equal(result, expected)


def test_combine_quality_masks_mismatched_lengths():
    """Test that combine_quality_masks raises error for mismatched lengths."""
    quality1 = xr.DataArray(
        [[0.3, 0.7]], dims=["y", "x"], coords={"y": [0], "x": [0, 1]}
    )
    quality2 = xr.DataArray(
        [[0.4, 0.2]], dims=["y", "x"], coords={"y": [0], "x": [0, 1]}
    )

    quality_datasets = [quality1, quality2]
    thresholds = [0.5]  # Wrong length

    with pytest.raises(ValueError, match="argument 2 is shorter than argument 1"):
        combine_quality_masks(quality_datasets, thresholds)
