from pathlib import Path

import pytest
import xarray as xr

from opera_utils.disp import reformat_stack
from opera_utils.disp._enums import (
    SAME_PER_MINISTACK_DATASETS,
    UNIQUE_PER_DATE_DATASETS,
)
from opera_utils.disp._reformat import combine_quality_masks

# UserWarning: Consolidated metadata is currently not part in the Zarr format 3 specification.
# zarr/api/asynchronous.py:203: UserWarning: Consolidated metadata is currently not
# part in the Zarr format 3 specification.

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*Consolidated metadata is currently not part in the Zarr format 3"
    " specification.*:UserWarning",
)


INPUT_DISP_S1_DIR = Path(__file__).parent / "data" / "subsets-new-orleans-small"
SKIP_TESTS = len(list(INPUT_DISP_S1_DIR.glob("*.nc"))) == 0


@pytest.mark.skipif(
    SKIP_TESTS, reason=f"No DISP-S1 input files found in {INPUT_DISP_S1_DIR}"
)
def test_reformat_stack_zarr(tmp_path):
    input_files = list(INPUT_DISP_S1_DIR.glob("*.nc"))
    output_name = tmp_path / "test.zarr"

    reformat_stack(input_files=input_files, output_name=output_name)

    # Inspect results
    ds = xr.open_zarr(output_name, consolidated=False)
    for ds_name in UNIQUE_PER_DATE_DATASETS + SAME_PER_MINISTACK_DATASETS:
        assert ds_name in ds.data_vars


@pytest.mark.skipif(
    SKIP_TESTS, reason=f"No DISP-S1 input files found in {INPUT_DISP_S1_DIR}"
)
def test_reformat_stack_netcdf(tmp_path):
    input_files = list(INPUT_DISP_S1_DIR.glob("*.nc"))
    output_name = tmp_path / "test.nc"

    reformat_stack(input_files=input_files, output_name=output_name)

    # Inspect results
    ds = xr.open_dataset(output_name, engine="h5netcdf")
    for ds_name in UNIQUE_PER_DATE_DATASETS + SAME_PER_MINISTACK_DATASETS:
        assert ds_name in ds.data_vars


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
    import numpy as np

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
