from pathlib import Path

import pytest
import xarray as xr

from opera_utils.disp import reformat_stack
from opera_utils.disp._enums import (
    SAME_PER_MINISTACK_DATASETS,
    UNIQUE_PER_DATE_DATASETS,
)

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
    ds = xr.open_zarr(output_name)
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
