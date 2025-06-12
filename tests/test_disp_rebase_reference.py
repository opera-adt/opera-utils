from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from opera_utils.disp._enums import (
    SAME_PER_MINISTACK_DATASETS,
    UNIQUE_PER_DATE_DATASETS,
)
from opera_utils.disp.rebase_reference import main

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
def test_rebase_reference(tmp_path):
    input_files = list(INPUT_DISP_S1_DIR.glob("*.nc"))
    output_dir = tmp_path / "aligned"

    main(nc_files=input_files, output_dir=output_dir)

    in_ds = xr.open_mfdataset(input_files, engine="h5netcdf")
    for ds_name in UNIQUE_PER_DATE_DATASETS:
        # Check for the number of geotiff files
        assert len(list(output_dir.glob(str(ds_name) + "*.tif"))) == len(input_files)

    num_unique_ref_dates = np.unique(in_ds.reference_time).size
    for ds_name in SAME_PER_MINISTACK_DATASETS:
        # Check for the number of geotiff files
        assert (
            len(list(output_dir.glob(str(ds_name) + "*.tif"))) == num_unique_ref_dates
        )
