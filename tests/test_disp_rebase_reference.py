from pathlib import Path

import pytest
import rasterio as rio

from opera_utils.disp.rebase_reference import QUALITY_DATASETS, main

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

    for ds_name in QUALITY_DATASETS:
        # Check for the number of geotiff files
        assert len(list(output_dir.glob(str(ds_name) + "*.tif"))) == len(input_files)

    # Check for the units on displacement
    assert all(
        rio.open(f).units[0] == "meters" for f in output_dir.glob("displacement*.tif")
    )


@pytest.mark.skipif(
    SKIP_TESTS, reason=f"No DISP-S1 input files found in {INPUT_DISP_S1_DIR}"
)
def test_rebase_reference_subsample(tmp_path):
    """Test subsample functionality with coarsening factor."""
    input_files = list(INPUT_DISP_S1_DIR.glob("*.nc"))
    output_dir = tmp_path / "aligned_subsample"

    # Run with subsample=2 (2x coarsening)
    main(nc_files=input_files, output_dir=output_dir, subsample=2)

    # Check that files were created
    displacement_files = list(output_dir.glob("displacement*.tif"))
    if displacement_files:
        # Read original and subsampled files to check dimensions
        with rio.open(f"NETCDF:{input_files[0]}:/displacement") as orig_src:
            orig_width, orig_height = orig_src.width, orig_src.height

        with rio.open(displacement_files[0]) as sub_src:
            sub_width, sub_height = sub_src.width, sub_src.height
            units = sub_src.units

        # Check that dimensions are approximately halved (allowing for rounding)
        assert sub_width == orig_width // 2
        assert sub_height == orig_height // 2

        # Check units are preserved
        assert units[0] == "meters"
