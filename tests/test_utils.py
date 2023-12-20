from pathlib import Path

import pytest

from opera_utils._utils import format_nc_filename, get_snwe, scratch_directory


def test_scratch_directory(tmp_path):
    test_dir = tmp_path / "test_dir"
    with scratch_directory(test_dir) as d:
        (d / "test1.txt").write_text("asf")
    assert not Path(test_dir).exists()

    with scratch_directory(test_dir, delete=False) as d:
        (d / "test1.txt").write_text("asdf")
    assert Path(test_dir).exists()
    assert Path(test_dir / "test1.txt").read_text() == "asdf"


def test_scratch_directory_already_exists(tmp_path):
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    assert Path(test_dir).exists()

    with scratch_directory(test_dir, delete=True) as d:
        (d / "test1.txt").write_text("asdf")
    assert Path(test_dir).exists()

    with scratch_directory(test_dir, delete=False) as d:
        (d / "test1.txt").write_text("asdf")
    assert Path(test_dir).exists()


def test_format_nc_filename():
    expected = 'NETCDF:"/usr/19990101/20200303_20210101.nc":"//variable"'
    assert (
        format_nc_filename("/usr/19990101/20200303_20210101.nc", "variable") == expected
    )

    # check on Path
    assert (
        format_nc_filename(Path("/usr/19990101/20200303_20210101.nc"), "variable")
        == expected
    )

    # check non-netcdf file
    assert (
        format_nc_filename("/usr/19990101/20200303_20210101.tif")
        == "/usr/19990101/20200303_20210101.tif"
    )
    assert (
        format_nc_filename("/usr/19990101/20200303_20210101.int", "ignored")
        == "/usr/19990101/20200303_20210101.int"
    )

    with pytest.raises(ValueError):
        # Missing the subdataset name
        format_nc_filename("/usr/19990101/20200303_20210101.nc")


def test_get_snwe():
    bounds = (-110, 32, -109, 33)
    assert get_snwe(4326, bounds) == (32, 33, -110, -109)
