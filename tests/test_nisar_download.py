"""Tests for opera_utils.nisar._download module."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from opera_utils.nisar._download import (
    _extract_subset_from_h5,
    process_file,
    run_download,
)
from opera_utils.nisar._product import NISAR_GSLC_GRIDS, NISAR_POLARIZATIONS

FILE_1 = "NISAR_L2_PR_GSLC_004_076_A_022_2005_QPDH_A_20251103T110514_20251103T110549_X05007_N_F_J_001.h5"

NROWS = 50
NCOLS = 80


@pytest.fixture
def gslc_h5(tmp_path):
    """Create a minimal GSLC HDF5 file with known data."""
    h5path = tmp_path / FILE_1
    with h5py.File(h5path, "w") as f:
        freq_path = f"{NISAR_GSLC_GRIDS}/frequencyA"
        freq_group = f.create_group(freq_path)

        x_coords = np.arange(400000.0, 400000.0 + NCOLS * 10, 10)
        y_coords = np.arange(1510000.0, 1510000.0 - NROWS * 10, -10)
        freq_group.create_dataset("xCoordinates", data=x_coords)
        freq_group.create_dataset("yCoordinates", data=y_coords)
        freq_group.create_dataset("xCoordinateSpacing", data=10.0)
        freq_group.create_dataset("yCoordinateSpacing", data=-10.0)
        freq_group.create_dataset("projection", data=32637)

        rng = np.random.default_rng(42)
        data = (
            rng.standard_normal((NROWS, NCOLS))
            + 1j * rng.standard_normal((NROWS, NCOLS))
        ).astype(np.complex64)

        hh = freq_group.create_dataset("HH", data=data)
        hh.attrs["description"] = "HH polarization"
        freq_group.create_dataset("VV", data=data * 0.5)

    return h5path


class TestRunDownloadValidation:
    def test_bbox_and_rows_mutually_exclusive(self):
        with pytest.raises(ValueError, match="Cannot specify both bbox/wkt and rows"):
            run_download(bbox=(0, 0, 1, 1), rows=(0, 100))

    def test_bbox_and_cols_mutually_exclusive(self):
        with pytest.raises(ValueError, match="Cannot specify both bbox/wkt and rows"):
            run_download(bbox=(0, 0, 1, 1), cols=(0, 100))

    def test_wkt_and_rows_mutually_exclusive(self):
        with pytest.raises(ValueError, match="Cannot specify both bbox/wkt and rows"):
            run_download(wkt="POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))", rows=(0, 100))

    def test_bbox_and_wkt_mutually_exclusive(self):
        with pytest.raises(ValueError, match="Cannot specify both bbox and wkt"):
            run_download(bbox=(0, 0, 1, 1), wkt="POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")


class TestExtractSubsetFromH5:
    def test_full_extraction(self, gslc_h5, tmp_path):
        """Extract without subsetting (rows=None, cols=None)."""
        outpath = tmp_path / "output.h5"
        with h5py.File(gslc_h5, "r") as src:
            _extract_subset_from_h5(src, outpath, rows=None, cols=None)

        with h5py.File(outpath, "r") as dst:
            freq_path = f"{NISAR_GSLC_GRIDS}/frequencyA"
            # Both polarizations should be extracted
            assert "HH" in dst[freq_path]
            assert "VV" in dst[freq_path]
            assert dst[f"{freq_path}/HH"].shape == (NROWS, NCOLS)
            # Coordinates should be full length
            assert dst[f"{freq_path}/xCoordinates"].shape == (NCOLS,)
            assert dst[f"{freq_path}/yCoordinates"].shape == (NROWS,)
            # Projection should be copied
            assert int(dst[f"{freq_path}/projection"][()]) == 32637

    def test_spatial_subset(self, gslc_h5, tmp_path):
        """Extract a spatial subset."""
        outpath = tmp_path / "subset.h5"
        rows = slice(10, 30)
        cols = slice(20, 50)
        with h5py.File(gslc_h5, "r") as src:
            _extract_subset_from_h5(src, outpath, rows=rows, cols=cols)

        with h5py.File(outpath, "r") as dst:
            freq_path = f"{NISAR_GSLC_GRIDS}/frequencyA"
            assert dst[f"{freq_path}/HH"].shape == (20, 30)
            assert dst[f"{freq_path}/xCoordinates"].shape == (30,)
            assert dst[f"{freq_path}/yCoordinates"].shape == (20,)

    def test_polarization_filter(self, gslc_h5, tmp_path):
        """Extract only selected polarizations."""
        outpath = tmp_path / "hh_only.h5"
        with h5py.File(gslc_h5, "r") as src:
            _extract_subset_from_h5(
                src, outpath, rows=None, cols=None, polarizations=["HH"]
            )

        with h5py.File(outpath, "r") as dst:
            freq_path = f"{NISAR_GSLC_GRIDS}/frequencyA"
            assert "HH" in dst[freq_path]
            # VV should NOT be in the output
            pols_in_output = [k for k in dst[freq_path] if k in NISAR_POLARIZATIONS]
            assert pols_in_output == ["HH"]

    def test_attributes_copied(self, gslc_h5, tmp_path):
        """Dataset attributes should be preserved."""
        outpath = tmp_path / "attrs.h5"
        with h5py.File(gslc_h5, "r") as src:
            _extract_subset_from_h5(src, outpath, rows=None, cols=None)

        with h5py.File(outpath, "r") as dst:
            hh_path = f"{NISAR_GSLC_GRIDS}/frequencyA/HH"
            assert dst[hh_path].attrs["description"] == "HH polarization"

    def test_subset_metadata_stored(self, gslc_h5, tmp_path):
        """Output file should record subset metadata."""
        outpath = tmp_path / "meta.h5"
        with h5py.File(gslc_h5, "r") as src:
            _extract_subset_from_h5(src, outpath, rows=slice(5, 15), cols=slice(10, 40))

        with h5py.File(outpath, "r") as dst:
            assert "subset_rows" in dst.attrs
            assert "subset_cols" in dst.attrs
            assert "source_file" in dst.attrs

    def test_data_values_match_source(self, gslc_h5, tmp_path):
        """Subsetted data should exactly match the source slice."""
        outpath = tmp_path / "values.h5"
        rows = slice(5, 25)
        cols = slice(10, 60)
        with h5py.File(gslc_h5, "r") as src:
            expected = src[f"{NISAR_GSLC_GRIDS}/frequencyA/HH"][rows, cols]
            _extract_subset_from_h5(src, outpath, rows=rows, cols=cols)

        with h5py.File(outpath, "r") as dst:
            actual = dst[f"{NISAR_GSLC_GRIDS}/frequencyA/HH"][:]
            np.testing.assert_array_equal(actual, expected)

    def test_output_is_compressed(self, gslc_h5, tmp_path):
        """Output datasets should use gzip compression."""
        outpath = tmp_path / "compressed.h5"
        with h5py.File(gslc_h5, "r") as src:
            _extract_subset_from_h5(src, outpath, rows=None, cols=None)

        with h5py.File(outpath, "r") as dst:
            hh_dset = dst[f"{NISAR_GSLC_GRIDS}/frequencyA/HH"]
            assert hh_dset.compression == "gzip"


class TestProcessFile:
    def test_skips_existing_output(self, gslc_h5, tmp_path):
        """process_file returns early if output already exists."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        existing = output_dir / FILE_1
        existing.write_bytes(b"placeholder")

        result = process_file(
            url=str(gslc_h5),
            rows=None,
            cols=None,
            output_dir=output_dir,
        )
        assert result == existing
        # File should still be the placeholder (not overwritten)
        assert existing.read_bytes() == b"placeholder"

    def test_local_file_extraction(self, gslc_h5, tmp_path):
        """process_file extracts a subset from a local file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = process_file(
            url=str(gslc_h5),
            rows=slice(0, 20),
            cols=slice(0, 40),
            output_dir=output_dir,
            polarizations=["HH"],
        )
        assert result.exists()
        with h5py.File(result) as f:
            assert f[f"{NISAR_GSLC_GRIDS}/frequencyA/HH"].shape == (20, 40)
