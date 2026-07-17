"""Tests for opera_utils.nisar._wavelength module."""

from __future__ import annotations

import re

import h5py
import numpy as np
import pytest

from opera_utils.constants import SPEED_OF_LIGHT
from opera_utils.nisar import GslcProduct, get_nisar_wavelength

# Example filename from the NISAR naming convention (see constants.py)
GSLC_FILENAME = "NISAR_L2_PR_GSLC_004_076_A_022_2005_QPDH_A_20251103T110514_20251103T110549_X05007_N_F_J_001.h5"

# NISAR L-band center frequencies (Hz) for the 20 MHz (A) and 5 MHz (B) modes
FREQ_A = 1_257_500_000.0
FREQ_B = 1_293_500_000.0
FREQ_S = 3_200_000_000.0

LSAR_GRIDS = "/science/LSAR/GSLC/grids"
SSAR_GRIDS = "/science/SSAR/GSLC/grids"

# Sentinel: create a *group* named centerFrequency instead of a dataset
CENTER_FREQUENCY_AS_GROUP = object()


@pytest.fixture
def make_gslc(tmp_path):
    """Return a factory creating a minimal GSLC HDF5 file.

    The factory takes a mapping of frequency-group path to the value stored in
    that group's ``centerFrequency`` dataset. A value of ``None`` creates the
    group without a ``centerFrequency``; `CENTER_FREQUENCY_AS_GROUP` creates a
    subgroup named ``centerFrequency`` instead of a dataset.
    """

    def _make(center_frequencies, image_datasets=(), filename=GSLC_FILENAME):
        path = tmp_path / filename
        with h5py.File(path, "w") as hf:
            for group_path, value in center_frequencies.items():
                group = hf.require_group(group_path)
                if value is CENTER_FREQUENCY_AS_GROUP:
                    group.create_group("centerFrequency")
                elif value is not None:
                    group.create_dataset("centerFrequency", data=value)
            for dataset_path in image_datasets:
                hf.create_dataset(
                    dataset_path, data=np.zeros((2, 2), dtype=np.complex64)
                )
        return path

    return _make


@pytest.fixture
def gslc_h5(make_gslc):
    """A GSLC file with LSAR frequencyA/frequencyB groups and image data."""
    return make_gslc(
        {
            f"{LSAR_GRIDS}/frequencyA": FREQ_A,
            f"{LSAR_GRIDS}/frequencyB": FREQ_B,
        },
        image_datasets=[
            f"{LSAR_GRIDS}/frequencyA/HH",
            f"{LSAR_GRIDS}/frequencyB/VV",
        ],
    )


class TestSubdatasetPathResolution:
    @pytest.mark.parametrize(
        ("subdataset", "center_frequency"),
        [
            (f"{LSAR_GRIDS}/frequencyA/HH", FREQ_A),
            (f"{LSAR_GRIDS}/frequencyB/VV", FREQ_B),
            (f"{LSAR_GRIDS}/frequencyB", FREQ_B),
        ],
        ids=["frequencyA-dataset", "frequencyB-dataset", "frequency-group-itself"],
    )
    def test_resolves_parent_frequency_group(
        self, gslc_h5, subdataset, center_frequency
    ):
        wavelength = get_nisar_wavelength(gslc_h5, subdataset)
        assert wavelength == pytest.approx(SPEED_OF_LIGHT / center_frequency)

    def test_band_agnostic_ssar_path(self, make_gslc):
        h5path = make_gslc({f"{SSAR_GRIDS}/frequencyA": FREQ_S})
        wavelength = get_nisar_wavelength(h5path, f"{SSAR_GRIDS}/frequencyA")
        assert wavelength == pytest.approx(SPEED_OF_LIGHT / FREQ_S)

    def test_reads_from_subdataset_group_not_first_match(self, make_gslc):
        # Both groups exist; the wavelength must come from the group containing
        # the subdataset, not from whichever frequency group appears first
        h5path = make_gslc(
            {
                f"{LSAR_GRIDS}/frequencyA": FREQ_A,
                f"{LSAR_GRIDS}/frequencyB": FREQ_B,
            }
        )
        wavelength = get_nisar_wavelength(h5path, f"{LSAR_GRIDS}/frequencyB/HH")
        assert wavelength == pytest.approx(SPEED_OF_LIGHT / FREQ_B)

    @pytest.mark.parametrize(
        ("subdataset", "num_found"),
        [
            (f"{LSAR_GRIDS}/HH", 0),
            (f"{LSAR_GRIDS}/frequencyC/HH", 0),
            ("/x/frequencyAB/HH", 0),
            ("/x/frequencyA/frequencyB/HH", 2),
        ],
        ids=[
            "no-frequency-segment",
            "frequencyC",
            "substring-frequencyAB",
            "two-frequency-segments",
        ],
    )
    def test_invalid_subdataset_paths(self, gslc_h5, subdataset, num_found):
        match = re.escape(repr(subdataset)) + f".*found {num_found}"
        with pytest.raises(ValueError, match=match):
            get_nisar_wavelength(gslc_h5, subdataset)


class TestCenterFrequencyValidation:
    FREQ_A_GROUP = f"{LSAR_GRIDS}/frequencyA"

    def test_accepts_positive_integer_dtype(self, make_gslc):
        h5path = make_gslc({self.FREQ_A_GROUP: np.int64(1_257_500_000)})
        wavelength = get_nisar_wavelength(h5path, self.FREQ_A_GROUP)
        assert wavelength == pytest.approx(SPEED_OF_LIGHT / FREQ_A)

    @pytest.mark.parametrize(
        "value",
        [
            None,
            CENTER_FREQUENCY_AS_GROUP,
            np.array([FREQ_A, FREQ_B]),
            np.complex64(FREQ_A),
            True,
            "1257500000",
            -1.0,
            0.0,
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
        ids=[
            "missing",
            "group-not-dataset",
            "non-scalar",
            "complex-dtype",
            "bool-dtype",
            "string-dtype",
            "negative",
            "zero",
            "nan",
            "positive-inf",
            "negative-inf",
        ],
    )
    def test_rejects_invalid_center_frequency(self, make_gslc, value):
        h5path = make_gslc({self.FREQ_A_GROUP: value})
        with pytest.raises(ValueError, match="centerFrequency"):
            get_nisar_wavelength(h5path, self.FREQ_A_GROUP)

    def test_error_message_includes_filename_and_dataset_path(self, make_gslc):
        h5path = make_gslc({self.FREQ_A_GROUP: None})
        with pytest.raises(ValueError, match="dataset not found") as excinfo:
            get_nisar_wavelength(h5path, self.FREQ_A_GROUP)
        message = str(excinfo.value)
        assert str(h5path) in message
        assert f"{self.FREQ_A_GROUP}/centerFrequency" in message


class TestKnownWavelengths:
    def test_nisar_l_band_wavelength(self, gslc_h5):
        wavelength = get_nisar_wavelength(gslc_h5, f"{LSAR_GRIDS}/frequencyA/HH")
        assert wavelength == pytest.approx(SPEED_OF_LIGHT / 1_257_500_000.0)
        # Independent literal check: NISAR L-band 20 MHz mode is ~23.84 cm
        assert wavelength == pytest.approx(0.23840, rel=1e-4)


class TestGslcProductGetWavelength:
    def test_default_frequency_a(self, gslc_h5):
        product = GslcProduct.from_filename(gslc_h5)
        assert product.get_wavelength() == pytest.approx(SPEED_OF_LIGHT / FREQ_A)

    def test_frequency_b(self, gslc_h5):
        product = GslcProduct.from_filename(gslc_h5)
        assert product.get_wavelength("B") == pytest.approx(SPEED_OF_LIGHT / FREQ_B)

    def test_invalid_frequency(self, gslc_h5):
        product = GslcProduct.from_filename(gslc_h5)
        with pytest.raises(ValueError, match="Invalid frequency"):
            product.get_wavelength("C")

    def test_frequency_b_only_product(self, make_gslc):
        h5path = make_gslc({f"{LSAR_GRIDS}/frequencyB": FREQ_B})
        product = GslcProduct.from_filename(h5path)
        assert product.get_wavelength("B") == pytest.approx(SPEED_OF_LIGHT / FREQ_B)
        with pytest.raises(ValueError, match="dataset not found"):
            product.get_wavelength()
