"""Tests for opera_utils.disp._utils pure helper functions."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from opera_utils.disp._utils import (
    _clamp_chunk_dict,
    _get_border,
    _get_netcdf_encoding,
    flatten,
    last_per_ministack,
    round_mantissa,
)


class TestFlatten:
    def test_flatten_nested_lists(self):
        assert list(flatten([[1, 2], [3], [4, 5]])) == [1, 2, 3, 4, 5]

    def test_flatten_empty(self):
        assert list(flatten([])) == []

    def test_flatten_only_one_level(self):
        # Only one level is flattened; inner nesting is preserved
        assert list(flatten([[[1], [2]], [[3]]])) == [[1], [2], [3]]


class TestRoundMantissa:
    def test_idempotent(self):
        """Rounding an already-rounded array changes nothing further."""
        a = np.linspace(0, 100, 50, dtype="float32")
        once = round_mantissa(a.copy(), keep_bits=8)
        twice = round_mantissa(once.copy(), keep_bits=8)
        np.testing.assert_array_equal(once, twice)

    def test_close_to_original(self):
        a = np.array([1.23456789, 2.3456789, 100.5], dtype="float32")
        rounded = round_mantissa(a.copy(), keep_bits=12)
        assert np.allclose(rounded, a, rtol=1e-3)

    def test_keep_all_bits_is_noop(self):
        """keep_bits equal to the dtype mantissa width returns the array unchanged."""
        a = np.array([1.1, 2.2, 3.3], dtype="float32")
        out = round_mantissa(a.copy(), keep_bits=23)
        np.testing.assert_array_equal(out, a)

    def test_complex_handled(self):
        z = np.array([1.234 + 5.678j, 9.0 - 1.0j], dtype="complex64")
        out = round_mantissa(z.copy(), keep_bits=10)
        assert np.iscomplexobj(out)
        assert np.allclose(out, z, rtol=1e-2)

    def test_integer_array_raises(self):
        with pytest.raises(TypeError, match="float arrays"):
            round_mantissa(np.array([1, 2, 3], dtype="int32"))

    def test_keep_bits_too_large_raises(self):
        with pytest.raises(ValueError, match="keep_bits too large"):
            round_mantissa(np.array([1.0], dtype="float32"), keep_bits=99)


class TestClampChunkDict:
    def test_caps_to_data_shape(self):
        out = _clamp_chunk_dict({"time": 100, "y": 5, "x": 50}, (3, 10, 20))
        assert out == {"time": 3, "y": 5, "x": 20}

    def test_none_uses_full_shape(self):
        out = _clamp_chunk_dict(None, (4, 8, 16))
        assert out == {"time": 4, "y": 8, "x": 16}

    def test_partial_request_filled_from_shape(self):
        out = _clamp_chunk_dict({"y": 2}, (4, 8, 16))
        assert out == {"time": 4, "y": 2, "x": 16}


class TestGetBorder:
    def test_returns_border_median_per_band(self):
        # Interior is large; border is all ones, so the border median is 1.
        data = np.full((2, 4, 4), 100.0)
        data[:, 0, :] = 1.0
        data[:, -1, :] = 1.0
        data[:, :, 0] = 1.0
        data[:, :, -1] = 1.0
        out = _get_border(data)
        assert out.shape == (2, 1, 1)
        np.testing.assert_array_equal(out[:, 0, 0], [1.0, 1.0])

    def test_ignores_nan(self):
        data = np.full((1, 3, 3), 5.0)
        data[0, 0, 0] = np.nan
        out = _get_border(data)
        # nanmedian ignores the NaN; all remaining border pixels are 5.0
        assert out[0, 0, 0] == 5.0


class TestGetNetcdfEncoding:
    def _make_ds(self):
        return xr.Dataset(
            {
                "delay": (("time", "y", "x"), np.zeros((2, 10, 20), dtype="float32")),
                "mask2d": (("y", "x"), np.zeros((10, 20), dtype="float32")),
                "scalar": ((), np.float32(1.0)),
            }
        )

    def test_caps_chunksizes_and_sets_compression(self):
        ds = self._make_ds()
        enc = _get_netcdf_encoding(ds, chunks=(5, 100, 100), compression_level=4)
        # Each chunk is capped to its dimension size: time->2, y->10, x->20
        assert enc["delay"]["chunksizes"] == (2, 10, 20)
        assert enc["delay"]["zlib"] is True
        assert enc["delay"]["complevel"] == 4
        # 2D var uses only the last two chunk entries, capped
        assert enc["mask2d"]["chunksizes"] == (10, 20)

    def test_skips_sub_2d_variables(self):
        ds = self._make_ds()
        enc = _get_netcdf_encoding(ds, chunks=(5, 5, 5))
        assert "scalar" not in enc

    def test_data_vars_subset(self):
        ds = self._make_ds()
        enc = _get_netcdf_encoding(ds, chunks=(5, 5, 5), data_vars=["mask2d"])
        assert set(enc) == {"mask2d"}


class TestLastPerMinistack:
    def test_keeps_last_file_per_generation_time(self):
        # Two ministacks distinguished by their generation timestamp (3rd date).
        # groupby only collapses *consecutive* equal generation times, so the
        # same-ministack files must stay contiguous after a filename sort -- they
        # do here because they share the leading reference date (20160705 vs
        # 20160717).
        gen_a_1 = "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v1.0_20250318T222753Z.nc"
        gen_a_2 = "OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160810T140756Z_v1.0_20250318T222753Z.nc"
        gen_b_1 = "OPERA_L3_DISP-S1_IW_F11116_VV_20160717T140755Z_20160729T140756Z_v1.0_20250319T222753Z.nc"
        result = last_per_ministack([gen_a_1, gen_a_2, gen_b_1])
        # One entry per distinct generation time
        assert len(result) == 2
        # Within the first generation group, the lexicographically-last file wins
        assert gen_a_2 in result
        assert gen_b_1 in result
        assert gen_a_1 not in result
