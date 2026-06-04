"""Tests for opera_utils.tropo._helpers pure helper functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from opera_utils.tropo._helpers import (
    MissingTropoError,
    _bracket,
    _build_tropo_index,
    _create_total_delay,
    _interp_in_time,
    get_dem_url,
    get_los_url,
)


class TestStaticLayerUrls:
    def test_dem_url(self):
        assert (
            get_dem_url(11116)
            == "s3://opera-adt/disp/disp-s1-static-layers/F11116/dem_warped_utm.tif"
        )

    def test_los_url(self):
        assert (
            get_los_url(8889)
            == "s3://opera-adt/disp/disp-s1-static-layers/F08889/los_enu.tif"
        )

    def test_frame_id_zero_padded(self):
        # Frame ids are zero-padded to 5 digits
        assert "F00042" in get_dem_url(42)


class TestBuildTropoIndex:
    def test_builds_sorted_datetime_indexed_series(self):
        urls = [
            "tropo_20200101T120000_model.nc",
            "tropo_20200101T000000_model.nc",
            "tropo_20200101T060000_model.nc",
        ]
        series = _build_tropo_index(urls)
        assert isinstance(series.index, pd.DatetimeIndex)
        # Sorted ascending by time
        assert series.index.is_monotonic_increasing
        # First entry corresponds to the earliest (00:00) url
        assert series.iloc[0] == "tropo_20200101T000000_model.nc"
        assert len(series) == 3


class TestBracket:
    @staticmethod
    def _series():
        idx = pd.to_datetime(
            ["2020-01-01T00:00:00", "2020-01-01T06:00:00", "2020-01-01T12:00:00"]
        )
        return pd.Series(["u0", "u6", "u12"], index=idx)

    def test_brackets_within_interval(self):
        early, late = _bracket(self._series(), pd.Timestamp("2020-01-01T03:00:00"))
        assert (early, late) == ("u0", "u6")

    def test_exact_match_brackets_to_neighbors(self):
        # ts exactly equal to a grid point: searchsorted(side="left") puts i at that
        # point, so early=previous, late=that point.
        early, late = _bracket(self._series(), pd.Timestamp("2020-01-01T06:00:00"))
        assert (early, late) == ("u0", "u6")

    def test_before_first_raises(self):
        with pytest.raises(MissingTropoError, match="within"):
            _bracket(self._series(), pd.Timestamp("2019-12-31T23:00:00"))

    def test_after_last_raises(self):
        with pytest.raises(MissingTropoError, match="within"):
            _bracket(self._series(), pd.Timestamp("2020-01-01T13:00:00"))

    def test_gap_larger_than_interval_raises(self):
        idx = pd.to_datetime(["2020-01-01T00:00:00", "2020-01-01T18:00:00"])
        series = pd.Series(["u0", "u18"], index=idx)
        # 06:00 is within 6h of u0 but 12h from u18 -> outside tolerance
        with pytest.raises(MissingTropoError):
            _bracket(series, pd.Timestamp("2020-01-01T06:00:00"))

    def test_non_datetime_index_raises(self):
        series = pd.Series(["a", "b"], index=[0, 1])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _bracket(series, pd.Timestamp("2020-01-01T00:00:00"))

    def test_unsorted_index_raises(self):
        idx = pd.to_datetime(["2020-01-01T12:00:00", "2020-01-01T00:00:00"])
        series = pd.Series(["u12", "u0"], index=idx)
        with pytest.raises(ValueError, match="sorted"):
            _bracket(series, pd.Timestamp("2020-01-01T06:00:00"))


def _delay_ds(hydro: float, wet: float) -> xr.Dataset:
    """Build a tiny tropo dataset with a singleton time dimension."""
    shape = (1, 2, 2, 2)  # time, height, lat, lon
    dims = ("time", "height", "latitude", "longitude")
    return xr.Dataset(
        {
            "hydrostatic_delay": (dims, np.full(shape, hydro, dtype="float64")),
            "wet_delay": (dims, np.full(shape, wet, dtype="float64")),
        },
        coords={
            "time": [np.datetime64("2020-01-01T00:00:00")],
            "height": [0.0, 1000.0],
            "latitude": [10.0, 11.0],
            "longitude": [40.0, 41.0],
        },
    )


class TestCreateTotalDelay:
    def test_sums_and_squeezes_time(self):
        ds = _delay_ds(hydro=2.0, wet=3.0)
        td = _create_total_delay(ds)
        assert "time" not in td.dims
        assert td.dims == ("height", "latitude", "longitude")
        np.testing.assert_allclose(td.values, 5.0)


class TestInterpInTime:
    def test_midpoint_is_average(self):
        ds0 = _delay_ds(hydro=0.0, wet=0.0)  # total = 0
        ds1 = _delay_ds(hydro=5.0, wet=5.0)  # total = 10
        t0 = pd.Timestamp("2020-01-01T00:00:00")
        t1 = pd.Timestamp("2020-01-01T06:00:00")
        t = pd.Timestamp("2020-01-01T03:00:00")  # exact midpoint
        out = _interp_in_time(ds0, ds1, t0, t1, t)
        np.testing.assert_allclose(out["total_delay"].values, 5.0)

    def test_endpoint_returns_first(self):
        ds0 = _delay_ds(hydro=1.0, wet=1.0)  # total = 2
        ds1 = _delay_ds(hydro=5.0, wet=5.0)  # total = 10
        t0 = pd.Timestamp("2020-01-01T00:00:00")
        t1 = pd.Timestamp("2020-01-01T06:00:00")
        out = _interp_in_time(ds0, ds1, t0, t1, t0)  # t == t0 -> weight 0
        np.testing.assert_allclose(out["total_delay"].values, 2.0)
