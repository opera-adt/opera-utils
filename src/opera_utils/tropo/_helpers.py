from __future__ import annotations

import logging
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from scipy.interpolate import RegularGridInterpolator

from opera_utils import get_dates

logger = logging.getLogger(__name__)
TROPO_INTERVAL = timedelta(hours=6)


class MissingTropoError(ValueError):
    """No tropospheric file is available matching the requested time."""


def get_dem_url(frame_id: int) -> str:
    """Generate the URL for DEM data for a given frame ID."""
    return (
        f"s3://opera-adt/disp/disp-s1-static-layers/F{frame_id:05d}/dem_warped_utm.tif"
    )


def get_los_url(frame_id: int) -> str:
    """Generate the URL for LOS data for a given frame ID."""
    return f"s3://opera-adt/disp/disp-s1-static-layers/F{frame_id:05d}/los_enu.tif"


def _open_2d(filename: str | Path) -> xr.DataArray:
    """Open a 2D raster file and return as DataArray."""
    raster = rxr.open_rasterio(filename)
    if isinstance(raster, list):
        return raster[0].squeeze(drop=True)
    else:
        return raster.squeeze(drop=True)


def _build_tropo_index(urls: list[str]) -> pd.Series:
    """Return Series(url, index=datetime UTC)."""
    times = [get_dates(u, fmt="%Y%m%dT%H%M%S")[0] for u in urls]
    return pd.Series(urls, index=pd.to_datetime(times, utc=True).tz_localize(None))


def _bracket(url_series: pd.Series, ts: pd.Timestamp) -> tuple[str, str]:
    """Return (earlier, later) urls within ±6 h; raises if missing."""
    # Last url/date before `ts`
    early = url_series.loc[:ts]
    early_date = early.index[-1]
    early_url = early.iloc[-1]
    # First url/date after `ts`
    late = url_series.loc[ts:]
    late_date = late.index[0]
    late_url = late.iloc[0]
    if (ts - early_date) > TROPO_INTERVAL or (late_date - ts) > TROPO_INTERVAL:
        msg = f"No tropo product within ±6 h of {ts}"
        raise MissingTropoError(msg)
    return early_url, late_url


@lru_cache(maxsize=16)
def _open_crop(
    url: str,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    h_max: float,
) -> xr.Dataset:
    """Lazy-open a single L4 file and subset to bbox+height."""
    ds = xr.open_dataset(url, engine="h5netcdf")
    lat_max, lat_min = lat_bounds  # note south-to-north ordering in slice
    lon_min, lon_max = lon_bounds
    ds = ds.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max),
        height=slice(None, h_max),
    )
    return ds.load()  # pull the small cube into memory


def _create_total_delay(ds: xr.Dataset) -> xr.DataArray:
    return (ds.hydrostatic_delay + ds.wet_delay).squeeze("time", drop=True)


def _interp_in_time(
    ds0: xr.Dataset,
    ds1: xr.Dataset,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    t: pd.Timestamp,
) -> xr.Dataset:
    """Linear time interpolation of total_delay cube."""
    w = (t - t0) / (t1 - t0)
    # keep only (height, lat, lon)
    td0 = _create_total_delay(ds0)
    td1 = _create_total_delay(ds1)
    out = ds0.copy(deep=True)
    out["total_delay"] = (1.0 - w) * td0 + w * td1
    return out


def _height_to_utm_surface(
    td_3d: xr.DataArray,
    dem_utm: xr.DataArray,
    method: str = "linear",
) -> xr.DataArray:
    """Use RegularGridInterpolator exactly like your apply-tropo logic."""
    td_3d = td_3d.rename(latitude="y", longitude="x")  # rioxarray expects y/x
    if dem_utm.rio.crs != "epsg:4326":
        td_utm = td_3d.rio.write_crs("epsg:4326").rio.reproject(
            dem_utm.rio.crs, resampling=Resampling.cubic
        )
        td_utm = td_utm.isel(x=slice(2, -2), y=slice(2, -2))  # trim edges
    else:
        td_utm = td_3d

    rgi = RegularGridInterpolator(
        (td_utm.height.values, td_utm.y.values, td_utm.x.values),
        td_utm.values,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )
    # TODO: Interpolate this in patches? otherwise, its 7k x 9k...
    yy, xx = np.meshgrid(dem_utm.y, dem_utm.x, indexing="ij")
    interp = rgi((dem_utm.values.ravel(), yy.ravel(), xx.ravel()))
    out = dem_utm.copy()
    out.values[:] = interp.reshape(dem_utm.shape).astype("float32")
    return out
