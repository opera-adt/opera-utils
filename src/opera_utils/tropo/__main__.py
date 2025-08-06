"""Create tropospheric displacement corrections.

Accepts an OPERA frame ID, or list of DISP products (which
can be local, or filenames/urls)

One GeoTIFF per DISP secondary scene is written to --out-dir.
"""

from pathlib import Path

import pandas as pd
import tyro
from loguru import logger
from shapely.geometry import box

from opera_utils.disp import DispProductStack, search

from ._helpers import (
    MissingTropoError,
    _bracket,
    _build_tropo_index,
    _height_to_utm_surface,
    _interp_in_time,
    _open_2d,
    _open_crop,
    get_dem_url,
)


def main(
    frame_id: int | None = None,
    disp_files: list[Path] | None = None,
    tropo_urls_file: Path = Path("tropo_urls.txt"),
    dem_path: Path | str | None = None,
    out_dir: Path = Path("corrections"),
    margin_deg: float = 0.3,
):
    """Create tropospheric corrections for Displacement products."""
    if (frame_id is None) == (disp_files is None):
        msg = "Specify *either* --frame-id or --disp-files"
        raise ValueError(msg)

    # load DISP stack
    if disp_files is None:
        products = search(frame_id=frame_id)
        stack = DispProductStack(products=products)
    else:
        stack = DispProductStack.from_file_list(disp_files)

    frame_id = stack.frame_id
    stack.to_dataframe()
    # ref_dt = pd.to_datetime(df.reference_datetime.iloc[0], utc=True)
    # sec_dts = pd.to_datetime(df.secondary_datetime, utc=True)

    ref_dt_series = pd.to_datetime(stack.reference_dates).tz_localize(None).to_series()
    sec_dt_series = pd.to_datetime(stack.secondary_dates).tz_localize(None).to_series()

    dem_url = get_dem_url(stack.frame_id) if dem_path is None else dem_path
    logger.info(f"Opening DEM at {dem_url}")
    dem_utm = _open_2d(dem_url)
    bbox = box(*dem_utm.rio.transform_bounds("epsg:4326")).buffer(margin_deg).bounds
    lat_bounds = (bbox[3], bbox[1])  # north, south
    lon_bounds = (bbox[0], bbox[2])  # west , east
    h_max = float(dem_utm.max())

    tropo_urls = Path(tropo_urls_file).read_text(encoding="utf-8").splitlines()
    tropo_idx_series = _build_tropo_index(tropo_urls)

    # cache delays for every unique timestamp
    # TODO: figure out appropriate cacheing...
    delay_per_date = {}
    # for ts in {ref_dt, *sec_dts}:
    out_dir.mkdir(exist_ok=True, parents=True)
    for ref_ts, sec_ts in zip(ref_dt_series, sec_dt_series):
        logger.info(f"Running DISP {ref_ts} -> {sec_ts}")
        if ref_ts not in delay_per_date:
            try:
                early_u_ref, late_u_ref = _bracket(tropo_idx_series, ref_ts)
            except MissingTropoError:
                logger.info(f"No available tropo files for {ref_ts}")
                continue

            logger.info(f"Found {early_u_ref, late_u_ref}")
            ds0 = _open_crop(early_u_ref, lat_bounds, lon_bounds, h_max)
            ds1 = _open_crop(late_u_ref, lat_bounds, lon_bounds, h_max)

            # delay_2d[ts] = _height_to_utm_surface(td_interp.total_delay, dem_utm)
            td_interp_ref = _interp_in_time(
                ds0,
                ds1,
                ds0.time.to_pandas().item(),
                ds1.time.to_pandas().item(),
                ref_ts,
            )
            surf_ref = _height_to_utm_surface(td_interp_ref.total_delay, dem_utm)
            delay_per_date[ref_ts] = surf_ref
        else:
            surf_ref = delay_per_date[ref_ts]

        try:
            early_u_sec, late_u_sec = _bracket(tropo_idx_series, sec_ts)
        except MissingTropoError:
            logger.info(f"No available tropo files for {sec_ts}")
            continue
        logger.info(f"Interp. for  {early_u_sec, late_u_sec}")
        ds0s = _open_crop(early_u_sec, lat_bounds, lon_bounds, h_max)
        ds1s = _open_crop(late_u_sec, lat_bounds, lon_bounds, h_max)
        td_interp_sec = _interp_in_time(
            ds0s,
            ds1s,
            ds0s.time.to_pandas().item(),
            ds1s.time.to_pandas().item(),
            sec_ts,
        )
        surf_sec = _height_to_utm_surface(td_interp_sec.total_delay, dem_utm)
        corr = surf_sec - surf_ref

        #  write corrections
        # for _, row in df.iterrows():
        # sec_ts = pd.to_datetime(sec_ts, utc=True)
        # corr = delay_2d[sec_ts] - delay_2d[ref_ts]
        time_pair_str = f"{ref_ts:%Y%m%dT%H%M%S}Z_{sec_ts:%Y%m%dT%H%M%S}Z"
        out_name = f"tropo_corr_F{frame_id:05d}_{time_pair_str}.tif"
        corr.rio.to_raster(out_dir / out_name)
        logger.info(f"Wrote {out_name}")


if __name__ == "__main__":
    tyro.cli(main)
