"""Apply tropospheric corrections to DEM using LOS geometry."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm

from opera_utils import get_dates, sort_files_by_date

from ._helpers import _open_2d

logger = logging.getLogger(__name__)

GTIFF_KWARGS = {
    "compress": "deflate",
    "tiled": True,
    "predictor": 2,
    "dtype": "float32",
    "nbits": 16,
}


def _height_to_dem_surface(
    da_tropo_cube: xr.DataArray,
    dem: xr.DataArray,
    method: str = "linear",
) -> xr.DataArray:
    """Interpolate 3D tropospheric delay to DEM surface heights.

    Parameters
    ----------
    da_tropo_cube : xr.DataArray
        3D tropospheric delay with dims (height, lat/y, lon/x).
    dem : xr.DataArray
        DEM with surface heights.
    method : {"linear", "nearest"}
        Interpolation method for RegularGridInterpolator.

    Returns
    -------
    xr.DataArray
        2D tropospheric delay at DEM surface (same grid as `dem`).

    """
    # Ensure consistent coordinate naming
    if "latitude" in da_tropo_cube.dims:
        da_tropo_cube = da_tropo_cube.rename(latitude="y", longitude="x")

    # Reproject to DEM CRS if needed
    if dem.rio.crs and da_tropo_cube.rio.crs != dem.rio.crs:
        if da_tropo_cube.rio.crs is None:
            da_tropo_cube = da_tropo_cube.rio.write_crs("epsg:4326")
        td_utm = da_tropo_cube.rio.reproject(dem.rio.crs, resampling=Resampling.cubic)
        # Trim edges to avoid interpolation artifacts
        td_utm = td_utm.isel(x=slice(2, -2), y=slice(2, -2))
    else:
        td_utm = da_tropo_cube

    # Build interpolator
    rgi = RegularGridInterpolator(
        (td_utm.height.values, td_utm.y.values, td_utm.x.values),
        td_utm.values,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    # Coordinates for DEM grid
    yy, xx = np.meshgrid(dem.y.values, dem.x.values, indexing="ij")

    pts = np.column_stack([dem.values.ravel(), yy.ravel(), xx.ravel()])
    vals = rgi(pts)

    out = dem.copy()
    out.values[:] = vals.reshape(dem.shape).astype("float32")
    return out


def _compute_reference_correction(
    first_cropped: Path,
    dem_path: Path,
    incidence_angle_path: Path,
    interp_method: str,
) -> xr.DataArray:
    """Compute the day-1 LOS correction once (serial)."""
    dem = _open_2d(dem_path)
    los_up = _read_los_up(incidence_angle_path)

    ds0 = xr.open_dataset(first_cropped, engine="h5netcdf")
    zenith_delay_2d = _height_to_dem_surface(ds0.total_delay, dem, method=interp_method)

    if los_up.shape != zenith_delay_2d.shape:
        # reproject/align LOS raster to dem grid just in case
        los_up = los_up.rio.reproject_match(zenith_delay_2d)
    # Note the -1: match the line-of-sight convention of DISP
    # where positive means apparent uplift (decrease in delay)
    ref_corr = -1 * (zenith_delay_2d / los_up)
    # Attach rio attrs to keep CRS/transform for saving later if needed
    ref_corr = ref_corr.rio.write_crs(dem.rio.crs or "epsg:4326")
    ref_corr.rio.write_transform(dem.rio.transform(), inplace=True)
    return ref_corr


def _create_reference_correction(
    ref_tropo_file: Path,
    ref_date_str: str,
    dem_path: Path,
    incidence_angle_path: Path,
    interp_method: str,
    output_dir: Path,
) -> Path:
    ref_corr_path = output_dir / f"reference_tropo_correction_{ref_date_str}.tif"
    if ref_corr_path.exists():
        logger.info(f"Reference correction already exists: {ref_corr_path}")
    else:
        logger.info(f"Computing reference correction for {ref_date_str}")
        ref_corr = _compute_reference_correction(
            ref_tropo_file, dem_path, incidence_angle_path, interp_method
        )
        ref_corr.rio.to_raster(ref_corr_path, **GTIFF_KWARGS)
    return ref_corr_path


def _apply_one(
    cropped_file: Path,
    dem_path: Path,
    incidence_angle_path: Path,
    interp_method: str,
    output_file: Path,
    ref_corr_path: Path | None,
    ref_date_str: str | None,
    fmt: str = "%Y%m%dT%H%M%S",
) -> tuple[str, str]:
    """Worker for one date. Returns (date_str, status)."""
    try:
        date_str = get_dates(cropped_file, fmt=fmt)[0].strftime(fmt)

        if output_file.exists():
            return (date_str, "skipped")

        dem = _open_2d(dem_path)
        los_up = _read_los_up(incidence_angle_path)

        ds = xr.open_dataset(cropped_file, engine="h5netcdf")
        zenith_delay_2d = _height_to_dem_surface(
            ds.total_delay, dem, method=interp_method
        )
        if los_up.shape != zenith_delay_2d.shape:
            # reproject/align LOS raster to dem grid just in case
            los_up = los_up.rio.reproject_match(zenith_delay_2d)
        # Note the -1: match the line-of-sight convention of DISP
        # where positive means apparent uplift (decrease in delay)
        los_correction = -1 * (zenith_delay_2d / los_up)

        # Subtract reference if given
        if ref_corr_path is not None:
            ref_corr = rxr.open_rasterio(ref_corr_path, masked=True).squeeze()
            los_correction = (los_correction - ref_corr).astype("float32")

        attrs = {
            "interpolation_method": interp_method,
            "units": "meters",
            "line of sight convention": (
                "Positive means decrease in delay (apparent uplift towards the"
                " satellite)"
            ),
        }
        if ref_date_str is not None:
            attrs["reference_date"] = ref_date_str

        los_correction.rio.update_attrs(attrs, inplace=True)
        los_correction.rio.to_raster(output_file, **GTIFF_KWARGS)

    except Exception as e:
        return (str(cropped_file), f"error:{type(e).__name__}: {e}")
    else:
        return (date_str, "ok")


def _read_los_up(incidence_angle_path: Path) -> xr.DataArray:
    da = rxr.open_rasterio(incidence_angle_path).squeeze(drop=True)
    # LOS 'up' component equals cos(incidence). Convert degrees -> radians, then cos.
    return np.cos(np.radians(da))


def apply_tropo(
    cropped_tropo_list: list[Path],
    dem_path: Path,
    incidence_angle_path: Path,
    output_dir: Path = Path("tropo_corrections"),
    interp_method: str = "linear",
    subtract_first_date: bool = True,
    num_workers: int = 2,
) -> None:
    """Apply tropospheric corrections using DEM and LOS geometry (parallel).

    Parameters
    ----------
    cropped_tropo_list : list[Path]
        Paths to cropped TROPO NetCDFs from `tropo-crop`.
    dem_path : Path
        DEM GeoTIFF (UTM or WGS84).
    incidence_angle_path : Path
        Raster containing ellipsoidal incidence angle (in degrees) for each pixel
        of `dem_path`.
    output_dir : Path
        Output directory for correction GeoTIFFs.
    interp_method : {"linear", "nearest", "slinear", "cubic", "quintic", "pchip"}
        RegularGridInterpolator method.
    subtract_first_date : bool
        If True, subtract day-1 correction from every date.
    num_workers : int
        Number of processes.
        Default is 2.

    """
    if not cropped_tropo_list:
        msg = "No inputs provided."
        raise ValueError(msg)
    methods = {"linear", "nearest", "slinear", "cubic", "quintic", "pchip"}
    if interp_method not in methods:
        msg = f"interp_method must be in {methods} for RegularGridInterpolator"
        raise ValueError(msg)

    output_dir.mkdir(exist_ok=True, parents=True)

    fmt = "%Y%m%dT%H%M%S"
    files_sorted, date_tups_sorted = sort_files_by_date(
        cropped_tropo_list, file_date_fmt=fmt
    )
    dates_sorted = [d[0] for d in date_tups_sorted]

    ref_corr_path: Path | None = None
    ref_date_str: str | None = None

    # Precompute reference correction once if requested
    if subtract_first_date:
        ref_corr_path = _create_reference_correction(
            Path(files_sorted[0]),
            dates_sorted[0].strftime(fmt),
            dem_path,
            incidence_angle_path,
            interp_method,
            output_dir,
        )
        # Shift to ignore the first reference date now
        dates_sorted = dates_sorted[1:]
        files_sorted = files_sorted[1:]

    # Plan work outputs
    out_paths: list[Path] = []
    for d in dates_sorted:
        date_str = d.strftime(fmt)
        out_path = output_dir / f"tropo_correction_{date_str}.tif"
        out_paths.append(out_path)

    logger.info(f"Processing {len(dates_sorted)} date(s) with {num_workers} worker(s)")
    status_counts = {"ok": 0, "skipped": 0, "skipped_ref": 0, "error": 0}

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(
                _apply_one,
                Path(cropped_file),
                dem_path,
                incidence_angle_path,
                interp_method,
                Path(out_file),
                ref_corr_path,
                ref_date_str,
                fmt=fmt,
            )
            for cropped_file, out_file in zip(files_sorted, out_paths, strict=True)
        ]

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Interpolating tropospheric corrections",
            unit="scene",
        ):
            date_str, status = fut.result()
            if status.startswith("error:"):
                status_counts["error"] += 1
                logger.error(f"{date_str}: {status}")
            else:
                status_counts[status] = status_counts.get(status, 0) + 1

    logger.info(
        "Done. "
        f"ok={status_counts['ok']}, "
        f"skipped={status_counts['skipped']}, "
        f"skipped_ref={status_counts['skipped_ref']}, "
        f"errors={status_counts['error']}"
    )
