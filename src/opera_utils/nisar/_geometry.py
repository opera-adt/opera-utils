"""Incidence angle and LOS unit vector computation from NISAR GSLC radarGrid."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import rioxarray as rxr
import xarray as xr
from pyproj import Transformer
from rasterio.enums import Resampling
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm


def _interp_chunk(
    args: tuple,
) -> tuple[slice, list[np.ndarray]]:
    sl, y_vals, x_vals, dem_chunk, heights, y_rg_inc, x_rg, arrays_3d, src_epsg, dem_crs = args
    transformer = Transformer.from_crs(dem_crs, f"EPSG:{src_epsg}", always_xy=True)
    yy_c, xx_c = np.meshgrid(y_vals, x_vals, indexing="ij")
    xx_rg, yy_rg = transformer.transform(xx_c.ravel(), yy_c.ravel())
    pts = np.column_stack([dem_chunk.ravel(), yy_rg, xx_rg])
    results = []
    for arr in arrays_3d:
        rgi = RegularGridInterpolator(
            (heights, y_rg_inc, x_rg),
            arr,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        results.append(rgi(pts).reshape(dem_chunk.shape).astype("float32"))
    return sl, results


def prepare_incidence_angle(
    gslc_path: str | Path,
    dem_path: str | Path,
    ref_tif: str | Path,
    output_path: str | Path = "incidence_angle_surface.tif",
    los_output_path: str | Path = "los_unit_vector.tif",
    chunk_size: int = 200,
    n_workers: int = 8,
) -> tuple[Path, Path]:
    """Compute incidence angle and LOS unit vector rasters from a NISAR GSLC.

    Reads the 3-D radarGrid datacube from ``gslc_path``, interpolates it to
    the DEM surface height at each pixel, then reprojects the result to match
    ``ref_tif``.

    Parameters
    ----------
    gslc_path : str or Path
        Path to a NISAR GSLC HDF5 file containing
        ``/science/LSAR/GSLC/metadata/radarGrid``.
    dem_path : str or Path
        Path to a DEM GeoTIFF covering the area of interest.
    ref_tif : str or Path
        Reference GeoTIFF whose grid the outputs are reprojected to match.
    output_path : str or Path
        Output path for the incidence angle raster (single band, degrees).
    los_output_path : str or Path
        Output path for the LOS unit vector raster
        (3 bands: east, north, up).
    chunk_size : int
        Number of DEM rows to process per parallel chunk.
    n_workers : int
        Number of worker processes for parallel interpolation.

    Returns
    -------
    tuple[Path, Path]
        Paths to the written incidence angle and LOS unit vector rasters.

    """
    with h5py.File(gslc_path) as f:
        rg = f["science/LSAR/GSLC/metadata/radarGrid"]
        heights = rg["heightAboveEllipsoid"][:]
        x_rg = rg["xCoordinates"][:]
        y_rg = rg["yCoordinates"][:]
        inc_3d = rg["incidenceAngle"][:]
        los_x_3d = rg["losUnitVectorX"][:]
        los_y_3d = rg["losUnitVectorY"][:]
        src_epsg = int(rg["projection"].attrs["epsg_code"])

    assert np.all(np.diff(heights) > 0), "heights must be strictly increasing"
    assert np.all(np.diff(x_rg) > 0), "x_rg must be strictly increasing"

    t_to_rg = Transformer.from_crs("EPSG:4326", f"EPSG:{src_epsg}", always_xy=True)
    dem_da = rxr.open_rasterio(dem_path, masked=True).squeeze()
    dem_val = dem_da.values
    dem_crs = dem_da.rio.crs

    dem_xs = [float(dem_da.x.min()), float(dem_da.x.max())]
    dem_ys = [float(dem_da.y.min()), float(dem_da.y.max())]
    t_dem_to_rg = Transformer.from_crs(dem_crs, f"EPSG:{src_epsg}", always_xy=True)
    dem_xs_rg, dem_ys_rg = t_dem_to_rg.transform(dem_xs, dem_ys)

    x_overlap = max(dem_xs_rg) > x_rg.min() and min(dem_xs_rg) < x_rg.max()
    y_overlap = max(dem_ys_rg) > y_rg.min() and min(dem_ys_rg) < y_rg.max()
    if not (x_overlap and y_overlap):
        raise ValueError(
            "DEM and radarGrid do not overlap. "
            "Check that gslc_path covers your area of interest."
        )

    # Flip y axis so RegularGridInterpolator gets strictly increasing coords
    y_rg_flip = y_rg[::-1]
    inc_3d_flip = inc_3d[:, ::-1, :]
    los_x_3d_flip = los_x_3d[:, ::-1, :]
    los_y_3d_flip = los_y_3d[:, ::-1, :]

    dem_crs_str = str(dem_crs)
    n_rows = dem_da.shape[0]
    slices = [
        slice(i, min(i + chunk_size, n_rows)) for i in range(0, n_rows, chunk_size)
    ]
    tasks = [
        (
            sl,
            dem_da.y.values[sl],
            dem_da.x.values,
            dem_val[sl],
            heights,
            y_rg_flip,
            x_rg,
            [inc_3d_flip, los_x_3d_flip, los_y_3d_flip],
            src_epsg,
            dem_crs_str,
        )
        for sl in slices
    ]

    inc_surface = np.full(dem_da.shape, np.nan, dtype="float32")
    los_east_surf = np.full(dem_da.shape, np.nan, dtype="float32")
    los_north_surf = np.full(dem_da.shape, np.nan, dtype="float32")

    with mp.get_context("fork").Pool(processes=n_workers) as pool:
        for sl, results in tqdm(
            pool.imap(_interp_chunk, tasks),
            total=len(tasks),
            desc="incidence + LOS interp",
        ):
            inc_surface[sl] = results[0]
            los_east_surf[sl] = results[1]
            los_north_surf[sl] = results[2]

    los_up_surf = np.sqrt(
        np.clip(1.0 - los_east_surf**2 - los_north_surf**2, 0, None)
    ).astype("float32")

    ref = rxr.open_rasterio(ref_tif, masked=True).squeeze()

    inc_da = dem_da.copy(data=inc_surface).rio.write_crs(dem_crs)
    inc_matched = inc_da.rio.reproject_match(ref, resampling=Resampling.bilinear)
    inc_matched.rio.to_raster(output_path, compress="deflate", dtype="float32")

    los_stack = np.stack([los_east_surf, los_north_surf, los_up_surf], axis=0)
    los_da = xr.DataArray(
        los_stack,
        dims=("band", "y", "x"),
        coords={"band": [1, 2, 3], "y": dem_da.y.values, "x": dem_da.x.values},
    ).rio.write_crs(dem_crs)
    los_matched = los_da.rio.reproject_match(ref, resampling=Resampling.bilinear)
    los_matched.rio.to_raster(
        los_output_path,
        compress="deflate",
        dtype="float32",
        descriptions=["los_east", "los_north", "los_up"],
    )

    return Path(output_path), Path(los_output_path)
