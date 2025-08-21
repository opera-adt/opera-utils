from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr

from opera_utils.disp import DispProduct

DEFAULT_CHUNKS = (1, 512, 512)


def _decimal_year(t: np.ndarray) -> np.ndarray:
    """Convert np.datetime64 array to decimal years."""
    dt = t.astype("datetime64[D]").astype(int)
    yr_start = np.datetime64("1970-01-01", "D").astype(int)
    days = dt - yr_start
    year = 1970 + days // 365.25
    frac = (days % 365.25) / 365.25
    return year + frac


def _mintpy_metadata(prod: DispProduct) -> dict[str, str]:
    """Minimal attribute set expected by MintPy."""
    gt = prod.transform  # affine
    meta = {
        "FILE_TYPE": "timeseries",
        "UNIT": "m",
        "LENGTH": str(prod.shape[0]),
        "WIDTH": str(prod.shape[1]),
        "X_FIRST": str(gt.c),
        "Y_FIRST": str(gt.f),
        "X_STEP": str(gt.a),
        "Y_STEP": str(gt.e),
        "X_UNIT": "meters",
        "Y_UNIT": "meters",
        "EPSG": str(prod.epsg),
        "UTM_ZONE": prod.crs.utm_zone,
        "PLATFORM": "Sentinel-1",
        "ORBIT_DIRECTION": _get_orbit_direction(prod),
        # Reasonable constants
        "HEIGHT": "750000.0",
        "WAVELENGTH": "0.05546576",  # m (C-band S-1)
    }
    return meta


def _get_orbit_direction(prod: DispProduct) -> str:
    """Get the orbit direction from the product."""
    dset = "/identification/orbit_pass_direction"
    with h5py.File(prod.filename, "r") as f:
        return f[dset][()].decode("utf-8")


def _write_hdf5(
    out_f: Path,
    datasets: dict[str, tuple[Any, tuple[Any, ...], Any]],
    attrs: dict[str, str],
) -> None:
    """Writer that mimics MintPy's `writefile.layout_hdf5()`."""
    with h5py.File(out_f, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        # Copy all datasets from the input file
        compression: str | None = "gzip"
        for name, (dtype, shape, data) in datasets.items():
            requested_chunks = DEFAULT_CHUNKS[-len(shape) :]  # Pick only last 2
            if len(shape) < 2:
                chunks = None
                compression = None
            else:
                chunks = tuple(min(c, s) for c, s in zip(requested_chunks, shape))
            dset = f.create_dataset(
                name, shape, dtype=dtype, chunks=chunks, compression=compression
            )
            if data is not None:
                dset[...] = data


def _write_timeseries_vds(
    out_path: Path,
    src_nc: Path,
    ds: xr.Dataset,
    meta: dict[str, str],
    dates: np.ndarray,
) -> None:
    """Create timeseries.h5 that virtually links to /displacement in src_nc."""
    with h5py.File(out_path, "w", libver="latest") as f:
        # VDS needs HDF5 >=1.10, use libver="latest"
        # file-level attributes
        for k, v in meta.items():
            f.attrs[k] = v

        nt, ny, nx = ds["displacement"].shape

        # date / bperp vectors
        f.create_dataset("date", data=dates, dtype=dates.dtype)
        f.create_dataset(
            "bperp",
            data=ds["perpendicular_baseline"].data,
            dtype=np.float32,
        )

        # the virtual dataset itself
        src_shape = (nt, ny, nx)
        dtype = ds["displacement"].dtype

        layout = h5py.VirtualLayout(shape=src_shape, dtype=dtype)
        layout[:] = h5py.VirtualSource(
            str(src_nc),  # external file
            "/displacement",  # path inside that file
            shape=src_shape,
        )

        # NOTE: compression/chunks must be None on a VDS
        vds = f.create_virtual_dataset("timeseries", layout, fillvalue=np.nan)
        # MintPy sometimes checks for UNIT on the dataset itself
        vds.attrs["UNIT"] = "m"


def create_reliability_mask(
    da_recommended: xr.DataArray,
    out_dir: Path,
    *,
    reliability_threshold: float = 0.90,
) -> None:
    """Build 2-D reliability mask in MintPy format.

    Parameters
    ----------
    da_recommended : xr.DataArray
        Input 3D stack of recommended masks
    out_dir : Path
        Destination folder.
    reliability_threshold : float, default 0.9
        Pixel must be valid in >= `reliability_threshold` of epochs to be kept.

    """
    out_dir.mkdir(exist_ok=True, parents=True)

    nt, ny, nx = da_recommended.shape

    # Compute density map and reliability mask
    sum_valid = da_recommended.sum(axis=0).astype(np.float32)  # (#y, #x)
    density = sum_valid / nt
    thresh = int(np.ceil(nt * reliability_threshold))
    reliable = (sum_valid >= thresh).astype(np.int8)

    # Fetch minimal metadata (just copy attrs from timeseries.h5)
    if (out_dir / "timeseries.h5").exists():
        with h5py.File(out_dir / "timeseries.h5", "r") as hf:
            meta = dict(hf.attrs.items())
    else:
        meta = {"UNIT": "1"}

    percent = int(reliability_threshold * 100)

    # 4a) timeseries_density.h5
    density_meta = meta | {"FILE_TYPE": "timeseriesdensity", "UNIT": "1"}
    _write_hdf5(
        out_dir / "timeseries_density.h5",
        {"timeseriesDensity": (np.float32, (ny, nx), density)},
        density_meta,
    )

    mask_meta = meta | {"FILE_TYPE": "mask", "DATA_TYPE": "int8", "UNIT": "1"}
    _write_hdf5(
        out_dir / f"recommended_mask_{percent}thresh.h5",
        {"recommendedMask": (np.int8, (ny, nx), reliable)},
        mask_meta,
    )

    print(
        "reliability products: "
        f"timeseries_density.h5 & recommended_mask_{percent}thresh.h5"
    )


def create_static_layers(
    los_enu_path: Path | str,
    meta: dict[str, Any],
    layover_shadow_mask_path: Path | str | None = None,
    dem_path: Path | str | None = None,
    outdir: Path = Path("mintpy"),
) -> None:
    """Create static layer products from reformatted DISP-S1 data.

    Parameters
    ----------
    los_enu_path : Path | str
        Path to the LOS-ENU geometry file.
    meta : dict[str, Any]
        Metadata to be written to the static layer files.
    dem_path : Path | str, optional
        Path to the DEM file.
    layover_shadow_mask_path : Path | str | None, optional
        Path to the layover/shadow mask file.
    outdir : Path, optional
        Output directory for the static layer files.
        Default is "mintpy".

    """
    outdir.mkdir(parents=True, exist_ok=True)

    ny, nx = int(meta["LENGTH"]), int(meta["WIDTH"])
    geometry_meta = meta | {"FILE_TYPE": "geometry"}

    los_enu_data = xr.open_dataarray(los_enu_path).data
    east, north, up = los_enu_data
    # Incidence angle is the angle between the line-of-sight (LOS) vector and
    # the normal to the ellipsoid at the target
    incidence_angle = np.degrees(np.arccos(up))

    # Calculate azimuth angle from East and North components
    # See calc_azimuth_from_east_north_obs for reference
    azimuth_angle = -1 * np.rad2deg(np.arctan2(east, north)) % 360

    if dem_path is not None:
        dem = xr.open_dataarray(dem_path).data
    else:
        dem = np.zeros((ny, nx), dtype=np.float32)
    if layover_shadow_mask_path is not None:
        shadow_mask = xr.open_dataarray(layover_shadow_mask_path).data
    else:
        shadow_mask = np.zeros((ny, nx), dtype=np.int8)
    geometry_dsets = {
        "height": (np.float32, (ny, nx), dem.data),  # DEM heights
        "incidenceAngle": (np.float32, (ny, nx), incidence_angle),
        "azimuthAngle": (np.float32, (ny, nx), azimuth_angle),
        "shadowMask": (np.int8, (ny, nx), shadow_mask),
    }

    _write_hdf5(outdir / "geometryGeo.h5", geometry_dsets, geometry_meta)
    print("Created geometryGeo.h5.")


def disp_nc_to_mintpy(
    reformatted_nc_path: Path,
    /,
    sample_disp_nc: Path,
    geometry_dir: Path | str | None = None,
    los_enu_path: Path | str | None = None,
    dem_path: Path | str | None = None,
    layover_shadow_mask_path: Path | str | None = None,
    outdir: Path = Path("mintpy"),
    virtual: bool = False,
    reliability_threshold: float = 0.90,
) -> None:
    """Convert a reformatted DISP-S1 NetCDF file to MintPy inputs.

    Parameters
    ----------
    reformatted_nc_path : Path
        Path to the reformatted DISP-S1 NetCDF file.
        Result from `opera-utils disp-s1-reformat`
    sample_disp_nc : Path
        Path to one of the sample DISP-S1 NetCDF files.
    geometry_dir : Path | str, optional
        Path to a directory containing the LOS ENU, DEM, and layover/shadow mask files.
        Alternative to specifying the individual files separately.
    los_enu_path : Path | str, optional
        Path to the line-of-sight (LOS) 3-band east, north, up file.
    dem_path : Path | str, optional
        Path to the DEM file.
    layover_shadow_mask_path : Path | str, optional
        Path to the layover/shadow mask file.
    outdir : Path, optional
        Output directory for the MintPy inputs.
        Default is "mintpy".
    virtual : bool, optional
        If True, uses the virtual dataset (VDS) feature of HDF5
        to avoid copying data from the NetCDF's to the mintpy files.
        See https://docs.h5py.org/en/stable/vds.html
        VDS can be quicker to run and use less disk space, but it may
        cause problems when reading the files from other directories
        or moving the `timeseries.h5` file.
        Default is False
    reliability_threshold : float, optional
        Pixel must be valid in >= `reliability_threshold` of epochs to be kept.
        Default is 0.90


    """
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load existing NetCDF
    ds = xr.open_dataset(reformatted_nc_path)
    prod = DispProduct.from_filename(sample_disp_nc)

    disp = ds["displacement"].transpose("time", "y", "x").data.astype(np.float32)
    times = ds["time"].data  # np.datetime64[…]
    dates = np.array(
        [t.astype("datetime64[D]").astype(str).replace("-", "") for t in times],
        dtype="S8",
    )

    ny, nx = disp.shape[1:]
    nt = disp.shape[0]

    ts_meta = _mintpy_metadata(prod)
    if virtual:
        # Write timeseries.h5 as a virtual link to the /displacement stack
        _write_timeseries_vds(
            out_path=outdir / "timeseries.h5",
            src_nc=reformatted_nc_path,
            ds=ds,
            meta=ts_meta,
            dates=dates,
        )
        print("Done with timeseries.h5 (virtual link)")
    else:
        # Copy displacement data directly to avoid VDS issues with MintPy
        ts_dsets = {
            "date": (dates.dtype, dates.shape, dates),
            "bperp": (
                np.float32,
                ds["perpendicular_baseline"].shape,
                ds["perpendicular_baseline"].data.astype(np.float32),
            ),
            "timeseries": (np.float32, disp.shape, disp),
        }
        _write_hdf5(outdir / "timeseries.h5", ts_dsets, ts_meta)
        print("Done with timeseries.h5 (copied data)")

    # 3. avgSpatialCoh.h5  (use average_temporal_coherence layer)
    if "average_temporal_coherence" in ds:
        avg_coh = ds["average_temporal_coherence"].data.astype(np.float32)
    else:
        avg_coh = ds["temporal_coherence"].mean("time").data.astype(np.float32)

    # TODO: fix these to work in blocks!
    coh_meta = ts_meta | {"FILE_TYPE": "mask", "UNIT": "1"}
    coh_dsets = {"avgSpatialCoh": (np.float32, (ny, nx), avg_coh)}
    _write_hdf5(outdir / "avgSpatialCoh.h5", coh_dsets, coh_meta)
    print("Done with avgSpatialCoh.h5")

    # Create reliability products with 90% threshold
    create_reliability_mask(
        da_recommended=ds["recommended_mask"],
        out_dir=outdir,
        reliability_threshold=reliability_threshold,
    )

    # TODO: Use dolphin, or mintpy, or xarray to do this in batches
    # 4. velocity.h5  (OLS fit, slope is m / yr)
    tdecimal = _decimal_year(times)  # shape (nt,)
    A = np.vstack([tdecimal, np.ones_like(tdecimal)]).T  # nt, 2

    # reshape to (nt, ny*nx), fit in one go
    y = disp.reshape(nt, -1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)  # 2, (ny*nx)
    vel = coef[0, :].reshape(ny, nx).astype(np.float32)  # slope in m/yr

    vel_meta = ts_meta | {
        "FILE_TYPE": "velocity",
        "UNIT": "m/year",
        "REF_DATE": dates[0].decode(),
        "START_DATE": dates[0].decode(),
        "END_DATE": dates[-1].decode(),
    }
    vel_dsets = {"velocity": (np.float32, (ny, nx), vel)}
    _write_hdf5(outdir / "velocity.h5", vel_dsets, vel_meta)
    print("Done with velocity.h5")

    # geometryGeo.h5
    # Download UTM DEM/LOS ENU/Layover shadow mask in `opera_utils.disp._download.py`
    if geometry_dir:
        los_enu_path = next(Path(geometry_dir).glob("*los_enu.tif"))
        dem_path = next(Path(geometry_dir).glob("*dem.tif"))
        layover_shadow_mask_path = next(
            Path(geometry_dir).glob("*layover_shadow_mask.tif")
        )

    if los_enu_path:
        create_static_layers(
            los_enu_path,
            ts_meta,
            dem_path=dem_path,
            layover_shadow_mask_path=layover_shadow_mask_path,
            outdir=outdir,
        )
    else:
        print("No LOS-ENU path provided, skipping geometryGeo.h5")

    print(f"\nAll done : {outdir.resolve()}\n")


if __name__ == "__main__":
    import tyro

    tyro.cli(disp_nc_to_mintpy)
