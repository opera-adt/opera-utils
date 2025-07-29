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
            if len(shape) > 2:
                chunks: tuple[int, ...] | None = DEFAULT_CHUNKS
            elif len(shape) == 2:
                chunks = DEFAULT_CHUNKS[-2:]
            else:
                chunks = None
                compression = None
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


def disp_nc_to_mintpy(
    reformatted_nc_path: Path,
    /,
    sample_disp_nc: Path,
    outdir: Path = Path("mintpy"),
) -> None:
    """Convert a reformatted DISP-S1 NetCDF file to MintPy inputs.

    Parameters
    ----------
    reformatted_nc_path : Path
        Path to the reformatted DISP-S1 NetCDF file.
        Result from `opera-utils disp-s1-reformat`
    sample_disp_nc : Path
        Path to one of the sample DISP-S1 NetCDF files.
    outdir : Path, optional
        Output directory for the MintPy inputs.
        Default is "mintpy".

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

    # 2. Write timeseries.h5 as a virtual link to the /displacement stack
    ts_meta = _mintpy_metadata(prod)
    _write_timeseries_vds(
        out_path=outdir / "timeseries.h5",
        src_nc=reformatted_nc_path,
        ds=ds,
        meta=ts_meta,
        dates=dates,
    )
    print("Done with timeseries.h5 (virtual link)")

    # 3. avgSpatialCoh.h5  (use average_temporal_coherence layer)
    if "average_temporal_coherence" in ds:
        avg_coh = ds["average_temporal_coherence"].data.astype(np.float32)
    else:
        avg_coh = ds["temporal_coherence"].mean("time").data.astype(np.float32)

    coh_meta = ts_meta | {"FILE_TYPE": "mask", "UNIT": "1"}
    coh_dsets = {"avgSpatialCoh": (np.float32, (ny, nx), avg_coh)}
    _write_hdf5(outdir / "avgSpatialCoh.h5", coh_dsets, coh_meta)
    print("Done with avgSpatialCoh.h5")

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

    # 5. geometryGeo.h5
    # TODO: implement once public DISP-S1-STATIC layers are available
    # Download UTM DEM/LOS ENU/Layover shadow mask in `opera_utils.disp._download.py`

    print(f"\nAll done : {outdir.resolve()}\n")


if __name__ == "__main__":
    import tyro

    tyro.cli(disp_nc_to_mintpy)
