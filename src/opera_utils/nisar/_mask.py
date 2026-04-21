"""Extract validity masks from NISAR GSLC and GUNW HDF5 products."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import rioxarray as rxr
import xarray as xr


def get_gslc_mask(
    gslc_path: str | Path,
    frequency: str = "frequencyA",
    output_path: str | Path | None = None,
) -> xr.DataArray:
    """Extract the validity mask from a NISAR GSLC product.

    The GSLC mask encodes the subswath number of valid samples.
    Valid pixels have value >= 1; invalid pixels are 0; fill (outside
    radar extent) is 255.  This function returns a boolean mask where
    ``True`` means valid (subswath value >= 1).

    Parameters
    ----------
    gslc_path : str or Path
        Path to a NISAR GSLC HDF5 file.
    frequency : str
        Frequency layer to read (``"frequencyA"`` or ``"frequencyB"``).
    output_path : str or Path, optional
        If provided, write the boolean mask as a GeoTIFF (uint8, 1=valid).

    Returns
    -------
    xr.DataArray
        Boolean DataArray (True = valid pixel) on the GSLC native grid,
        with CRS and spatial coordinates attached.

    """
    gslc_path = Path(gslc_path)
    grp_path = f"science/LSAR/GSLC/grids/{frequency}"

    with h5py.File(gslc_path) as f:
        grp = f[grp_path]
        raw = grp["mask"][:]
        x_coords = grp["xCoordinates"][:]
        y_coords = grp["yCoordinates"][:]
        epsg = int(grp["projection"][()])

    # valid = subswath value >= 1 (0 = invalid focus, 255 = fill/nodata)
    valid = (raw >= 1) & (raw != 255)

    da = xr.DataArray(
        valid.astype(np.uint8),
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
        attrs={"long_name": "valid sample mask", "flag_values": "0,1",
               "flag_meanings": "invalid valid"},
    ).rio.write_crs(f"EPSG:{epsg}")

    if output_path is not None:
        da.rio.write_nodata(255, inplace=True)
        da.rio.to_raster(output_path, dtype="uint8", compress="deflate")

    return da


GunwLayer = Literal["wrappedInterferogram", "unwrappedInterferogram", "pixelOffsets"]


def get_gunw_mask(
    gunw_path: str | Path,
    layer: GunwLayer = "unwrappedInterferogram",
    frequency: str = "frequencyA",
    output_path: str | Path | None = None,
) -> xr.DataArray:
    """Extract a validity mask from a NISAR GUNW product.

    The GUNW mask is a 3-digit decimal bitfield (uint8):
    - hundreds digit: water flag (1 = water, 0 = non-water)
    - tens digit: reference RSLC subswath (0 = invalid)
    - units digit: secondary RSLC subswath (0 = invalid)

    A pixel is considered valid when both subswath digits are non-zero
    and the water flag is 0.  Fill value is 255.

    Parameters
    ----------
    gunw_path : str or Path
        Path to a NISAR GUNW HDF5 file.
    layer : {"unwrappedInterferogram", "wrappedInterferogram", "pixelOffsets"}
        Which grid layer's mask to read.  Default is ``"unwrappedInterferogram"``.
    frequency : str
        Frequency layer to read (currently only ``"frequencyA"`` in GUNW).
    output_path : str or Path, optional
        If provided, write the boolean mask as a GeoTIFF (uint8, 1=valid).

    Returns
    -------
    xr.DataArray
        Boolean DataArray (True = valid, non-water pixel) with CRS and
        spatial coordinates attached.

    """
    gunw_path = Path(gunw_path)
    grp_path = f"science/LSAR/GUNW/grids/{frequency}/{layer}"

    with h5py.File(gunw_path) as f:
        grp = f[grp_path]
        raw = grp["mask"][:]
        x_coords = grp["xCoordinates"][:]
        y_coords = grp["yCoordinates"][:]
        epsg = int(grp["projection"][()])

    # Decode 3-digit decimal encoding
    water_flag = raw // 100          # hundreds digit
    ref_subswath = (raw % 100) // 10  # tens digit
    sec_subswath = raw % 10           # units digit
    fill = raw == 255

    valid = (water_flag == 0) & (ref_subswath != 0) & (sec_subswath != 0) & ~fill

    da = xr.DataArray(
        valid.astype(np.uint8),
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
        attrs={"long_name": "valid sample mask", "flag_values": "0,1",
               "flag_meanings": "invalid valid",
               "source_layer": layer},
    ).rio.write_crs(f"EPSG:{epsg}")

    if output_path is not None:
        da.rio.write_nodata(255, inplace=True)
        da.rio.to_raster(output_path, dtype="uint8", compress="deflate")

    return da
