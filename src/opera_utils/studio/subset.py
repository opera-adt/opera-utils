"""Crop OPERA CSLC-S1 (and CSLC-S1-STATIC) HDF5 files to a lon/lat box.

These products keep their rasters in ``/data`` on a UTM grid described by
``/data/x_coordinates``, ``/data/y_coordinates`` and the EPSG code in
``/data/projection``. Every ``/data`` raster on that grid is cut to the window
covering the box; everything else (metadata, identification, quality, small
``/data`` arrays) is copied unchanged, so the result reads like the original.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import h5py
import numpy as np
from pyproj import Transformer


class SubsetError(ValueError):
    """The file is not a gridded OPERA HDF5, or the box misses it."""


def _epsg(data: h5py.Group) -> int:
    if "projection" not in data:
        msg = "no /data/projection"
        raise SubsetError(msg)
    proj = data["projection"]
    for key in ("epsg_code", "epsg"):
        if key in proj.attrs:
            return int(np.asarray(proj.attrs[key]).ravel()[0])
    return int(np.asarray(proj[()]).ravel()[0])


def window_for_bbox(
    x: np.ndarray,
    y: np.ndarray,
    epsg: int,
    bbox: tuple[float, float, float, float],
    pad: int = 2,
) -> tuple[slice, slice]:
    """Return the row/column slices of the grid (centres ``x``, ``y``) over ``bbox``."""
    west, south, east, north = bbox
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    # Densify the edges: a lon/lat box is curved in UTM.
    t = np.linspace(0, 1, 21)
    lons = np.concatenate(
        [
            west + (east - west) * t,
            np.full_like(t, east),
            east - (east - west) * t,
            np.full_like(t, west),
        ]
    )
    lats = np.concatenate(
        [
            np.full_like(t, south),
            south + (north - south) * t,
            np.full_like(t, north),
            north - (north - south) * t,
        ]
    )
    xs, ys = tf.transform(lons, lats)
    cols = np.nonzero((x >= min(xs)) & (x <= max(xs)))[0]
    rows = np.nonzero((y >= min(ys)) & (y <= max(ys)))[0]
    if cols.size == 0 or rows.size == 0:
        msg = "the area does not overlap this file"
        raise SubsetError(msg)
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, len(y))
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, len(x))
    return slice(int(r0), int(r1)), slice(int(c0), int(c1))


def _copy_attrs(src, dst) -> None:
    for k, v in src.attrs.items():
        with contextlib.suppress(TypeError, OSError):
            dst.attrs[k] = v


def subset_h5(
    src_path: Path, out_path: Path, bbox: tuple[float, float, float, float]
) -> dict:
    """Write the part of ``src_path`` in ``bbox`` to ``out_path``; return the window."""
    out_path = Path(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with h5py.File(src_path, "r") as src:
        if "data" not in src or "x_coordinates" not in src["data"]:
            msg = (
                f"{Path(src_path).name}: not a gridded OPERA HDF5 (/data/x_coordinates"
                " missing)"
            )
            raise SubsetError(msg)
        data = src["data"]
        x = np.asarray(data["x_coordinates"][()], np.float64)
        y = np.asarray(data["y_coordinates"][()], np.float64)
        epsg = _epsg(data)
        rows, cols = window_for_bbox(x, y, epsg, bbox)
        ny, nx = len(y), len(x)
        with h5py.File(tmp, "w") as dst:
            _copy_attrs(src, dst)
            for name in src:
                if name != "data":
                    src.copy(name, dst, name=name)
            out = dst.create_group("data")
            _copy_attrs(data, out)
            for name, obj in data.items():
                if not isinstance(obj, h5py.Dataset):
                    data.copy(name, out, name=name)
                    continue
                if name == "x_coordinates":
                    arr = obj[cols]
                elif name == "y_coordinates":
                    arr = obj[rows]
                elif obj.ndim == 2 and obj.shape == (ny, nx):
                    arr = obj[rows, cols]
                else:
                    data.copy(name, out, name=name)
                    continue
                kw = {}
                if arr.ndim == 2 and min(arr.shape) >= 64:
                    kw = {
                        "chunks": (min(256, arr.shape[0]), min(256, arr.shape[1])),
                        "compression": "gzip",
                        "compression_opts": 4,
                        "shuffle": True,
                    }
                ds = out.create_dataset(name, data=arr, **kw)
                _copy_attrs(obj, ds)
            dst.attrs["studio_subset_window"] = np.array(
                [rows.start, rows.stop, cols.start, cols.stop]
            )
    tmp.replace(out_path)
    return {
        "rows": [rows.start, rows.stop],
        "cols": [cols.start, cols.stop],
        "epsg": epsg,
    }
