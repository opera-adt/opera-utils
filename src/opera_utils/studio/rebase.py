"""DISP-S1 rebasing and point re-referencing for the studio.

Two steps turn a folder of DISP-S1 subsets into a series a user can read:

1. **Temporal rebasing** (``reformat_stack``): each DISP-S1 product is relative
   to its ministack's reference date, which moves forward over time. The stack
   adds the displacement at each reference change ("crossover") so every epoch
   is relative to the first date. ``raw_series`` reads the products as they are,
   so the UI can show the saw-tooth before rebasing next to the result.
2. **Spatial reference**: ``reformat_stack`` subtracts the median of the
   high-coherence pixels from every epoch (an area, not a place). As in the
   DISP portal, ``pick_reference`` chooses one point instead: the most coherent
   spot (average temporal coherence over 5 x 5 pixels) among pixels valid in
   (nearly) every epoch, or the best pixel within ~300 m of a point the user
   gave. Its series (median of a 5 x 5 window) is subtracted from every pixel,
   and its slope from the velocity.

Nothing is rewritten when the reference changes: the stack keeps the area
reference and the point is applied on read (``Frame.series``,
``Frame.velocity``) and on export (``Frame.export_geotiffs``).
"""

from __future__ import annotations

import json
import struct
import zlib
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

WINDOW = 5  # the reference is the median of a 5 x 5 window (150 m)
VALID_SHARE = 0.98  # reference pixel valid in this share of the best pixel's epochs
SNAP_PIXELS = 10  # a user point snaps to the best pixel within ~300 m
YEAR_S = 365.25 * 86400

# Diverging palette (blue = toward the satellite, red = away), light centre.
_PALETTE = np.array(
    [
        (103, 0, 31),
        (178, 24, 43),
        (214, 96, 77),
        (244, 165, 130),
        (247, 247, 247),
        (146, 197, 222),
        (67, 147, 195),
        (33, 102, 172),
        (5, 48, 97),
    ],
    dtype=np.float64,
)


# -- pure helpers (tested directly) ------------------------------------------
def years_since_first(times: np.ndarray) -> np.ndarray:
    """Return decimal years since the first of ``times``."""
    t = np.asarray(times).astype("datetime64[s]").astype(np.float64)
    return (t - t[0]) / YEAR_S


def fit_velocity(stack: np.ndarray, years: np.ndarray, min_epochs: int = 3):
    """Per-pixel least-squares slope of ``stack`` (time, y, x) against ``years``.

    NaN epochs are skipped pixel by pixel. Returns (slope, valid_epochs); slope
    is NaN where fewer than ``min_epochs`` epochs are valid.
    """
    d = np.asarray(stack, dtype=np.float64)
    t = np.asarray(years, dtype=np.float64)[:, None, None]
    ok = np.isfinite(d)
    n = ok.sum(axis=0)
    dz = np.where(ok, d, 0.0)
    tz = np.where(ok, t, 0.0)
    st, sd = tz.sum(0), dz.sum(0)
    stt, std = (tz * tz).sum(0), (tz * dz).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = n * stt - st * st
        slope = (n * std - st * sd) / den
    slope[(n < min_epochs) | (den == 0)] = np.nan
    return slope.astype(np.float32), n.astype(np.int32)


def slope_of(series: np.ndarray, years: np.ndarray) -> float:
    """Return the least-squares slope of one ``series`` against ``years``."""
    s, _ = fit_velocity(np.asarray(series)[:, None, None], years, min_epochs=2)
    return float(s[0, 0])


def pick_reference_pixel(
    coherence: np.ndarray,
    valid_epochs: np.ndarray,
    near: tuple[int, int] | None = None,
) -> tuple[int, int, str]:
    """Row, column and method of the reference pixel (see module docstring)."""
    from scipy.ndimage import uniform_filter

    coh = np.asarray(coherence, np.float64)
    valid = np.asarray(valid_epochs, np.float64)
    ok = np.isfinite(coh) & (valid >= VALID_SHARE * np.nanmax(valid))
    if not ok.any():
        msg = "no pixel is observed in (nearly) every epoch"
        raise ValueError(msg)
    filled = np.where(ok, coh, 0.0)
    share = uniform_filter(ok.astype(float), WINDOW, mode="constant")
    mean = uniform_filter(filled, WINDOW, mode="constant") / np.maximum(share, 1e-6)
    score = np.where(ok & (share > 0.8), mean, -np.inf)
    if not np.isfinite(score).any():  # tiny or patchy areas: drop the window rule
        score = np.where(ok, coh, -np.inf)
    method = "most coherent spot"
    if near is not None:
        r0, c0 = near
        if not (0 <= r0 < coh.shape[0] and 0 <= c0 < coh.shape[1]):
            msg = "the reference point is outside the downloaded area"
            raise ValueError(msg)
        h = SNAP_PIXELS
        win = np.full_like(score, -np.inf)
        sl = (slice(max(r0 - h, 0), r0 + h + 1), slice(max(c0 - h, 0), c0 + h + 1))
        win[sl] = score[sl]
        if not np.isfinite(win).any():
            msg = "no well-observed pixel within ~300 m of the reference point"
            raise ValueError(msg)
        score, method = win, "user point"
    row, col = (int(v) for v in np.unravel_index(np.argmax(score), score.shape))
    return row, col, method


def window_median(
    cube: np.ndarray, row: int, col: int, size: int = WINDOW
) -> np.ndarray:
    """Median over a size x size window around (row, col), per epoch."""
    h = size // 2
    w = np.asarray(
        cube[:, max(row - h, 0) : row + h + 1, max(col - h, 0) : col + h + 1],
        np.float64,
    )
    with np.errstate(all="ignore"), _Quiet():
        return np.nanmedian(w.reshape(w.shape[0], -1), axis=1)


class _Quiet:
    """Silence RuntimeWarnings (all-NaN slices and the like) inside the block."""

    def __enter__(self):
        import warnings

        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def colorize(
    values: np.ndarray, vmax: float, alpha: np.ndarray | None = None
) -> np.ndarray:
    """RGBA uint8 image of ``values`` on the diverging palette, ±vmax."""
    v = np.asarray(values, np.float64)
    finite = np.isfinite(v)
    x = np.clip((np.where(finite, v, 0) / max(vmax, 1e-12) + 1) / 2, 0, 1)
    # palette runs negative (red, away) -> positive (blue, toward)
    pos = x * (len(_PALETTE) - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, len(_PALETTE) - 2)
    f = (pos - i0)[..., None]
    rgb = _PALETTE[i0] * (1 - f) + _PALETTE[i0 + 1] * f
    a = np.where(finite, 255, 0).astype(np.float64)
    if alpha is not None:
        a = a * np.asarray(alpha, np.float64)
    return np.dstack([rgb, a]).round().astype(np.uint8)


def encode_png(rgba: np.ndarray) -> bytes:
    """Minimal RGBA PNG encoder (no Pillow needed)."""
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _downsample(a: np.ndarray, max_px: int) -> np.ndarray:
    step = int(np.ceil(max(a.shape[-2:]) / max_px)) if max(a.shape[-2:]) > max_px else 1
    return a[..., ::step, ::step]


# -- one downloaded frame ----------------------------------------------------
class Frame:
    """A DISP-S1 frame of a job: ``<job>/frames/Fxxxxx`` with stack.zarr inside."""

    def __init__(self, frame_dir: Path, download_dir: Path | None = None):
        """Open the frame in ``frame_dir``; source files may be in ``download_dir``."""
        self.dir = Path(frame_dir)
        self.stack_path = self.dir / "stack.zarr"
        self.download_dir = Path(download_dir) if download_dir else None

    # dataset and grid
    @cached_property
    def ds(self):
        import xarray as xr

        return xr.open_zarr(self.stack_path, consolidated=False)

    @cached_property
    def times(self) -> np.ndarray:
        return np.asarray(self.ds["time"].values).astype("datetime64[s]")

    @cached_property
    def years(self) -> np.ndarray:
        return years_since_first(self.times)

    @cached_property
    def shape(self) -> tuple[int, int]:
        return tuple(self.ds["displacement"].shape[-2:])

    @cached_property
    def crs(self):
        from pyproj import CRS

        sr = self.ds.get("spatial_ref")
        if sr is not None and "crs_wkt" in sr.attrs:
            return CRS.from_wkt(sr.attrs["crs_wkt"])
        return CRS.from_user_input(self.meta["crs"])

    @cached_property
    def transform(self):
        from affine import Affine

        x = np.asarray(self.ds["x"].values, np.float64)
        y = np.asarray(self.ds["y"].values, np.float64)
        dx = float(x[1] - x[0]) if len(x) > 1 else 30.0
        dy = float(y[1] - y[0]) if len(y) > 1 else -30.0
        return Affine(dx, 0, x[0] - dx / 2, 0, dy, y[0] - dy / 2)

    def lonlat_to_rowcol(self, lon: float, lat: float) -> tuple[int, int]:
        from pyproj import Transformer

        x, y = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True).transform(
            lon, lat
        )
        col, row = _apply(~self.transform, x, y)
        return int(np.floor(row)), int(np.floor(col))

    def rowcol_to_lonlat(self, row: int, col: int) -> tuple[float, float]:
        from pyproj import Transformer

        x, y = _apply(self.transform, col + 0.5, row + 0.5)
        lon, lat = Transformer.from_crs(
            self.crs, "EPSG:4326", always_xy=True
        ).transform(x, y)
        return float(lon), float(lat)

    def corners(self) -> list[list[float]]:
        """Return lon/lat corners: top-left, top-right, bottom-right, bottom-left."""
        from pyproj import Transformer

        h, w = self.shape
        tf = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        pts = [
            _apply(self.transform, c, r) for c, r in ((0, 0), (w, 0), (w, h), (0, h))
        ]
        return [list(map(float, tf.transform(x, y))) for x, y in pts]

    # analysis (velocity, coherence, valid epochs), computed once
    @property
    def analysis_path(self) -> Path:
        return self.dir / "analysis.npz"

    def compute_analysis(self, block_rows: int = 256) -> None:
        da = self.ds["displacement"]
        h, w = self.shape
        vel = np.full((h, w), np.nan, np.float32)
        valid = np.zeros((h, w), np.int32)
        for r0 in range(0, h, block_rows):
            block = np.asarray(da[:, r0 : r0 + block_rows, :].values, np.float32)
            v, n = fit_velocity(block, self.years)
            vel[r0 : r0 + block_rows], valid[r0 : r0 + block_rows] = v, n
        if "average_temporal_coherence" in self.ds:
            coh = np.asarray(self.ds["average_temporal_coherence"].values, np.float32)
        else:
            coh = np.where(valid > 0, 1.0, np.nan).astype(np.float32)
        last = np.asarray(da[-1].values, np.float32)
        np.savez_compressed(
            self.analysis_path, velocity=vel, coherence=coh, valid=valid, last=last
        )
        self.__dict__.pop("analysis", None)

    @cached_property
    def analysis(self) -> dict[str, np.ndarray]:
        if not self.analysis_path.exists():
            self.compute_analysis()
        with np.load(self.analysis_path) as z:
            return {k: z[k] for k in z.files}

    # the reference point
    @property
    def reference_path(self) -> Path:
        return self.dir / "reference.json"

    @property
    def reference(self) -> dict | None:
        if self.reference_path.exists():
            return json.loads(self.reference_path.read_text())
        return None

    def set_reference(
        self, lonlat: tuple[float, float] | None = None, mode: str = "auto"
    ) -> dict:
        """Pick and save the reference (``mode``: auto, point or area)."""
        if mode == "area":
            rec: dict[str, Any] = {
                "method": "median of high-coherence pixels",
                "mode": "area",
            }
        else:
            a = self.analysis
            near = (
                self.lonlat_to_rowcol(*lonlat) if (mode == "point" and lonlat) else None
            )
            row, col, method = pick_reference_pixel(a["coherence"], a["valid"], near)
            series = window_median(self.ds["displacement"], row, col)
            if not np.isfinite(series).all():
                msg = "the reference window has epochs without data"
                raise ValueError(msg)
            lon, lat = self.rowcol_to_lonlat(row, col)
            rec = {
                "mode": "point",
                "method": method,
                "lon": round(lon, 6),
                "lat": round(lat, 6),
                "row": row,
                "col": col,
                "window_pixels": WINDOW,
                "coherence": round(float(a["coherence"][row, col]), 3),
                "slope_m_yr": round(slope_of(series, self.years), 6),
                "series": [round(float(v), 6) for v in series],
            }
            if lonlat:
                rec["requested"] = [round(lonlat[0], 6), round(lonlat[1], 6)]
        self.reference_path.write_text(json.dumps(rec, indent=1))
        return rec

    def _ref_series(self) -> np.ndarray:
        ref = self.reference
        if ref and ref.get("mode") == "point":
            return np.asarray(ref["series"], np.float64)
        return np.zeros(len(self.times))

    def _ref_slope(self) -> float:
        ref = self.reference
        return float(ref["slope_m_yr"]) if ref and ref.get("mode") == "point" else 0.0

    # reads for the UI
    def series(self, lon: float, lat: float) -> dict:
        """Series at a point: rebased (area reference), re-referenced, and raw."""
        row, col = self.lonlat_to_rowcol(lon, lat)
        h, w = self.shape
        if not (0 <= row < h and 0 <= col < w):
            msg = "this point is outside the downloaded area"
            raise ValueError(msg)
        rebased = np.asarray(self.ds["displacement"][:, row, col].values, np.float64)
        referenced = rebased - self._ref_series()
        out = {
            "lon": lon,
            "lat": lat,
            "row": row,
            "col": col,
            "dates": [str(t)[:19] + "Z" for t in self.times],
            "rebased": _clean(rebased),
            "referenced": _clean(referenced),
            "velocity_m_yr": _num(slope_of(referenced, self.years)),
            "coherence": _num(self.analysis["coherence"][row, col]),
            "reference": {
                k: v for k, v in (self.reference or {}).items() if k != "series"
            },
        }
        raw = self.raw_series(row, col)
        if raw:
            ref = self.reference
            if ref and ref.get("mode") == "point":
                ref_raw = self.raw_series(ref["row"], ref["col"], window=True)
                if ref_raw:
                    raw["values"] = _clean(
                        np.asarray(raw["values"], float)
                        - np.asarray(ref_raw["values"], float)
                    )
            out["raw"] = raw
        return out

    def raw_series(self, row: int, col: int, window: bool = False) -> dict | None:
        """Displacement as published (moving reference dates), from the subsets."""
        import h5py

        from opera_utils.disp._product import DispProduct

        if self.download_dir is None or not self.download_dir.exists():
            return None
        files = sorted(self.download_dir.glob("*.nc"))
        rows = []
        for f in files:
            try:
                p = DispProduct.from_filename(f)
                with h5py.File(f, "r") as hf:
                    d = hf["displacement"]
                    if window:
                        h = WINDOW // 2
                        win = np.asarray(
                            d[
                                max(row - h, 0) : row + h + 1,
                                max(col - h, 0) : col + h + 1,
                            ],
                            np.float64,
                        )
                        with _Quiet():
                            v = float(np.nanmedian(win))
                    else:
                        v = float(d[row, col])
            except (OSError, KeyError, ValueError, IndexError):
                continue
            rows.append((p.secondary_datetime, p.reference_datetime, v))
        if not rows:
            return None
        rows.sort()
        return {
            "dates": [_iso(r[0]) for r in rows],
            "reference_dates": [_iso(r[1]) for r in rows],
            "values": _clean(np.array([r[2] for r in rows])),
        }

    def velocity(self) -> np.ndarray:
        return self.analysis["velocity"] - np.float32(self._ref_slope())

    def suggested_vmax(self) -> float:
        v = self.velocity()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 0.01
        return float(max(np.percentile(np.abs(v - np.median(v)), 98), 0.002))

    def velocity_png(
        self, vmax: float | None = None, min_coherence: float = 0.0, max_px: int = 2048
    ) -> bytes:
        vel = self.velocity()
        coh = self.analysis["coherence"]
        vmax = vmax or self.suggested_vmax()
        alpha = np.where(np.nan_to_num(coh, nan=0) >= min_coherence, 1.0, 0.0)
        img = colorize(_downsample(vel, max_px), vmax, _downsample(alpha, max_px))
        return encode_png(img)

    def meta(self) -> dict:
        ref = self.reference or {}
        v = self.velocity()
        finite = v[np.isfinite(v)]
        return {
            "shape": list(self.shape),
            "corners": self.corners(),
            "dates": [str(t)[:10] for t in self.times],
            "n_epochs": len(self.times),
            "crs": self.crs.to_string(),
            "vmax": self.suggested_vmax(),
            "velocity_median_m_yr": _num(np.median(finite)) if finite.size else None,
            "reference": {k: v for k, v in ref.items() if k != "series"},
        }

    def export_geotiffs(self, out_dir: Path | None = None) -> list[Path]:
        """Write velocity, displacement and coherence GeoTIFFs (current reference)."""
        import rasterio

        out_dir = Path(out_dir or self.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        a = self.analysis
        last = a["last"] - np.float32(self._ref_series()[-1])
        t0, t1 = (str(t)[:10].replace("-", "") for t in (self.times[0], self.times[-1]))
        span = f"{t0}_{t1}"
        layers = {
            f"velocity_m_per_yr_{span}.tif": self.velocity(),
            f"cumulative_displacement_m_{span}.tif": last,
            "average_temporal_coherence.tif": a["coherence"],
        }
        h, w = self.shape
        written = []
        for name, data in layers.items():
            path = out_dir / name
            profile = {
                "driver": "GTiff",
                "width": w,
                "height": h,
                "count": 1,
                "dtype": "float32",
                "crs": self.crs.to_wkt(),
                "transform": self.transform,
                "nodata": np.nan,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
            if w < 256 or h < 256:
                profile.pop("blockxsize")
                profile.pop("blockysize")
                profile["tiled"] = False
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(np.asarray(data, np.float32), 1)
                ref = self.reference or {}
                dst.update_tags(
                    reference=json.dumps(
                        {k: v for k, v in ref.items() if k != "series"}
                    ),
                    first_date=str(self.times[0])[:10],
                    last_date=str(self.times[-1])[:10],
                )
            written.append(path)
        return written


def _apply(t, x: float, y: float) -> tuple[float, float]:
    """Apply an affine transform to one point.

    Written out instead of ``t * (x, y)``: affine 3 deprecates ``*`` for ``@``,
    which affine 2 does not have.
    """
    return t.a * x + t.b * y + t.c, t.d * x + t.e * y + t.f


def _iso(d: datetime) -> str:
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(v) -> float | None:
    v = float(v)
    return round(v, 6) if np.isfinite(v) else None


def _clean(a: np.ndarray) -> list[float | None]:
    return [
        round(float(v), 6) if np.isfinite(v) else None
        for v in np.asarray(a, np.float64)
    ]
