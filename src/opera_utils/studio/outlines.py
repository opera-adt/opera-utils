"""Reference outlines and archive depth for the studio map.

Three layers:

* ``frames``: OPERA DISP-S1 frames, North America (opera-utils burst database);
* ``bursts``: Sentinel-1 burst IDs, North America (burst database, ~35k);
* ``nisar``: NISAR track-frames over land, worldwide, from the NISAR track-frame
  GeoPackage (``NISAR_TrackFrame_*.gpkg``: the real, tilted frame outlines and
  their pass direction). Without it, the opera-utils frame-to-bounds file is
  used: only North America, keyed by ``(track - 1) * 176 + (frame - 1)``, and
  only each frame's map-projected bounding box, so outlines are approximate
  and have no direction.

Polygons crossing ±180° are stored cut in two; they are unwrapped into one
continuous polygon (see ``geo.py``) and queries also look one turn east and
west, so outlines stay whole when the map is panned across the antimeridian.

``depth`` counts the granules each outline has in CMR (one ``page_size=0``
request each, cached); ``timeline`` lists the acquisition dates of one outline.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from shapely import STRtree
from shapely.geometry import Polygon, box, mapping, shape

from . import geo

logger = logging.getLogger("opera_utils")

CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
LAYERS = ("frames", "bursts", "nisar")
NISAR_FRAMES_PER_TRACK = 176
DEPTH_TTL_S = 12 * 3600
COLLECTIONS = {
    "disp": "OPERA_L3_DISP-S1_V1",
    "disp-static": "OPERA_L3_DISP-S1-STATIC_V1",
    "cslc": "OPERA_L2_CSLC-S1_V1",
    "cslc-static": "OPERA_L2_CSLC-S1-STATIC_V1",
    "gslc": "NISAR_L2_GSLC_BETA_V1",
    "rslc": "NISAR_L1_RSLC_PROVISIONAL_V1",
    "gunw": "NISAR_L2_GUNW_BETA_V1",
}


class OutlineError(ValueError):
    """Unknown layer or outline id."""


def nisar_index_to_track_frame(idx: int) -> tuple[int, int]:
    """NISAR frame-database key -> (track, frame), both 1-based."""
    return idx // NISAR_FRAMES_PER_TRACK + 1, idx % NISAR_FRAMES_PER_TRACK + 1


def projected_box_to_lonlat(epsg: int, xmin, ymin, xmax, ymax, n: int = 12) -> Polygon:
    """Return a map-projected box as a lon/lat polygon (edges densified)."""
    from pyproj import Transformer

    t = np.linspace(0, 1, n)
    xs = np.concatenate(
        [
            xmin + (xmax - xmin) * t,
            np.full(n, xmax),
            xmax - (xmax - xmin) * t,
            np.full(n, xmin),
        ]
    )
    ys = np.concatenate(
        [
            np.full(n, ymin),
            ymin + (ymax - ymin) * t,
            np.full(n, ymax),
            ymax - (ymax - ymin) * t,
        ]
    )
    lon, lat = Transformer.from_crs(
        f"EPSG:{epsg}", "EPSG:4326", always_xy=True
    ).transform(xs, ys)
    return Polygon(zip(lon, lat))


def find_nisar_gpkg(explicit: Path | str | None = None) -> Path | None:
    """Find the NISAR track-frame GeoPackage.

    Tries ``explicit``, then ``$OPERA_STUDIO_NISAR_GPKG``, then the newest
    ``NISAR_TrackFrame*.gpkg`` in the opera-utils data cache.
    """
    for cand in (explicit, os.environ.get("OPERA_STUDIO_NISAR_GPKG")):
        if cand and Path(cand).expanduser().is_file():
            return Path(cand).expanduser()
    try:
        import pooch

        cache = Path(pooch.os_cache("opera_utils"))
    except ImportError:
        cache = Path.home() / ".cache" / "opera_utils"
    found = sorted(cache.rglob("NISAR_TrackFrame*.gpkg")) if cache.exists() else []
    return found[-1] if found else None


class _Layer:
    def __init__(
        self, ids: list[str], geoms: list, props: list[dict], note: str | None = None
    ):
        self.ids = ids
        self.index = {i: k for k, i in enumerate(ids)}
        self.geoms = [geo.unwrap(g) for g in geoms]
        self.props = props
        self.note = note
        self.tree = STRtree(self.geoms)
        self.centres = (
            np.array(
                [
                    [(g.bounds[0] + g.bounds[2]) / 2, (g.bounds[1] + g.bounds[3]) / 2]
                    for g in self.geoms
                ]
            )
            if self.geoms
            else np.zeros((0, 2))
        )

    def query(self, bbox, limit: int) -> tuple[list[dict], bool]:
        w, s, e, n = bbox
        found: set[int] = set()
        # unwrapped outlines live in -180..~200; also look one turn east and west
        for dx in (0, 360, -360):
            found.update(
                int(k)
                for k in self.tree.query(
                    box(w + dx, s, e + dx, n), predicate="intersects"
                )
            )
        hits = sorted(found)
        truncated = len(hits) > limit
        if truncated:
            # keep the outlines nearest the middle of the view (any world copy)
            cx, cy = (w + e) / 2, (s + n) / 2
            c = self.centres[hits]
            dx = (c[:, 0] - cx + 180) % 360 - 180
            hits = [hits[i] for i in np.argsort(dx**2 + (c[:, 1] - cy) ** 2)[:limit]]
        feats = []
        for k in sorted(hits):
            feats.append(
                {
                    "type": "Feature",
                    "id": self.ids[k],
                    "geometry": mapping(self.geoms[k]),
                    "properties": self.props[k],
                }
            )
        return feats, truncated


def _load_frames() -> _Layer:
    from opera_utils import datasets
    from opera_utils.burst_frame_db import read_zipped_json

    ids, geoms, props = [], [], []
    for f in read_zipped_json(datasets.fetch_frame_geometries_simple())["features"]:
        p = f["properties"]
        if not p.get("is_north_america"):
            continue
        fid = int(f["id"])
        ids.append(f"F{fid:05d}")
        geoms.append(shape(f["geometry"]))
        props.append(
            {
                "id": f"F{fid:05d}",
                "label": f"DISP frame {fid}",
                "direction": str(p.get("orbit_pass", "")).lower(),
                "land": bool(p.get("is_land")),
            }
        )
    return _Layer(ids, geoms, props)


def _load_bursts() -> _Layer:
    from opera_utils import datasets
    from opera_utils.burst_frame_db import read_zipped_json

    ids, geoms, props = [], [], []
    for f in read_zipped_json(datasets.fetch_burst_id_geometries_simple())["features"]:
        p = f["properties"]
        if not p.get("is_north_america"):
            continue
        bid = p["burst_id_jpl"].upper()
        ids.append(bid)
        geoms.append(shape(f["geometry"]))
        props.append(
            {
                "id": bid,
                "label": f"Burst {bid}",
                "direction": str(p.get("orbit_pass", "")).lower(),
                "land": bool(p.get("is_land")),
            }
        )
    return _Layer(ids, geoms, props)


def _load_nisar(gpkg: Path | None = None) -> _Layer:
    path = find_nisar_gpkg(gpkg)
    if path is not None:
        try:
            return _load_nisar_gpkg(path)
        except Exception as e:
            logger.warning(
                "NISAR GeoPackage %s unreadable (%s); using bounding boxes", path, e
            )
    return _load_nisar_boxes()


def _load_nisar_gpkg(path: Path) -> _Layer:
    """Real NISAR frame outlines (land frames, worldwide) with pass direction."""
    import pyogrio

    df = pyogrio.read_dataframe(
        path, columns=["track", "frame", "passDirection", "hasLand"]
    )
    df = df[df["hasLand"].astype(bool)]
    ids, geoms, props = [], [], []
    for raw_track, raw_frame, direction, g in zip(
        df["track"], df["frame"], df["passDirection"], df.geometry
    ):
        if g is None or g.is_empty:
            continue
        track, frame = int(raw_track), int(raw_frame)
        gid = f"{track:03d}_{frame:03d}"
        ids.append(gid)
        geoms.append(g)
        props.append(
            {
                "id": gid,
                "label": f"NISAR track {track} · frame {frame}",
                "direction": str(direction or "").lower(),
                "land": True,
                "track": track,
                "frame": frame,
            }
        )
    logger.info("outlines: NISAR frames from %s", path)
    return _Layer(ids, geoms, props)


def _load_nisar_boxes() -> _Layer:
    from opera_utils.nisar._datasets import fetch_nisar_frame_to_bounds_file

    data = json.loads(
        Path(fetch_nisar_frame_to_bounds_file()).read_text(encoding="utf-8")
    )["data"]
    ids, geoms, props = [], [], []
    for key, v in data.items():
        track, frame = nisar_index_to_track_frame(int(key))
        gid = f"{track:03d}_{frame:03d}"
        ids.append(gid)
        geoms.append(
            projected_box_to_lonlat(
                v["epsg"], v["xmin"], v["ymin"], v["xmax"], v["ymax"]
            )
        )
        props.append(
            {
                "id": gid,
                "label": f"NISAR track {track} · frame {frame}",
                "direction": "",
                "land": bool(v.get("is_land")),
                "track": track,
                "frame": frame,
            }
        )
    note = (
        "NISAR outlines are bounding boxes without direction: set"
        " OPERA_STUDIO_NISAR_GPKG to the NISAR track-frame GeoPackage for the real"
        " frames"
    )
    return _Layer(ids, geoms, props, note=note)


_LOADERS: dict[str, Callable[..., _Layer]] = {
    "frames": _load_frames,
    "bursts": _load_bursts,
    "nisar": _load_nisar,
}


class Outlines:
    """Lazy outline layers, CMR depth counts (cached) and timelines."""

    def __init__(
        self,
        cache_file: Path | None = None,
        workers: int = 8,
        nisar_gpkg: Path | None = None,
    ):
        """Cache CMR counts in ``cache_file``; query CMR with ``workers`` threads."""
        self.nisar_gpkg = nisar_gpkg
        self._layers: dict[str, _Layer] = {}
        self._locks = {k: threading.Lock() for k in LAYERS}
        self._cache: dict[str, tuple[float, int]] = {}
        self._cache_lock = threading.Lock()
        self.cache_file = cache_file
        self.workers = workers
        self.session = requests.Session()
        if cache_file and cache_file.exists():
            try:
                raw = json.loads(cache_file.read_text())
                now = time.time()
                self._cache = {
                    k: tuple(v) for k, v in raw.items() if now - v[0] < DEPTH_TTL_S
                }
            except (json.JSONDecodeError, TypeError, ValueError):
                self._cache = {}

    # -- layers -----------------------------------------------------------------
    def layer(self, name: str) -> _Layer:
        if name not in _LOADERS:
            msg = f"unknown layer {name!r}"
            raise OutlineError(msg)
        with self._locks[name]:
            if name not in self._layers:
                t0 = time.time()
                self._layers[name] = (
                    _load_nisar(self.nisar_gpkg)
                    if name == "nisar"
                    else _LOADERS[name]()
                )
                logger.info(
                    "outlines: %s loaded (%d) in %.1fs",
                    name,
                    len(self._layers[name].ids),
                    time.time() - t0,
                )
            return self._layers[name]

    def preload(self) -> None:
        threading.Thread(
            target=lambda: [self.layer(n) for n in LAYERS], daemon=True
        ).start()

    def features(self, name: str, bbox, limit: int = 2500) -> dict:
        lay = self.layer(name)
        feats, truncated = lay.query(bbox, limit)
        out = {"type": "FeatureCollection", "features": feats, "truncated": truncated}
        if lay.note:
            out["note"] = lay.note
        return out

    def geometry(self, name: str, oid: str) -> dict:
        lay = self.layer(name)
        if oid not in lay.index:
            msg = f"no {name} outline {oid!r}"
            raise OutlineError(msg)
        return mapping(lay.geoms[lay.index[oid]])

    # -- CMR ----------------------------------------------------------------------
    @staticmethod
    def collection(name: str, kind: str | None) -> str:
        """Return the collection an outline layer is counted in (``kind`` picks one)."""
        if name == "frames":
            return COLLECTIONS["disp-static" if kind == "static" else "disp"]
        if name == "bursts":
            return COLLECTIONS["cslc-static" if kind == "static" else "cslc"]
        if name == "nisar":
            return COLLECTIONS.get(kind or "gslc", COLLECTIONS["gslc"])
        msg = f"unknown layer {name!r}"
        raise OutlineError(msg)

    def cmr_params(
        self, name: str, oid: str, kind: str | None, start: str | None, end: str | None
    ) -> dict:
        params: dict = {"short_name": self.collection(name, kind)}
        if name == "frames":
            m = re.fullmatch(r"F(\d{5})", oid)
            if not m:
                raise OutlineError(oid)
            attrs = [f"int,FRAME_NUMBER,{int(m.group(1))}"]
            if kind != "static":
                attrs.append("float,PRODUCT_VERSION,1.0")
            params["attribute[]"] = attrs
        elif name == "bursts":
            if not re.fullmatch(r"T\d{3}_\d{6}_IW[123]", oid):
                raise OutlineError(oid)
            params["attribute[]"] = [f"string,OPERA_BURST_ID,{oid}"]
        else:
            m = re.fullmatch(r"(\d{3})_(\d{3})", oid)
            if not m:
                raise OutlineError(oid)
            params["attribute[]"] = [
                f"int,TRACK_NUMBER,{int(m.group(1))}",
                f"int,FRAME_NUMBER,{int(m.group(2))}",
            ]
            params["provider"] = "ASF"
        if (start or end) and kind != "static":
            t0 = f"{start}T00:00:00Z" if start else ""
            t1 = f"{end}T23:59:59Z" if end else ""
            params["temporal"] = f"{t0},{t1}"
        return params

    def _count(self, name, oid, kind, start, end) -> int | None:
        key = f"{name}|{oid}|{kind}|{start}|{end}"
        with self._cache_lock:
            hit = self._cache.get(key)
        if hit and time.time() - hit[0] < DEPTH_TTL_S:
            return hit[1]
        params = {**self.cmr_params(name, oid, kind, start, end), "page_size": 0}
        for attempt in range(3):
            try:
                r = self.session.get(CMR, params=params, timeout=30)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                n = int(r.headers.get("CMR-Hits", "0"))
            except (requests.RequestException, ValueError) as e:
                logger.debug("depth %s %s: %s", name, oid, e)
                time.sleep(1.0)
            else:
                with self._cache_lock:
                    self._cache[key] = (time.time(), n)
                return n
        return None

    def depth(
        self,
        name: str,
        ids: list[str],
        kind: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, int | None]:
        """Granule count per outline id (None where CMR did not answer)."""
        ids = list(dict.fromkeys(ids))[:400]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            counts = list(
                pool.map(lambda i: self._count(name, i, kind, start, end), ids)
            )
        self._save()
        return dict(zip(ids, counts))

    def _save(self) -> None:
        if not self.cache_file:
            return
        with self._cache_lock:
            data = dict(self._cache)
        try:
            tmp = self.cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(self.cache_file)
        except OSError:
            pass

    def timeline(
        self,
        name: str,
        oid: str,
        kind: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict:
        """Acquisition dates of one outline, with revisit statistics and gaps."""
        from opera_utils._cmr import fetch_cmr_pages

        params = {**self.cmr_params(name, oid, kind, start, end), "page_size": 2000}
        items = fetch_cmr_pages(CMR, params)
        rows = []
        for it in items:
            umm = it["umm"]
            ur = umm.get("GranuleUR", "")
            when, ref = _granule_dates(name, kind, umm)
            if when:
                rows.append((when, ref, ur))
        rows.sort()
        out = summarize_dates([r[0] for r in rows])
        out.update(
            {
                "layer": name,
                "id": oid,
                "collection": params["short_name"],
                "n_files": len(rows),
                "reference_dates": sorted({r[1] for r in rows if r[1]}),
                "geometry": self.geometry(name, oid),
            }
        )
        return out


_STAMP = re.compile(r"(\d{8}T\d{6})")


def _granule_dates(
    name: str, kind: str | None, umm: dict
) -> tuple[str | None, str | None]:
    """(acquisition date, reference date or None) of one granule, as YYYY-MM-DD."""
    ur = umm.get("GranuleUR", "")
    stamps = _STAMP.findall(ur)
    to_day = lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}"  # noqa: E731
    if name == "frames" and kind != "static" and len(stamps) >= 2:
        return to_day(stamps[1]), to_day(stamps[0])  # DISP: reference, secondary
    if name == "nisar" and kind == "gunw" and len(stamps) >= 3:
        return to_day(stamps[2]), to_day(stamps[0])
    if stamps:
        return to_day(stamps[0]), None
    t = umm.get("TemporalExtent", {}).get("RangeDateTime", {}).get("BeginningDateTime")
    return (t[:10], None) if t else (None, None)


def summarize_dates(dates: list[str]) -> dict:
    """Summarize unique dates, median revisit and gaps over 2.5x the median revisit."""
    uniq = sorted(set(dates))
    out = {
        "dates": uniq,
        "n_dates": len(uniq),
        "first": uniq[0] if uniq else None,
        "last": uniq[-1] if uniq else None,
        "gaps": [],
        "revisit_days": None,
    }
    if len(uniq) < 3:
        return out
    days = np.array(
        [
            datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() / 86400
            for d in uniq
        ]
    )
    step = np.diff(days)
    med = float(np.median(step))
    out["revisit_days"] = round(med, 1)
    out["gaps"] = [
        [uniq[i], uniq[i + 1], round(float(step[i]))]
        for i in np.nonzero(step > max(2.5 * med, med + 12))[0]
    ]
    return out
