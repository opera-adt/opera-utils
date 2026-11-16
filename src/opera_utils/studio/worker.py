"""Run one studio job: ``python -m opera_utils.studio.worker <job_dir>/job.json``.

The job folder holds ``job.json`` (what to do, written by the server),
``status.json`` (written here, read by the server) and ``job.log``.
Outputs go under ``<job_dir>/out`` (or ``frames`` for DISP stacks).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shapely.geometry import shape

from . import geo

log = logging.getLogger("opera_utils.studio")

RETRY_ATTEMPTS = 3
RETRY_WAIT_S = 20


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Status:
    """status.json, rewritten atomically on every change."""

    def __init__(self, path: Path, job: dict):
        """Start ``path`` for ``job``: running, every group queued."""
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {
            "state": "running",
            "step": "starting",
            "progress": 0.0,
            "started": _now(),
            "pid": os.getpid(),
            "groups": [{"id": g, "state": "queued"} for g in job.get("groups", [])],
            "outputs": [],
            "frames": [],
        }
        self.write()

    def write(self) -> None:
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1, default=str))
            tmp.replace(self.path)

    def update(self, **kw) -> None:
        with self.lock:
            self.data.update(kw)
        self.write()

    def group(self, gid: str, **kw) -> None:
        with self.lock:
            for g in self.data["groups"]:
                if g["id"] == gid:
                    g.update(kw)
        self.write()


class Progress:
    """Count files appearing in a folder while a library call downloads them."""

    def __init__(
        self,
        folder: Path,
        expected: int,
        on_change: Callable[[int], None],
        pattern: str = "*",
    ):
        """Call ``on_change(n)`` as the number of ``pattern`` files changes."""
        self.folder, self.expected, self.on_change, self.pattern = (
            folder,
            max(expected, 1),
            on_change,
            pattern,
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        last = -1
        while not self._stop.wait(2.0):
            n = (
                len(
                    [
                        p
                        for p in self.folder.glob(self.pattern)
                        if not p.name.endswith((".part", ".tmp"))
                    ]
                )
                if self.folder.exists()
                else 0
            )
            if n != last:
                last = n
                self.on_change(n)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=3)


def _transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        s in text
        for s in (
            "timeout",
            "timed out",
            "connection",
            "503",
            "502",
            "504",
            "429",
            "reset by peer",
            "temporarily",
        )
    )


def with_retries(
    fn: Callable, cleanup: Callable[[], None] | None = None, on_retry=None
):
    """Call ``fn``, retrying transient network errors with a growing wait."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == RETRY_ATTEMPTS or not _transient(e):
                raise
            if cleanup:
                cleanup()
            wait = RETRY_WAIT_S * attempt
            log.warning("transient error (%s); retrying in %s s", e, wait)
            if on_retry:
                on_retry(wait, attempt + 1)
            time.sleep(wait)
    return None


def drop_broken_h5(folder: Path, pattern: str = "*.nc") -> int:
    """Remove half-written files left by an interrupted download."""
    import h5py

    n = 0
    for f in folder.glob(pattern):
        try:
            with h5py.File(f, "r"):
                pass
        except OSError:
            f.unlink(missing_ok=True)
            n += 1
    return n


def _session():
    import asf_search as asf

    from opera_utils.credentials import get_earthdata_username_password

    user, pw = get_earthdata_username_password()
    return asf.ASFSession().auth_with_creds(user, pw)


def fetch(url: str, out: Path, session, chunk: int = 4 << 20) -> Path:
    """Stream one file over HTTPS (Earthdata auth), via a .part file."""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk):
                f.write(block)
    tmp.replace(out)
    return out


def _rel(job_dir: Path, p: Path) -> str:
    return str(Path(p).resolve().relative_to(job_dir.resolve()))


def _add_outputs(
    status: Status, job_dir: Path, paths, kind: str, group: str | None = None
) -> None:
    for path in paths:
        p = Path(path)
        if p.is_file():
            status.data["outputs"].append(
                {
                    "path": _rel(job_dir, p),
                    "size": p.stat().st_size,
                    "kind": kind,
                    "group": group,
                }
            )
        elif p.is_dir():
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            status.data["outputs"].append(
                {
                    "path": _rel(job_dir, p),
                    "size": size,
                    "kind": kind,
                    "group": group,
                    "dir": True,
                }
            )
    status.write()


# -- products -------------------------------------------------------------------
def run_disp(job: dict, job_dir: Path, status: Status) -> None:
    """Download, rebase and fit each DISP-S1 frame of the job."""
    from opera_utils.disp._download import run_download
    from opera_utils.disp._enums import QualityDataset, ReferenceMethod
    from opera_utils.disp._reformat import reformat_stack

    from .rebase import Frame

    opts = job.get("options", {})
    aoi = shape(job["aoi"])
    frames = job["groups"]
    footprints = {
        f["properties"]["id"]: geo.unwrap(shape(f["geometry"]))
        for f in job.get("footprints", [])
    }
    per_group: defaultdict[str, int] = defaultdict(int)
    for it in job.get("items", []):
        per_group[it["group"]] += 1
    ref_mode = opts.get(
        "reference", "auto"
    )  # auto | point | high_coherence | median | border | none
    method = {
        "median": ReferenceMethod.MEDIAN,
        "border": ReferenceMethod.BORDER,
        "none": ReferenceMethod.NONE,
    }.get(ref_mode, ReferenceMethod.HIGH_COHERENCE)
    n = len(frames)
    for i, gid in enumerate(frames):
        frame_id = int(gid.lstrip("F"))
        area = geo.intersect_lon(aoi, footprints[gid]) if gid in footprints else aoi
        if area.is_empty:
            area = aoi

        def mark(step: str, frac: float, gid=gid, i=i) -> None:
            status.group(gid, state=step)
            status.update(step=f"{gid}: {step}", progress=round((i + frac) / n, 3))

        dl = job_dir / "download" / gid
        expected = per_group.get(gid, 1)
        mark("downloading", 0.0)

        def on_files(k: int, gid=gid, i=i, expected=expected) -> None:
            status.group(gid, state=f"downloading {k}/{expected}")
            status.update(
                step=f"{gid}: downloading {k}/{expected}",
                progress=round((i + 0.6 * min(k / expected, 1)) / n, 3),
            )

        def cleanup(dl=dl) -> None:
            drop_broken_h5(dl)

        with Progress(dl, expected, on_files, "*.nc"):
            files = with_retries(
                lambda frame_id=frame_id, area=area, dl=dl: run_download(
                    frame_id=frame_id,
                    start_datetime=_day(job.get("start")),
                    end_datetime=_day(job.get("end"), end=True),
                    wkt=area.wkt,
                    num_workers=int(opts.get("workers", 4)),
                    product_version=opts.get("version") or "1.0",
                    output_dir=dl,
                ),
                cleanup=cleanup,
                on_retry=lambda w, k: mark(
                    f"ASF busy, retrying in {w} s (try {k})", 0.0
                ),
            )
        files = sorted(Path(f) for f in files)
        status.group(gid, n_files=len(files))
        if not files:
            status.group(gid, state="no data")
            continue

        mark("rebasing (reformat_stack)", 0.6)
        fdir = job_dir / "frames" / gid
        shutil.rmtree(fdir / "stack.zarr", ignore_errors=True)
        fdir.mkdir(parents=True, exist_ok=True)
        mask = bool(opts.get("recommended_mask", False))
        reformat_stack(
            files,
            str(fdir / "stack.zarr"),
            apply_solid_earth_corrections=bool(opts.get("solid_earth", True)),
            apply_ionospheric_corrections=bool(opts.get("ionosphere", False)),
            quality_datasets=[QualityDataset.RECOMMENDED_MASK] if mask else None,
            quality_thresholds=[0.5] if mask else None,
            reference_method=method,
        )
        frame = Frame(fdir, dl)
        mark("fitting velocity", 0.8)
        frame.compute_analysis()

        mark("setting the reference point", 0.88)
        if ref_mode in ("auto", "point"):
            lonlat = (
                tuple(opts["reference_lonlat"])
                if ref_mode == "point" and opts.get("reference_lonlat")
                else None
            )
            try:
                ref = frame.set_reference(lonlat, mode="point" if lonlat else "auto")
            except ValueError as e:
                log.warning("%s: keeping the area reference: %s", gid, e)
                ref = frame.set_reference(mode="area")
                ref["error"] = str(e)
        else:
            ref = frame.set_reference(mode="area")
            ref["method"] = {
                "median": "median of all pixels",
                "border": "median of border pixels",
                "none": "none (no spatial reference)",
            }.get(ref_mode, ref["method"])
            frame.reference_path.write_text(json.dumps(ref, indent=1))

        mark("writing GeoTIFFs", 0.93)
        tifs = frame.export_geotiffs()
        _add_outputs(status, job_dir, tifs, "geotiff", gid)
        _add_outputs(status, job_dir, [fdir / "stack.zarr"], "stack", gid)

        if opts.get("static", True):
            mark("static layers (line of sight)", 0.96)
            try:
                from opera_utils.disp._static import download_static

                written = download_static(
                    frame_id,
                    wkt=area.wkt,
                    output_dir=fdir / "static",
                    layers=("line_of_sight_enu", "layover_shadow_mask"),
                )
                _add_outputs(status, job_dir, written.values(), "static", gid)
            except Exception as e:
                log.warning("%s: no static layers: %s", gid, e)
        if not opts.get("keep_downloads", True):
            shutil.rmtree(dl, ignore_errors=True)
        else:
            _add_outputs(status, job_dir, [dl], "subsets", gid)
        meta = frame.meta()
        status.data["frames"].append(
            {
                "id": gid,
                "direction": job.get("group_info", {}).get(gid, {}).get("direction"),
                **meta,
            }
        )
        status.group(gid, state="done")


def _items_by_group(job: dict) -> dict[str, list[dict]]:
    out = defaultdict(list)
    chosen = set(job.get("groups", []))
    for it in job.get("items", []):
        if it["group"] in chosen:
            out[it["group"]].append(it)
    return out


def run_cslc(job: dict, job_dir: Path, status: Status) -> None:
    """Download each CSLC-S1 burst and crop it to the area."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .subset import subset_h5

    opts = job.get("options", {})
    aoi = shape(job["aoi"])
    outlines = {
        f["properties"]["id"]: geo.unwrap(shape(f["geometry"]))
        for f in job.get("footprints", [])
    }

    def crop_box(g: str) -> tuple[float, float, float, float]:
        """Return the area within this burst (continuous bounds across ±180°)."""
        if g in outlines:
            part = geo.intersect_lon(aoi, outlines[g])
            if not part.is_empty:
                return part.bounds
        return aoi.bounds

    do_subset = opts.get("subset", True)
    session = _session()
    groups = _items_by_group(job)
    todo = [(g, it) for g, its in groups.items() for it in its]
    done = 0
    lock = threading.Lock()

    def one(g: str, it: dict) -> Path:
        name = it.get("name") or it["url"].rsplit("/", 1)[-1]
        out_dir = job_dir / "out" / g
        final = out_dir / name
        if final.exists():
            return final
        raw = job_dir / "download" / name if do_subset else final
        with_retries(
            lambda: fetch(it["url"], raw, session),
            cleanup=lambda: raw.with_name(raw.name + ".part").unlink(missing_ok=True),
        )
        if do_subset:
            out_dir.mkdir(parents=True, exist_ok=True)
            subset_h5(raw, final, crop_box(g))
            raw.unlink(missing_ok=True)
        return final

    with ThreadPoolExecutor(max_workers=int(opts.get("workers", 3))) as pool:
        futs = {pool.submit(one, g, it): g for g, it in todo}
        for fut in as_completed(futs):
            g = futs[fut]
            path = fut.result()
            with lock:
                done += 1
                _add_outputs(status, job_dir, [path], "h5", g)
                status.update(
                    step=(
                        f"{'downloading + subsetting' if do_subset else 'downloading'}"
                        f" {done}/{len(todo)}"
                    ),
                    progress=round(done / max(len(todo), 1), 3),
                )
    for g in groups:
        status.group(g, state="done")


def is_frame_id(gid: str) -> bool:
    """DISP-S1 frame groups are ``Fxxxxx``; CSLC burst groups are burst IDs."""
    return len(gid) == 6 and gid[0] == "F" and gid[1:].isdigit()


def run_static(job: dict, job_dir: Path, status: Status) -> None:
    """DISP-S1-STATIC frames and CSLC-S1-STATIC bursts, routed by group id."""
    frames = [g for g in job["groups"] if is_frame_id(g)]
    bursts = [g for g in job["groups"] if not is_frame_id(g)]
    if frames:
        _disp_static(job, job_dir, status, frames)
    if bursts:
        run_cslc({**job, "groups": bursts}, job_dir, status)


def _disp_static(job: dict, job_dir: Path, status: Status, frames: list[str]) -> None:
    from opera_utils.disp._static import LAYERS, download_static

    opts = job.get("options", {})
    aoi = shape(job["aoi"])
    footprints = {
        f["properties"]["id"]: geo.unwrap(shape(f["geometry"]))
        for f in job.get("footprints", [])
    }
    layers = tuple(opts.get("layers") or LAYERS)
    for i, gid in enumerate(frames):
        status.group(gid, state="downloading")
        status.update(
            step=f"{gid}: reading {', '.join(layers)}",
            progress=round(i / len(frames), 3),
        )
        area = geo.intersect_lon(aoi, footprints[gid]) if gid in footprints else aoi
        written = with_retries(
            lambda gid=gid, area=area: download_static(
                int(gid.lstrip("F")),
                wkt=(
                    None
                    if opts.get("full_frame")
                    else (area if not area.is_empty else aoi).wkt
                ),
                output_dir=job_dir / "out" / gid,
                layers=layers,
            )
        )
        _add_outputs(status, job_dir, written.values(), "geotiff", gid)
        status.group(gid, state="done")


def run_nisar(job: dict, job_dir: Path, status: Status) -> None:
    """Download NISAR granules, whole or cropped to the area's bounding box."""
    from .search import NISAR_COLLECTIONS

    kind = job.get("kind") or "gslc"
    opts = job.get("options", {})
    groups = _items_by_group(job)
    if kind == "gunw" or not opts.get("subset", True):
        session = _session()
        todo = [(g, it) for g, its in groups.items() for it in its]
        for k, (g, it) in enumerate(todo, 1):
            status.update(
                step=f"downloading {k}/{len(todo)} (full granules)",
                progress=round((k - 1) / len(todo), 3),
            )
            status.group(g, state="downloading")
            out = job_dir / "out" / g / (it.get("name") or it["url"].rsplit("/", 1)[-1])
            with_retries(lambda it=it, out=out: fetch(it["url"], out, session))
            _add_outputs(status, job_dir, [out], "h5", g)
        for g in groups:
            status.group(g, state="done")
        return

    from opera_utils.nisar._download import run_download

    if geo.crosses(shape(job["aoi"])):
        msg = (
            "NISAR cropping across the antimeridian is not supported yet: split the"
            " area at 180°, or untick cropping."
        )
        raise RuntimeError(msg)
    bbox = shape(job["aoi"]).bounds
    n = len(groups)
    for i, (g, its) in enumerate(groups.items()):
        rel, direction, tf = g.split("_")
        out = job_dir / "out" / g
        status.group(g, state="downloading + subsetting")
        m = len(its)

        def on_files(k: int, g=g, i=i, m=m) -> None:
            status.group(g, state=f"subsetting {k}/{m}")
            status.update(
                step=f"{g}: subsetting {k}/{m}",
                progress=round((i + min(k / m, 1)) / n, 3),
            )

        with Progress(out, m, on_files, "*.h5"):
            files = with_retries(
                lambda rel=rel, direction=direction, tf=tf, out=out: run_download(
                    bbox=bbox,
                    relative_orbit_number=int(rel),
                    track_frame_number=int(tf),
                    orbit_direction=direction,
                    start_datetime=_day(job.get("start")),
                    end_datetime=_day(job.get("end"), end=True),
                    polarizations=opts.get("polarizations") or None,
                    short_name=NISAR_COLLECTIONS[kind],
                    num_workers=int(opts.get("workers", 4)),
                    output_dir=out,
                )
            )
        _add_outputs(status, job_dir, files, "h5", g)
        status.group(g, state="done")


def run_tropo(job: dict, job_dir: Path, status: Status) -> None:
    """Crop the selected global TROPO cubes to the area."""
    from opera_utils.tropo._crop import crop_tropo

    opts = job.get("options", {})
    if geo.crosses(shape(job["aoi"])):
        msg = (
            "TROPO cropping cannot cross the antimeridian yet: split the area at 180°."
        )
        raise RuntimeError(msg)
    groups = _items_by_group(job)
    items = [it for its in groups.values() for it in its]
    if not items:
        msg = "No TROPO products selected."
        raise RuntimeError(msg)
    urls_file = job_dir / "tropo_urls.txt"
    urls_file.write_text("\n".join(it["url"] for it in items) + "\n")
    datetimes = sorted(
        datetime.strptime(it["date"], "%Y-%m-%dT%H:%M:%SZ") for it in items
    )
    out = job_dir / "out" / "tropo"
    status.update(step=f"cropping {len(datetimes)} TROPO cubes", progress=0.05)
    with Progress(
        out,
        len(datetimes),
        lambda k: status.update(
            step=f"cropping {k}/{len(datetimes)}",
            progress=round(min(k / len(datetimes), 1), 3),
        ),
        "*.nc",
    ):
        crop_tropo(
            urls_file,
            datetimes,
            aoi_bounds=shape(job["aoi"]).bounds,
            output_dir=out,
            skip_time_interpolation=True,
            height_max=float(opts.get("height_max", 10000.0)),
            margin_deg=float(opts.get("margin_deg", 0.3)),
            num_workers=int(opts.get("workers", 2)),
        )
    _add_outputs(status, job_dir, sorted(out.glob("*.nc")), "netcdf", None)
    for g in groups:
        status.group(g, state="done")


RUNNERS = {
    "disp": run_disp,
    "cslc": run_cslc,
    "static": run_static,
    "nisar": run_nisar,
    "tropo": run_tropo,
}


def _day(value: str | None, end: bool = False) -> datetime | None:
    if not value:
        return None
    d = datetime.fromisoformat(value[:10])
    return d.replace(hour=23, minute=59, second=59) if end else d


def run(job_json: Path) -> int:
    """Run the job in ``job_json``; return 0 on success, 1 on failure."""
    job_dir = job_json.parent
    job = json.loads(job_json.read_text(encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(job_dir / "job.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.getLogger("opera_utils").setLevel(logging.INFO)
    status = Status(job_dir / "status.json", job)
    t0 = time.time()
    try:
        log.info(
            "job %s: %s %s, %d groups",
            job["id"],
            job["product"],
            job.get("kind") or "",
            len(job.get("groups", [])),
        )
        RUNNERS[job["product"]](job, job_dir, status)
        status.update(
            state="done",
            step="done",
            progress=1.0,
            finished=_now(),
            seconds=round(time.time() - t0, 1),
        )
    except Exception as e:
        log.exception("job failed")
        status.update(
            state="failed",
            step="failed",
            error=f"{type(e).__name__}: {e}"[:600],
            finished=_now(),
        )
        return 1
    return 0


def main() -> None:
    """Run the job file given on the command line and exit with its code."""
    sys.exit(run(Path(sys.argv[1])))


if __name__ == "__main__":
    main()
