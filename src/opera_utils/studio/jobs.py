"""Job queue: each job is a folder and a ``worker`` subprocess.

``<workdir>/jobs/<id>/job.json`` is written here; the worker writes
``status.json`` and ``job.log``. Jobs wait in a queue while ``max_running``
are active. Cancelling kills the worker's process group.
"""

from __future__ import annotations

import builtins
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from shapely.geometry import mapping

from .search import PRODUCTS, area_km2, parse_aoi

PRODUCT_LABELS = {
    "disp": "DISP-S1",
    "cslc": "CSLC-S1",
    "nisar": "NISAR",
    "tropo": "TROPO",
    "static": "Static layers",
}


class JobError(ValueError):
    """The job request is not valid."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A finished child stays a zombie until reaped; treat it as gone.
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return done == 0


class JobManager:
    """Create, queue, start, cancel and bundle jobs kept as folders under ``root``."""

    def __init__(
        self,
        root: Path,
        env_factory=None,
        max_running: int = 2,
        python: str | None = None,
    ):
        """Keep jobs under ``root``; ``env_factory`` gives each worker's environment."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.env_factory = env_factory or (lambda: dict(os.environ))
        self.max_running = max_running
        self.python = python or sys.executable
        self._lock = threading.RLock()
        self._procs: dict[str, subprocess.Popen] = {}

    # -- records -------------------------------------------------------------
    def _dir(self, job_id: str) -> Path:
        if not job_id or "/" in job_id or job_id.startswith("."):
            msg = "bad job id"
            raise JobError(msg)
        d = self.root / job_id
        if not (d / "job.json").exists():
            raise FileNotFoundError(job_id)
        return d

    def record(self, job_id: str, log_lines: int = 0) -> dict:
        d = self._dir(job_id)
        job = _read(d / "job.json") or {}
        status = _read(d / "status.json") or {}
        meta = _read(d / "meta.json") or {}
        state = meta.get("state") or status.get("state") or "queued"
        if state == "running" and status.get("state") in ("done", "failed"):
            state = status["state"]
        rec = {
            "id": job_id,
            "product": job.get("product"),
            "kind": job.get("kind"),
            "title": job.get("title"),
            "created": job.get("created"),
            "aoi": job.get("aoi"),
            "aoi_km2": job.get("aoi_km2"),
            "start": job.get("start"),
            "end": job.get("end"),
            "n_groups": len(job.get("groups", [])),
            "n_items": len(job.get("items", [])),
            "options": job.get("options", {}),
            "dir": str(d),
            **{k: v for k, v in status.items() if k not in ("pid",)},
            "state": state,
        }
        if log_lines:
            log = d / "job.log"
            if log.exists():
                with open(log, "rb") as f:
                    f.seek(max(0, log.stat().st_size - 64_000))
                    rec["log"] = (
                        f.read().decode(errors="replace").splitlines()[-log_lines:]
                    )
        return rec

    def list(self) -> list[dict]:
        out = []
        for d in self.root.iterdir():
            if (d / "job.json").exists():
                try:
                    out.append(self.record(d.name))
                except (FileNotFoundError, JobError):
                    continue
        return sorted(out, key=lambda r: r.get("created") or "", reverse=True)

    # -- lifecycle -------------------------------------------------------------
    def create(self, req: dict) -> dict:
        product = req.get("product")
        if product not in PRODUCTS:
            msg = f"Unknown product {product!r}"
            raise JobError(msg)
        groups = [str(g) for g in req.get("groups") or []]
        if not groups:
            msg = "Pick at least one result to download."
            raise JobError(msg)
        aoi = parse_aoi(req.get("aoi"))
        chosen = set(groups)
        items = [it for it in req.get("items") or [] if it.get("group") in chosen]
        footprints = [
            f
            for f in req.get("footprints") or []
            if f.get("properties", {}).get("id") in chosen
        ]
        job_id = (
            datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        )
        d = self.root / job_id
        d.mkdir(parents=True)
        unit = "frame" if product in ("disp", "static") else "group"
        plural = "s" if len(groups) != 1 else ""
        label = PRODUCT_LABELS[product] + (
            f" {req['kind'].upper()}"
            if req.get("kind") and product in ("nisar", "static")
            else ""
        )
        job = {
            "id": job_id,
            "product": product,
            "kind": req.get("kind"),
            "title": req.get("title") or f"{label} · {len(groups)} {unit}{plural}",
            "created": _now(),
            # normalised: centre in [-180, 180), continuous across ±180°
            "aoi": json.loads(json.dumps(mapping(aoi))),
            "aoi_km2": round(area_km2(aoi), 2),
            "start": req.get("start"),
            "end": req.get("end"),
            "groups": groups,
            "group_info": {
                g["id"]: {k: g.get(k) for k in ("direction", "label")}
                for g in req.get("group_info") or []
                if g.get("id") in chosen
            },
            "items": items,
            "footprints": footprints,
            "options": req.get("options") or {},
        }
        (d / "job.json").write_text(json.dumps(job, indent=1))
        (d / "meta.json").write_text(json.dumps({"state": "queued"}))
        self.tick()
        return self.record(job_id)

    def _set_meta(self, d: Path, **kw) -> None:
        meta = _read(d / "meta.json") or {}
        meta.update(kw)
        (d / "meta.json").write_text(json.dumps(meta))

    def _running_ids(self) -> builtins.list[str]:
        out = []
        for d in self.root.iterdir():
            meta = _read(d / "meta.json") or {}
            if meta.get("state") == "running":
                out.append(d.name)
        return out

    def tick(self) -> None:
        """Reap finished workers and start queued jobs."""
        with self._lock:
            for d in sorted(self.root.iterdir()):
                meta = _read(d / "meta.json")
                if not meta or meta.get("state") != "running":
                    continue
                proc = self._procs.get(d.name)
                finished = (
                    proc.poll() is not None if proc else not _alive(meta.get("pid"))
                )
                if finished:
                    status = _read(d / "status.json") or {}
                    state = (
                        status.get("state")
                        if status.get("state") in ("done", "failed")
                        else "failed"
                    )
                    if state == "failed" and not status.get("error"):
                        status.update(
                            state="failed",
                            step="failed",
                            error="the worker stopped unexpectedly (see the log)",
                        )
                        (d / "status.json").write_text(json.dumps(status, indent=1))
                    self._set_meta(d, state=state)
                    self._procs.pop(d.name, None)
            running = len(self._running_ids())
            queued = sorted(
                (
                    d
                    for d in self.root.iterdir()
                    if (_read(d / "meta.json") or {}).get("state") == "queued"
                ),
                key=lambda d: d.name,
            )
            for d in queued:
                if running >= self.max_running:
                    break
                self._launch(d)
                running += 1

    def _launch(self, d: Path) -> None:
        with open(d / "worker.out", "ab") as log:  # the child keeps its own copy
            proc = subprocess.Popen(
                [self.python, "-m", "opera_utils.studio.worker", str(d / "job.json")],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=self.env_factory(),
                cwd=str(d),
                start_new_session=True,
            )
        self._procs[d.name] = proc
        self._set_meta(d, state="running", pid=proc.pid, launched=_now())

    def cancel(self, job_id: str) -> dict:
        d = self._dir(job_id)
        with self._lock:
            meta = _read(d / "meta.json") or {}
            if meta.get("state") == "running" and meta.get("pid"):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(meta["pid"], signal.SIGTERM)
            if meta.get("state") in ("running", "queued"):
                self._set_meta(d, state="cancelled")
                status = _read(d / "status.json") or {}
                status.update(state="cancelled", step="cancelled")
                (d / "status.json").write_text(json.dumps(status, indent=1))
        return self.record(job_id)

    def delete(self, job_id: str) -> None:
        d = self._dir(job_id)
        self.cancel(job_id)
        shutil.rmtree(d, ignore_errors=True)

    # -- files -------------------------------------------------------------------
    def file_path(self, job_id: str, rel: str) -> Path:
        d = self._dir(job_id).resolve()
        p = (d / rel).resolve()
        if d not in p.parents or not p.exists():
            raise FileNotFoundError(rel)
        return p

    def bundle(self, job_id: str) -> Path:
        """Zip the job's outputs (directories such as Zarr stores included)."""
        d = self._dir(job_id)
        rec = self.record(job_id)
        z = d / "bundle.zip"
        if z.exists() and z.stat().st_mtime >= (d / "status.json").stat().st_mtime:
            return z
        tmp = d / "bundle.zip.part"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            for o in rec.get("outputs", []):
                p = d / o["path"]
                files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
                for f in files:
                    zf.write(f, f.relative_to(d))
            zf.write(d / "job.json", "job.json")
        tmp.replace(z)
        return z
