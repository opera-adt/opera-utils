"""HTTP server for the studio (standard library only).

Binds to 127.0.0.1 by default: the server holds Earthdata credentials, so it
is meant for the person at this machine (or an SSH tunnel), not a network.

Browser access is limited to

* the studio's own pages (same origin), and
* the origins given with ``--allow-origin`` (the hosted interface, e.g. on
  GitHub Pages), which get CORS headers, including Chrome's
  ``Access-Control-Allow-Private-Network`` for a public page calling localhost.

Requests from any other web origin are refused, and the ``Host`` header must
name this machine (``localhost`` / ``127.0.0.1`` / ``[::1]`` on the server's
port), which stops DNS-rebinding pages from reaching the API.
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import re
import threading
import time
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .auth import AuthError, AuthStore
from .jobs import JobError, JobManager
from .outlines import OutlineError, Outlines
from .search import SearchError, run_search

log = logging.getLogger("opera_utils.studio")
STATIC = Path(__file__).parent / "static"
MAX_BODY = 50 * 1024 * 1024
LOCAL_NAMES = ("localhost", "127.0.0.1", "[::1]")


def allowed_origins(
    extra: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Return the web origins allowed to call the API.

    These are ``extra`` plus ``$OPERA_STUDIO_ALLOW_ORIGIN`` (comma separated),
    each reduced to ``scheme://host[:port]``.
    """
    raw = list(extra or []) + [
        o
        for o in os.environ.get("OPERA_STUDIO_ALLOW_ORIGIN", "").split(",")
        if o.strip()
    ]
    out = []
    for o in raw:
        u = urlparse(o.strip())
        if u.scheme in ("http", "https") and u.netloc:
            out.append(f"{u.scheme}://{u.netloc}")
    return tuple(dict.fromkeys(out))


def local_hosts(host: str, port: int) -> set[str] | None:
    """Return the ``Host`` headers to answer; None (any) when bound beyond localhost."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        return None
    return {f"{name}:{port}" for name in LOCAL_NAMES}


class HttpError(Exception):
    """An error answered to the client with an HTTP ``status``."""

    def __init__(self, status: int, message: str):
        """Answer ``message`` with HTTP ``status``."""
        super().__init__(message)
        self.status = status


class Studio:
    """Application state shared by request handlers."""

    def __init__(
        self,
        workdir: Path,
        config_dir: Path | None = None,
        max_running: int = 2,
        nisar_gpkg: Path | None = None,
        origins: tuple[str, ...] = (),
        hosts: set[str] | None = None,
    ):
        """Keep jobs under ``workdir`` and Earthdata credentials in ``config_dir``."""
        self.origins = origins
        self.hosts = hosts
        self.workdir = Path(workdir).expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.auth = AuthStore(
            config_dir or (Path.home() / ".config" / "opera-utils-studio")
        )
        self.jobs = JobManager(
            self.workdir / "jobs",
            env_factory=self.auth.job_env,
            max_running=max_running,
        )
        self._frames: dict[tuple[str, str], object] = {}
        self._frames_lock = threading.Lock()
        self.outlines = Outlines(
            cache_file=self.workdir / "depth-cache.json", nisar_gpkg=nisar_gpkg
        )
        self._stop = threading.Event()
        threading.Thread(target=self._ticker, daemon=True).start()

    def start_background(self) -> None:
        """Load the outline layers off the request path (bursts take ~15 s)."""
        self.outlines.preload()

    def _ticker(self) -> None:
        while not self._stop.wait(2.0):
            try:
                self.jobs.tick()
            except Exception:
                log.exception("job tick failed")

    def frame(self, job_id: str, frame_id: str):
        from .rebase import Frame

        if not re.fullmatch(r"F\d{5}", frame_id):
            raise HttpError(404, "no such frame")
        key = (job_id, frame_id)
        with self._frames_lock:
            fr = self._frames.get(key)
            if fr is None:
                d = self.jobs._dir(job_id)
                fdir = d / "frames" / frame_id
                if not (fdir / "stack.zarr").exists():
                    raise HttpError(404, "this frame has no stack yet")
                fr = Frame(fdir, d / "download" / frame_id)
                self._frames[key] = fr
            return fr

    # -- routing ---------------------------------------------------------------
    def routes(self):
        return [
            ("GET", r"/api/health", self.health),
            ("GET", r"/api/auth", lambda _h, _m: self.auth.status()),
            ("POST", r"/api/auth", self.sign_in),
            ("DELETE", r"/api/auth", lambda _h, _m: self.auth.sign_out()),
            ("POST", r"/api/search", self.search),
            ("GET", r"/api/jobs", lambda _h, _m: {"jobs": self.jobs.list()}),
            ("POST", r"/api/jobs", self.create_job),
            ("GET", r"/api/jobs/(?P<id>[\w-]+)", self.job),
            ("DELETE", r"/api/jobs/(?P<id>[\w-]+)", self.delete_job),
            (
                "POST",
                r"/api/jobs/(?P<id>[\w-]+)/cancel",
                lambda _h, m: self.jobs.cancel(m["id"]),
            ),
            ("GET", r"/api/jobs/(?P<id>[\w-]+)/bundle\.zip", self.bundle),
            ("GET", r"/api/jobs/(?P<id>[\w-]+)/files/(?P<rel>.+)", self.file),
            (
                "GET",
                r"/api/jobs/(?P<id>[\w-]+)/disp/(?P<frame>F\d+)/meta",
                self.disp_meta,
            ),
            (
                "GET",
                r"/api/jobs/(?P<id>[\w-]+)/disp/(?P<frame>F\d+)/velocity\.png",
                self.disp_png,
            ),
            (
                "GET",
                r"/api/jobs/(?P<id>[\w-]+)/disp/(?P<frame>F\d+)/series",
                self.disp_series,
            ),
            (
                "POST",
                r"/api/jobs/(?P<id>[\w-]+)/disp/(?P<frame>F\d+)/reference",
                self.disp_reference,
            ),
            ("GET", r"/api/outlines/(?P<layer>\w+)", self.outline_features),
            ("POST", r"/api/depth/(?P<layer>\w+)", self.outline_depth),
            (
                "GET",
                r"/api/timeline/(?P<layer>\w+)/(?P<oid>[\w-]+)",
                self.outline_timeline,
            ),
        ]

    # outlines, archive depth and timelines
    def outline_features(self, h, m):
        try:
            bbox = [float(v) for v in h.query["bbox"][0].split(",")]
            assert len(bbox) == 4
        except (KeyError, ValueError, AssertionError) as e:
            raise HttpError(400, "give bbox=west,south,east,north") from e
        limit = min(int(h.query.get("limit", ["2500"])[0]), 5000)
        try:
            return self.outlines.features(m["layer"], bbox, limit)
        except OutlineError as e:
            raise HttpError(404, str(e)) from e

    def outline_depth(self, h, m):
        body = h.json()
        try:
            counts = self.outlines.depth(
                m["layer"],
                [str(i) for i in body.get("ids", [])],
                body.get("kind"),
                body.get("start"),
                body.get("end"),
            )
        except OutlineError as e:
            raise HttpError(400, str(e)) from e
        return {
            "layer": m["layer"],
            "counts": counts,
            "collection": self.outlines.collection(m["layer"], body.get("kind")),
        }

    def outline_timeline(self, h, m):
        q = h.query
        get = lambda k: q.get(k, [None])[0] or None  # noqa: E731
        try:
            return self.outlines.timeline(
                m["layer"], m["oid"], get("kind"), get("start"), get("end")
            )
        except OutlineError as e:
            raise HttpError(404, str(e)) from e

    def health(self, _h, _m):
        try:
            from opera_utils import __version__
        except ImportError:
            __version__ = "unknown"
        return {"ok": True, "version": __version__, "workdir": str(self.workdir)}

    def sign_in(self, h, _m):
        body = h.json()
        try:
            return self.auth.sign_in(
                body.get("username", ""),
                body.get("password", ""),
                bool(body.get("remember")),
            )
        except AuthError as e:
            raise HttpError(401, str(e)) from e

    def search(self, h, _m):
        t0 = time.time()
        try:
            res = run_search(h.json())
        except SearchError as e:
            raise HttpError(400, str(e)) from e
        res["seconds"] = round(time.time() - t0, 2)
        return res

    def create_job(self, h, _m):
        try:
            return self.jobs.create(h.json())
        except (JobError, SearchError) as e:
            raise HttpError(400, str(e)) from e

    def job(self, h, m):
        return self.jobs.record(m["id"], log_lines=int(h.query.get("log", ["40"])[0]))

    def delete_job(self, _h, m):
        with self._frames_lock:
            for k in [k for k in self._frames if k[0] == m["id"]]:
                self._frames.pop(k)
        self.jobs.delete(m["id"])
        return {"deleted": m["id"]}

    def file(self, h, m):
        p = self.jobs.file_path(m["id"], unquote(m["rel"]))
        if p.is_dir():
            raise HttpError(400, "this is a folder; download the bundle instead")
        h.send_file(p, attachment=True)

    def bundle(self, h, m):
        h.send_file(
            self.jobs.bundle(m["id"]),
            attachment=True,
            name=f"opera-studio-{m['id']}.zip",
        )

    # DISP exploration
    def disp_meta(self, _h, m):
        return self.frame(m["id"], m["frame"]).meta()

    def disp_png(self, h, m):
        q = h.query
        vmax = float(q["vmax"][0]) if "vmax" in q else None
        min_coh = float(q.get("min_coherence", ["0"])[0])
        png = self.frame(m["id"], m["frame"]).velocity_png(
            vmax=vmax, min_coherence=min_coh
        )
        h.send_bytes(png, "image/png", cache=False)

    def disp_series(self, h, m):
        try:
            lon, lat = float(h.query["lon"][0]), float(h.query["lat"][0])
        except (KeyError, ValueError) as e:
            raise HttpError(400, "give lon and lat") from e
        try:
            return self.frame(m["id"], m["frame"]).series(lon, lat)
        except ValueError as e:
            raise HttpError(400, str(e)) from e

    def disp_reference(self, h, m):
        body = h.json()
        fr = self.frame(m["id"], m["frame"])
        mode = body.get("mode", "point")
        lonlat = (float(body["lon"]), float(body["lat"])) if mode == "point" else None
        try:
            ref = fr.set_reference(lonlat, mode=mode)
        except ValueError as e:
            raise HttpError(400, str(e)) from e
        fr.export_geotiffs()
        return {
            "reference": {k: v for k, v in ref.items() if k != "series"},
            "meta": fr.meta(),
        }


def make_handler(app: Studio):
    """Build the request handler class for ``app``."""
    routes = [(meth, re.compile(pat + r"$"), fn) for meth, pat, fn in app.routes()]

    class Handler(BaseHTTPRequestHandler):
        server_version = "OperaStudio/1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter than stderr per request
            log.debug("%s %s", self.address_string(), fmt % args)

        # helpers
        def json(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise HttpError(413, "request too large")
            raw = self.rfile.read(n) if n else b"{}"
            try:
                return json.loads(raw or b"{}")
            except json.JSONDecodeError as e:
                raise HttpError(400, f"bad JSON: {e}") from e

        def _cors(self) -> None:
            origin = getattr(self, "cors_origin", None)
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def send_bytes(
            self, data: bytes, ctype: str, status: int = 200, cache: bool = True
        ):
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=60" if cache else "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def send_json(self, obj, status: int = 200):
            self.send_bytes(
                json.dumps(obj, default=str).encode(),
                "application/json",
                status,
                cache=False,
            )

        def send_file(
            self, path: Path, attachment: bool = False, name: str | None = None
        ):
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            size = path.stat().st_size
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            if attachment:
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{name or path.name}"'
                )
            self.end_headers()
            with open(path, "rb") as f:
                while chunk := f.read(1 << 20):
                    self.wfile.write(chunk)

        def _check_origin(self) -> None:
            """Allow the same origin, or an allowed one (answered with CORS headers)."""
            self.cors_origin = None
            host = self.headers.get("Host", "")
            if app.hosts is not None and host.lower() not in app.hosts:
                raise HttpError(403, "unexpected Host header")
            origin = self.headers.get("Origin")
            if not origin:
                return
            if urlparse(origin).netloc == host:
                return
            if origin in app.origins:
                self.cors_origin = origin
                return
            raise HttpError(403, "cross-origin request refused (see --allow-origin)")

        def do_OPTIONS(self):
            try:
                self._check_origin()
            except HttpError as e:
                self.send_json({"error": str(e)}, e.status)
                return
            self.send_response(204)
            self._cors()
            self.send_header(
                "Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"
            )
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            if self.headers.get("Access-Control-Request-Private-Network") == "true":
                self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _dispatch(self, method: str):
            url = urlparse(self.path)
            self.query = parse_qs(url.query)
            path = url.path
            try:
                self._check_origin()
                self._route(method, path)
            except HttpError as e:
                self.send_json({"error": str(e)}, e.status)
            except FileNotFoundError:
                self.send_json({"error": "not found"}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log.exception("%s %s failed", method, path)
                self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _route(self, method: str, path: str) -> None:
            if path.startswith("/api/"):
                for meth, rx, fn in routes:
                    mm = rx.match(path)
                    if mm and meth == method:
                        out = fn(self, mm.groupdict())
                        if out is not None:
                            self.send_json(out)
                        return
                raise HttpError(404, "not found")
            if method != "GET":
                raise HttpError(405, "method not allowed")
            self._static(path)

        def _static(self, path: str):
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            p = (STATIC / rel).resolve()
            if STATIC.resolve() not in p.parents or not p.is_file():
                p = STATIC / "index.html"  # single-page app
            data = _read_static(p)
            ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
            if p.suffix == ".js":
                ctype = "text/javascript"
            self.send_bytes(
                data,
                ctype + ("; charset=utf-8" if ctype.startswith("text") else ""),
                cache=False,
            )

        def do_GET(self):
            self._dispatch("GET")

        def do_HEAD(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_DELETE(self):
            self._dispatch("DELETE")

    return Handler


@lru_cache(maxsize=64)
def _read_cached(path: str, _mtime: float) -> bytes:
    return Path(path).read_bytes()


def _read_static(p: Path) -> bytes:
    return _read_cached(str(p), p.stat().st_mtime)


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    workdir: Path = Path("./opera-studio"),
    open_browser: bool = False,
    max_running: int = 2,
    nisar_frames: Path | None = None,
    allow_origin: list[str] | tuple[str, ...] = (),
):
    """Serve the Studio UI and API on ``host:port`` until interrupted."""
    origins = allowed_origins(allow_origin)
    app = Studio(
        workdir,
        max_running=max_running,
        nisar_gpkg=nisar_frames,
        origins=origins,
        hosts=local_hosts(host, port),
    )
    app.start_background()
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    shown_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    url = f"http://{shown_host}:{httpd.server_address[1]}"
    print(f"OPERA Studio on {url}  (jobs in {app.workdir / 'jobs'})", flush=True)
    for o in origins:
        print(f"  allowing the hosted interface at {o}", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app._stop.set()
        httpd.server_close()


def main() -> None:
    """Run the Studio server from the command line."""
    p = argparse.ArgumentParser(
        prog="python -m opera_utils.studio",
        description="OPERA Studio: search, subset and download OPERA products.",
    )
    p.add_argument(
        "--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1)"
    )
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--workdir",
        type=Path,
        default=Path("./opera-studio"),
        help="where job folders go",
    )
    p.add_argument("--max-running", type=int, default=2, help="jobs running at once")
    p.add_argument("--open", action="store_true", help="open a browser")
    p.add_argument(
        "--nisar-frames",
        type=Path,
        default=None,
        help=(
            "NISAR track-frame GeoPackage (NISAR_TrackFrame_*.gpkg) for the real frame"
            " outlines; default: $OPERA_STUDIO_NISAR_GPKG or one in the opera-utils"
            " data cache"
        ),
    )
    p.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help=(
            "a web origin allowed to use this server, e.g. the hosted interface"
            " https://<user>.github.io (repeatable; also $OPERA_STUDIO_ALLOW_ORIGIN)"
        ),
    )
    a = p.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    serve(
        a.host,
        a.port,
        a.workdir,
        a.open,
        a.max_running,
        a.nisar_frames,
        a.allow_origin,
    )


if __name__ == "__main__":
    main()
