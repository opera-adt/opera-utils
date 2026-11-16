"""Tests for opera_utils.studio (no network)."""

from __future__ import annotations

import json
import os
import stat
import threading
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np
import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")
pytest.importorskip("scipy")

from opera_utils.studio import auth as auth_mod
from opera_utils.studio import rebase, search, subset
from opera_utils.studio.jobs import JobError, JobManager

BOX = {
    "type": "Polygon",
    "coordinates": [
        [
            [-119.6, 36.07],
            [-119.53, 36.07],
            [-119.53, 36.12],
            [-119.6, 36.12],
            [-119.6, 36.07],
        ]
    ],
}


# -- rebase: pure maths --------------------------------------------------------
def test_fit_velocity_matches_polyfit_and_skips_nans():
    years = np.array([0.0, 0.1, 0.35, 0.5, 0.9, 1.2])
    rng = np.random.default_rng(0)
    cube = 0.02 * years[:, None, None] + rng.normal(0, 1e-3, (6, 2, 3))
    cube[2, 0, 0] = np.nan
    slope, n = rebase.fit_velocity(cube, years)
    ok = np.isfinite(cube[:, 0, 0])
    assert slope[0, 0] == pytest.approx(
        np.polyfit(years[ok], cube[ok, 0, 0], 1)[0], rel=1e-5
    )
    assert slope[1, 2] == pytest.approx(
        np.polyfit(years, cube[:, 1, 2], 1)[0], rel=1e-5
    )
    assert n[0, 0] == 5
    assert n[1, 1] == 6


def test_fit_velocity_needs_min_epochs():
    years = np.array([0.0, 0.5, 1.0])
    cube = np.array([1.0, np.nan, np.nan])[:, None, None]
    slope, n = rebase.fit_velocity(cube, years)
    assert np.isnan(slope[0, 0])
    assert n[0, 0] == 1


def test_pick_reference_pixel_finds_coherent_spot_and_snaps():
    coh = np.full((40, 40), 0.3)
    coh[8:15, 8:15] = 0.95  # best spot
    coh[28:35, 28:35] = 0.8  # second spot
    valid = np.full((40, 40), 50)
    r, c, method = rebase.pick_reference_pixel(coh, valid)
    assert 8 <= r < 15
    assert 8 <= c < 15
    assert method == "most coherent spot"
    # a user point near the second spot snaps to it, not to the global best
    r, c, method = rebase.pick_reference_pixel(coh, valid, near=(31, 31))
    assert 28 <= r < 35
    assert 28 <= c < 35
    assert method == "user point"


def test_pick_reference_pixel_ignores_pixels_missing_epochs():
    coh = np.full((30, 30), 0.5)
    coh[5:10, 5:10] = 0.99
    valid = np.full((30, 30), 40)
    valid[5:10, 5:10] = 20  # very coherent but half the epochs missing
    r, c, _ = rebase.pick_reference_pixel(coh, valid)
    assert not ((5 <= r < 10) and (5 <= c < 10))


def test_pick_reference_pixel_errors():
    coh = np.full((10, 10), np.nan)
    with pytest.raises(ValueError, match="observed"):
        rebase.pick_reference_pixel(coh, np.ones((10, 10)))
    with pytest.raises(ValueError, match="outside"):
        rebase.pick_reference_pixel(np.ones((10, 10)), np.ones((10, 10)), near=(50, 50))


def test_window_median_per_epoch():
    cube = np.arange(2 * 7 * 7, dtype=float).reshape(2, 7, 7)
    cube[0, 3, 3] = np.nan
    med = rebase.window_median(cube, 3, 3)
    assert med.shape == (2,)
    assert med[1] == np.median(cube[1, 1:6, 1:6])


def test_colorize_and_png():
    v = np.array([[-1.0, 0.0, 1.0], [np.nan, 0.5, -0.5]])
    img = rebase.colorize(v, vmax=1.0)
    assert img.shape == (2, 3, 4)
    assert img[1, 0, 3] == 0  # NaN transparent
    assert img[0, 1, :3].tolist() == [247, 247, 247]  # zero is the light centre
    assert img[0, 2, 2] > img[0, 2, 0]  # positive (toward) is blue
    png = rebase.encode_png(img)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    # IDAT decompresses to rows of filter byte + RGBA
    idat = png[png.index(b"IDAT") + 4 :]
    raw = zlib.decompressobj().decompress(idat)
    assert len(raw) == 2 * (1 + 3 * 4)


# -- rebase: a synthetic stack, read as the UI does ------------------------------
@pytest.fixture
def frame(tmp_path):
    import xarray as xr
    from pyproj import CRS

    nt, ny, nx = 12, 30, 40
    times = np.array(
        [np.datetime64("2023-01-01") + np.timedelta64(12 * i, "D") for i in range(nt)]
    ).astype("datetime64[ns]")
    years = rebase.years_since_first(times)
    x = 300015.0 + 30 * np.arange(nx)
    y = 4000015.0 - 30 * np.arange(ny)
    # pixel (r, c) moves at -0.01 * (c / nx) m/yr; reference area moves at 0
    vel = -0.01 * (np.arange(nx)[None, :] / nx) * np.ones((ny, 1))
    disp = (years[:, None, None] * vel[None]).astype(np.float32)
    coh = np.full((ny, nx), 0.4, np.float32)
    coh[2:9, 2:9] = 0.95
    crs = CRS.from_epsg(32611)
    ds = xr.Dataset(
        {
            "displacement": (("time", "y", "x"), disp),
            "average_temporal_coherence": (("y", "x"), coh),
            "spatial_ref": ((), 0, {"crs_wkt": crs.to_wkt()}),
        },
        coords={"time": times, "y": y, "x": x},
    )
    fdir = tmp_path / "frames" / "F00001"
    fdir.mkdir(parents=True)
    ds.to_zarr(fdir / "stack.zarr", consolidated=False)
    return rebase.Frame(fdir, tmp_path / "download" / "F00001"), vel


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning"
)  # inside rasterio
def test_frame_analysis_reference_series_and_export(frame):
    fr, vel = frame
    a = fr.analysis
    np.testing.assert_allclose(a["velocity"], vel, atol=1e-5)
    assert (a["valid"] == 12).all()

    ref = fr.set_reference(mode="auto")
    assert ref["mode"] == "point"
    assert 2 <= ref["row"] < 9
    assert 2 <= ref["col"] < 9
    ref_v = vel[ref["row"], ref["col"]]
    np.testing.assert_allclose(fr.velocity(), vel - ref_v, atol=1e-5)

    lon, lat = fr.rowcol_to_lonlat(15, 30)
    assert fr.lonlat_to_rowcol(lon, lat) == (15, 30)
    s = fr.series(lon, lat)
    assert len(s["dates"]) == 12
    assert s["referenced"][0] == pytest.approx(0, abs=1e-6)
    assert s["velocity_m_yr"] == pytest.approx(vel[15, 30] - ref_v, abs=1e-5)
    assert "raw" not in s  # no NetCDF subsets in this fixture

    with pytest.raises(ValueError, match="outside"):
        fr.series(lon + 5, lat)

    tifs = fr.export_geotiffs()
    import rasterio

    vel_tif = next(p for p in tifs if p.name.startswith("velocity"))
    with rasterio.open(vel_tif) as src:
        np.testing.assert_allclose(src.read(1), vel - ref_v, atol=1e-5)
        assert json.loads(src.tags()["reference"])["mode"] == "point"

    area = fr.set_reference(mode="area")
    assert area["mode"] == "area"
    np.testing.assert_allclose(fr.velocity(), vel, atol=1e-5)
    meta = fr.meta()
    assert meta["n_epochs"] == 12
    assert len(meta["corners"]) == 4
    assert fr.velocity_png(min_coherence=0.9).startswith(b"\x89PNG")


def test_frame_user_point_snaps_nearby(frame):
    fr, _ = frame
    lon, lat = fr.rowcol_to_lonlat(20, 20)
    ref = fr.set_reference((lon, lat), mode="point")
    assert ref["method"] == "user point"
    assert abs(ref["row"] - 20) <= rebase.SNAP_PIXELS
    assert abs(ref["col"] - 20) <= rebase.SNAP_PIXELS
    assert ref["requested"] == [round(lon, 6), round(lat, 6)]


def test_raw_series_reads_subsets_with_moving_reference(frame, tmp_path):
    fr, _ = frame
    dl = fr.download_dir
    dl.mkdir(parents=True)
    names = [
        ("20230101T000000Z", "20230113T000000Z", 0.01),
        ("20230101T000000Z", "20230125T000000Z", 0.02),
        ("20230125T000000Z", "20230206T000000Z", 0.005),  # new reference date
    ]
    for ref, sec, val in names:
        f = dl / f"OPERA_L3_DISP-S1_IW_F00001_VV_{ref}_{sec}_v1.0_20250101T000000Z.nc"
        with h5py.File(f, "w") as hf:
            hf["displacement"] = np.full((30, 40), val, np.float32)
    raw = fr.raw_series(5, 5)
    assert raw["values"] == [0.01, 0.02, 0.005]
    assert raw["reference_dates"][2].startswith("2023-01-25")


# -- auth ------------------------------------------------------------------------------
def test_auth_precedence_and_job_env(tmp_path, monkeypatch):
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    store = auth_mod.AuthStore(tmp_path / "cfg", home=home)
    assert store.status()["signed_in"] is False

    monkeypatch.setenv("EARTHDATA_USERNAME", "envuser")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "envpw")
    assert store.status()["source"] == "env"

    (home / ".netrc").write_text(
        "machine urs.earthdata.nasa.gov login nu password np\n"
    )
    os.chmod(home / ".netrc", 0o600)
    assert store.status()["username"] == "nu"
    env = store.job_env({"HOME": str(home)})
    assert env["HOME"] == str(home)  # the user's own netrc: nothing to override

    store.sign_in("uiuser", "uipw", remember=True, check=False)
    st = store.status()
    assert (st["source"], st["username"], st["can_sign_out"]) == (
        "session",
        "uiuser",
        True,
    )
    saved = tmp_path / "cfg" / "netrc"
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600

    env = store.job_env({"HOME": str(home)})
    job_netrc = Path(env["HOME"]) / ".netrc"
    assert "login uiuser password uipw" in job_netrc.read_text()
    assert stat.S_IMODE(job_netrc.stat().st_mode) == 0o600
    assert env["XDG_CACHE_HOME"] == str(home / ".cache")
    assert env["EARTHDATA_USERNAME"] == "uiuser"
    assert (
        (home / ".netrc")
        .read_text()
        .startswith("machine urs.earthdata.nasa.gov login nu")
    )

    store.sign_out()
    assert not saved.exists()
    assert store.status()["username"] == "nu"


def test_auth_rejects_spaces_and_bad_login(monkeypatch, tmp_path):
    with pytest.raises(auth_mod.AuthError):
        auth_mod.netrc_text("a b", "pw")

    class R:
        status_code = 401

    monkeypatch.setattr(auth_mod.requests, "get", lambda *_a, **_k: R())
    store = auth_mod.AuthStore(tmp_path / "cfg", home=tmp_path)
    with pytest.raises(auth_mod.AuthError, match="rejected"):
        store.sign_in("user", "wrong")
    assert store._session is None


# -- search helpers --------------------------------------------------------------------
def test_parse_aoi_and_dates():
    g = search.parse_aoi(BOX)
    assert g.bounds == pytest.approx((-119.6, 36.07, -119.53, 36.12))
    assert search.parse_aoi({"type": "Feature", "geometry": BOX}).equals(g)
    with pytest.raises(search.SearchError, match="Draw"):
        search.parse_aoi(None)
    with pytest.raises(search.SearchError, match="degrees"):
        search.parse_aoi(
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [500000, 0], [500000, 10], [0, 0]]],
            }
        )
    assert search.parse_day("2024-02-03", end=True).hour == 23
    with pytest.raises(search.SearchError):
        search.parse_day("yesterday")
    assert 30 < search.area_km2(g) < 45


def test_run_search_validates(monkeypatch):
    with pytest.raises(search.SearchError, match="Unknown product"):
        search.run_search({"product": "rtc", "aoi": BOX})
    with pytest.raises(search.SearchError, match="after"):
        search.run_search(
            {"product": "disp", "aoi": BOX, "start": "2024-02-01", "end": "2024-01-01"}
        )
    seen = {}
    monkeypatch.setitem(
        search.SEARCHES,
        "cslc",
        lambda _aoi, _s, _e, o: (
            seen.update(o=o)
            or {"groups": [], "items": [], "footprints": {}, "summary": {}}
        ),
    )
    out = search.run_search(
        {"product": "cslc", "aoi": BOX, "options": {"direction": "ascending"}}
    )
    assert out["product"] == "cslc"
    assert seen["o"]["direction"] == "ascending"


def test_tropo_time_and_coverage():
    t = search._tropo_time(
        "OPERA_L4_TROPO-ZENITH_20240101T060000Z_20250924T033947Z_HRES_v1.0"
    )
    assert (t.year, t.hour) == (2024, 6)
    aoi = search.parse_aoi(BOX)
    from shapely.geometry import box

    assert search._coverage(aoi, box(-119.6, 36.07, -119.565, 36.12)) == pytest.approx(
        0.5, abs=1e-6
    )


def test_burst_outlines_use_burst_database(monkeypatch):
    from opera_utils import burst_frame_db

    asked = {}

    def fake(ids):
        asked["ids"] = ids
        return {
            "features": [
                {"properties": {"burst_id_jpl": "t137_292327_iw2"}, "geometry": BOX}
            ]
        }

    monkeypatch.setattr(burst_frame_db, "get_burst_id_geojson", fake)
    out = search.burst_outlines(["T137_292327_IW2"])
    assert asked["ids"] == ["t137_292327_iw2"]
    assert out == {"t137_292327_iw2": BOX}
    assert search.burst_outlines([]) == {}


def test_umm_polygon():
    umm = {
        "SpatialExtent": {
            "HorizontalSpatialDomain": {
                "Geometry": {
                    "GPolygons": [
                        {
                            "Boundary": {
                                "Points": [
                                    {"Longitude": 0, "Latitude": 0},
                                    {"Longitude": 1, "Latitude": 0},
                                    {"Longitude": 1, "Latitude": 1},
                                    {"Longitude": 0, "Latitude": 0},
                                ]
                            }
                        }
                    ]
                }
            }
        }
    }
    g = search._umm_polygon(umm)
    assert g["type"] == "MultiPolygon"
    assert g["coordinates"][0][0][1] == [1, 0]


# -- subset ------------------------------------------------------------------------------
def test_subset_h5_cuts_grid_and_copies_rest(tmp_path):
    from pyproj import Transformer

    epsg = 32611
    x = 300015.0 + 30 * np.arange(200)
    y = 4000015.0 - 30 * np.arange(150)
    src = tmp_path / "cslc.h5"
    with h5py.File(src, "w") as f:
        d = f.create_group("data")
        d["VV"] = (np.arange(150 * 200).reshape(150, 200) * (1 + 1j)).astype(
            np.complex64
        )
        d["x_coordinates"] = x
        d["y_coordinates"] = y
        d["projection"] = np.int32(epsg)
        d["x_spacing"] = 30.0
        d["VV"].attrs["units"] = "unitless"
        f.create_group("metadata")["orbit"] = np.arange(5)
        f.create_group("identification")["burst_id"] = b"t071_151229_iw2"
    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    w, s = to_ll.transform(x[50], y[100])
    e, n = to_ll.transform(x[80], y[60])
    out = tmp_path / "sub.h5"
    win = subset.subset_h5(src, out, (w, s, e, n))
    r0, r1 = win["rows"]
    c0, c1 = win["cols"]
    assert r0 <= 60
    assert r1 >= 101
    assert c0 <= 50
    assert c1 >= 81
    assert (r1 - r0) < 60
    assert (c1 - c0) < 45
    with h5py.File(out) as f, h5py.File(src) as g:
        np.testing.assert_array_equal(f["data/VV"][()], g["data/VV"][r0:r1, c0:c1])
        np.testing.assert_array_equal(f["data/x_coordinates"][()], x[c0:c1])
        assert f["data/VV"].attrs["units"] == "unitless"
        assert f["data/x_spacing"][()] == 30.0
        assert f["identification/burst_id"][()] == b"t071_151229_iw2"
        assert f["metadata/orbit"].shape == (5,)
    with pytest.raises(subset.SubsetError, match="overlap"):
        subset.subset_h5(src, tmp_path / "none.h5", (10, 10, 11, 11))


# -- jobs ------------------------------------------------------------------------------------
def test_job_manager_create_record_and_guards(tmp_path):
    jm = JobManager(tmp_path / "jobs", max_running=0)  # queue only, never launch
    with pytest.raises(JobError, match="Pick"):
        jm.create({"product": "disp", "aoi": BOX, "groups": []})
    with pytest.raises(JobError, match="Unknown"):
        jm.create({"product": "rtc", "aoi": BOX, "groups": ["x"]})
    rec = jm.create(
        {
            "product": "disp",
            "aoi": BOX,
            "groups": ["F38502"],
            "items": [
                {"group": "F38502", "url": "u1"},
                {"group": "F00001", "url": "u2"},
            ],
            "footprints": [
                {"type": "Feature", "geometry": BOX, "properties": {"id": "F38502"}}
            ],
        }
    )
    assert rec["state"] == "queued"
    assert rec["n_items"] == 1
    assert "DISP-S1" in rec["title"]
    assert [j["id"] for j in jm.list()] == [rec["id"]]
    with pytest.raises(FileNotFoundError):
        jm.file_path(rec["id"], "../../etc/passwd")
    with pytest.raises(JobError):
        jm.record("../x")
    assert jm.cancel(rec["id"])["state"] == "cancelled"
    jm.delete(rec["id"])
    assert jm.list() == []


def test_job_manager_marks_dead_worker_failed(tmp_path):
    # A worker that exits without writing a final status.json.
    jm = JobManager(tmp_path / "jobs", max_running=1, python=str(tmp_path / "py"))
    (tmp_path / "py").write_text("#!/bin/sh\nexit 3\n")
    os.chmod(tmp_path / "py", 0o755)
    rec = jm.create({"product": "tropo", "aoi": BOX, "groups": ["2024-01-01"]})
    jm._procs[rec["id"]].wait(timeout=60)
    jm.tick()
    assert jm.record(rec["id"])["state"] == "failed"


# -- server ----------------------------------------------------------------------------------
@pytest.fixture
def server(tmp_path, monkeypatch):
    from http.server import ThreadingHTTPServer

    from opera_utils.studio import server as srv

    monkeypatch.setattr(
        srv,
        "run_search",
        lambda req: {
            "product": req["product"],
            "groups": [],
            "items": [],
            "footprints": {},
            "summary": {},
        },
    )
    app = srv.Studio(tmp_path / "work", config_dir=tmp_path / "cfg", max_running=0)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(app))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    app._stop.set()
    httpd.shutdown()
    httpd.server_close()


def _req(url, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read(), r.headers.get("Content-Type")
    except urllib.error.HTTPError as e:
        with e:  # an HTTPError holds the response: close it (ResourceWarning)
            return e.code, e.read(), e.headers.get("Content-Type")


def test_server_api_and_static(server):
    code, body, _ = _req(server + "/api/health")
    assert code == 200
    assert json.loads(body)["ok"]
    code, body, ctype = _req(server + "/")
    assert code == 200
    assert b"OPERA Studio" in body
    assert ctype.startswith("text/html")
    code, body, ctype = _req(server + "/app.js")
    assert code == 200
    assert ctype.startswith("text/javascript")
    code, body, _ = _req(
        server + "/api/search", "POST", {"product": "disp", "aoi": BOX}
    )
    assert code == 200
    assert json.loads(body)["product"] == "disp"
    code, body, _ = _req(
        server + "/api/jobs", "POST", {"product": "disp", "aoi": BOX, "groups": []}
    )
    assert code == 400
    assert "Pick" in json.loads(body)["error"]
    code, _, _ = _req(server + "/api/nope")
    assert code == 404
    code, _, _ = _req(server + "/api/jobs/x/disp/F12345/meta")
    assert code == 404


def test_server_refuses_cross_origin_writes(server):
    code, _body, _ = _req(
        server + "/api/auth",
        "POST",
        {"username": "a", "password": "b"},
        headers={"Origin": "https://evil.example"},
    )
    assert code == 403


# -- outlines, archive depth, timelines ---------------------------------------------------
from opera_utils.studio import outlines as ol  # noqa: E402


def test_nisar_index_decodes_track_frame():
    # checked against CMR: 034_A_020 -> 5827, 114_D_070 -> 19957
    assert ol.nisar_index_to_track_frame(5827) == (34, 20)
    assert ol.nisar_index_to_track_frame(19957) == (114, 70)
    assert ol.nisar_index_to_track_frame(0) == (1, 1)


def test_projected_box_to_lonlat():
    from pyproj import Transformer

    poly = ol.projected_box_to_lonlat(32611, 300000, 3960000, 400000, 4000000)
    lon, lat = Transformer.from_crs(32611, 4326, always_xy=True).transform(
        350000, 3980000
    )
    from shapely.geometry import Point

    assert poly.contains(Point(lon, lat))
    assert len(poly.exterior.coords) > 40  # densified


def test_summarize_dates_revisit_and_gaps():
    days = [
        "2024-01-01",
        "2024-01-13",
        "2024-01-25",
        "2024-02-06",
        "2024-05-06",
        "2024-05-18",
        "2024-05-18",
    ]
    s = ol.summarize_dates(days)
    assert s["n_dates"] == 6
    assert s["first"] == "2024-01-01"
    assert s["last"] == "2024-05-18"
    assert s["revisit_days"] == 12
    assert s["gaps"] == [["2024-02-06", "2024-05-06", 90]]
    assert ol.summarize_dates([])["first"] is None


def test_granule_dates_by_product():
    disp = {
        "GranuleUR": (
            "OPERA_L3_DISP-S1_IW_F38502_VV_20230206T135329Z_20230606T135332Z_v1.0_20250101T000000Z"
        )
    }
    assert ol._granule_dates("frames", None, disp) == ("2023-06-06", "2023-02-06")
    cslc = {
        "GranuleUR": (
            "OPERA_L2_CSLC-S1_T071-151229-IW2_20240127T135301Z_20240504T071841Z_S1A_VV_v1.1"
        )
    }
    assert ol._granule_dates("bursts", None, cslc) == ("2024-01-27", None)
    gunw = {
        "GranuleUR": (
            "NISAR_L2_PR_GUNW_007_034_A_019_009_4000_SH_20251206T130825_20251206T130900_20251230T130826_20251230T130901_X05010_N_F_J_001"
        )
    }
    assert ol._granule_dates("nisar", "gunw", gunw) == ("2025-12-30", "2025-12-06")


def test_cmr_params_per_layer():
    o = ol.Outlines()
    p = o.cmr_params("frames", "F38502", None, "2024-01-01", None)
    assert p["short_name"] == "OPERA_L3_DISP-S1_V1"
    assert "int,FRAME_NUMBER,38502" in p["attribute[]"]
    assert "float,PRODUCT_VERSION,1.0" in p["attribute[]"]
    assert p["temporal"] == "2024-01-01T00:00:00Z,"
    p = o.cmr_params("frames", "F38502", "static", "2024-01-01", None)
    assert p["short_name"] == "OPERA_L3_DISP-S1-STATIC_V1"
    assert "temporal" not in p
    assert o.cmr_params("bursts", "T071_151229_IW2", None, None, None)[
        "attribute[]"
    ] == ["string,OPERA_BURST_ID,T071_151229_IW2"]
    p = o.cmr_params("nisar", "034_020", "gunw", None, "2025-12-31")
    assert p["short_name"] == "NISAR_L2_GUNW_BETA_V1"
    assert p["attribute[]"] == ["int,TRACK_NUMBER,34", "int,FRAME_NUMBER,20"]
    for layer, bad in (
        ("frames", "38502"),
        ("bursts", "T71-1"),
        ("nisar", "34_A_20"),
        ("roads", "x"),
    ):
        with pytest.raises(ol.OutlineError):
            o.cmr_params(layer, bad, None, None, None)


def test_depth_counts_and_caches(tmp_path):
    calls = []

    class Resp:
        status_code = 200

        def __init__(self, n):
            self.headers = {"CMR-Hits": str(n)}

        def raise_for_status(self):
            pass

    class Session:
        def get(self, _url, params, **_kw):
            calls.append(params)
            frame = int(params["attribute[]"][0].split(",")[2])
            return Resp(frame % 1000)

    cache = tmp_path / "depth.json"
    o = ol.Outlines(cache_file=cache)
    o.session = Session()
    out = o.depth("frames", ["F38502", "F00007", "F38502"])
    assert out == {"F38502": 502, "F00007": 7}
    assert len(calls) == 2
    assert all(c["page_size"] == 0 for c in calls)
    o.depth("frames", ["F38502"])
    assert len(calls) == 2  # cached
    o2 = ol.Outlines(cache_file=cache)  # persisted across restarts
    o2.session = Session()
    assert o2.depth("frames", ["F00007"]) == {"F00007": 7}
    assert len(calls) == 2


def test_timeline_from_cmr(monkeypatch):
    import opera_utils._cmr as cmr

    urs = [
        "OPERA_L3_DISP-S1_IW_F00001_VV_20230101T000000Z_20230113T000000Z_v1.0_20250101T000000Z",
        "OPERA_L3_DISP-S1_IW_F00001_VV_20230101T000000Z_20230125T000000Z_v1.0_20250101T000000Z",
        "OPERA_L3_DISP-S1_IW_F00001_VV_20230125T000000Z_20230206T000000Z_v1.0_20250101T000000Z",
    ]
    monkeypatch.setattr(
        cmr,
        "fetch_cmr_pages",
        lambda _url, _params: [{"umm": {"GranuleUR": u}} for u in urs],
    )
    o = ol.Outlines()
    lay = ol._Layer(["F00001"], [shape_box()], [{"id": "F00001"}])
    o._layers["frames"] = lay
    tl = o.timeline("frames", "F00001")
    assert tl["dates"] == ["2023-01-13", "2023-01-25", "2023-02-06"]
    assert tl["reference_dates"] == ["2023-01-01", "2023-01-25"]
    assert tl["n_files"] == 3
    assert tl["collection"] == "OPERA_L3_DISP-S1_V1"
    assert tl["geometry"]["type"] == "Polygon"


def shape_box():
    from shapely.geometry import box

    return box(-119.6, 36.0, -119.5, 36.1)


def test_layer_query_limits():
    from shapely.geometry import box

    geoms = [box(i, 0, i + 0.9, 1) for i in range(10)]
    layer = ol._Layer(
        [f"F{i:05d}" for i in range(10)],
        geoms,
        [{"id": f"F{i:05d}"} for i in range(10)],
    )
    feats, truncated = layer.query((2.5, 0.2, 5.5, 0.8), limit=10)
    assert [f["id"] for f in feats] == ["F00002", "F00003", "F00004", "F00005"]
    assert not truncated
    feats, truncated = layer.query((0, 0, 10, 1), limit=3)
    assert len(feats) == 3
    assert truncated


def test_static_routes_frames_and_bursts(monkeypatch, tmp_path):
    from opera_utils.studio import worker

    assert worker.is_frame_id("F38502")
    assert not worker.is_frame_id("T071_151229_IW2")
    seen = {}
    monkeypatch.setattr(
        worker, "_disp_static", lambda _job, _d, _st, frames: seen.update(frames=frames)
    )
    monkeypatch.setattr(
        worker, "run_cslc", lambda job, _d, _st: seen.update(bursts=job["groups"])
    )
    worker.run_static(
        {"groups": ["F38502", "T071_151229_IW2"], "aoi": BOX}, tmp_path, None
    )
    assert seen == {"frames": ["F38502"], "bursts": ["T071_151229_IW2"]}


def test_static_search_returns_both_kinds(monkeypatch):
    disp_part = {
        "groups": [{"id": "F00001", "kind": "disp", "n": 1}],
        "items": [{"group": "F00001"}],
        "footprints": {
            "type": "FeatureCollection",
            "features": [{"properties": {"id": "F00001"}}],
        },
    }
    cslc_part = {
        "groups": [{"id": "T001_000001_IW1", "n": 1}],
        "items": [{"group": "T001_000001_IW1"}],
        "footprints": {
            "type": "FeatureCollection",
            "features": [{"properties": {"id": "T001_000001_IW1"}}],
        },
    }
    monkeypatch.setattr(search, "_disp_static", lambda _aoi, _d: disp_part)
    monkeypatch.setattr(search, "_asf_cslc", lambda *_a: cslc_part)
    out = search.search_static(search.parse_aoi(BOX), None, None, {})
    assert [g["kind"] for g in out["groups"]] == ["disp", "cslc"]
    assert out["summary"] == {"n_groups": 2, "n_items": 2}
    out = search.search_static(search.parse_aoi(BOX), None, None, {"kinds": ["disp"]})
    assert [g["id"] for g in out["groups"]] == ["F00001"]


def test_server_outline_routes_validate(server):
    code, body, _ = _req(server + "/api/outlines/frames")
    assert code == 400
    assert "bbox" in json.loads(body)["error"]
    code, _, _ = _req(server + "/api/outlines/roads?bbox=0,0,1,1")
    assert code == 404
    code, _, _ = _req(server + "/api/timeline/frames/notaframe")
    assert code == 404


# -- antimeridian -----------------------------------------------------------------------------
from shapely.geometry import MultiPolygon  # noqa: E402
from shapely.geometry import Polygon as SPolygon  # noqa: E402

from opera_utils.studio import geo  # noqa: E402

# A frame cut at ±180 the way the burst / NISAR databases store it.
CUT = MultiPolygon(
    [
        SPolygon([(178, 51), (180, 51), (180, 52), (178, 52)]),
        SPolygon([(-180, 51), (-178, 51), (-178, 52), (-180, 52)]),
    ]
)


def test_unwrap_joins_cut_polygon():
    u = geo.unwrap(CUT)
    assert u.geom_type == "Polygon"
    assert u.bounds == pytest.approx((178, 51, 182, 52))
    assert geo.crosses(u)
    # a ring whose vertices jump across the line
    jump = SPolygon([(179, 0), (-179, 0), (-179, 1), (179, 1)])
    assert geo.unwrap(jump).bounds == pytest.approx((179, 0, 181, 1))
    # ordinary polygons are left alone, Z dropped
    plain = SPolygon([(-120, 35, 5), (-119, 35, 5), (-119, 36, 5)])
    assert geo.unwrap(plain).bounds == pytest.approx((-120, 35, -119, 36))
    assert not geo.crosses(geo.unwrap(plain))


def test_wrap_parts_and_normalize():
    drawn = SPolygon([(175, 50), (185, 50), (185, 53), (175, 53)])
    parts = geo.wrap_parts(drawn)
    assert sorted(p.bounds for p in parts) == [
        pytest.approx((-180, 50, -175, 53)),
        pytest.approx((175, 50, 180, 53)),
    ]
    # an area drawn one world copy east comes back to [-180, 180)
    copy = SPolygon([(240, 30), (241, 30), (241, 31), (240, 31)])
    assert geo.normalize(copy).bounds == pytest.approx((-120, 30, -119, 31))


def test_intersect_lon_and_coverage_across_the_line():
    aoi = SPolygon(
        [(-179.5, 51.2), (-178.5, 51.2), (-178.5, 51.8), (-179.5, 51.8)]
    )  # west side
    frame = geo.unwrap(CUT)  # 178..182
    assert geo.coverage(aoi, frame) == pytest.approx(1.0)
    inter = geo.intersect_lon(aoi, frame)
    assert inter.area == pytest.approx(aoi.area)
    far = SPolygon([(10, 0), (11, 0), (11, 1), (10, 1)])
    assert geo.coverage(far, frame) == 0


def test_parse_aoi_across_the_line():
    g = search.parse_aoi(
        {
            "type": "Polygon",
            "coordinates": [[[175, 50], [185, 50], [185, 53], [175, 53], [175, 50]]],
        }
    )
    assert g.bounds == pytest.approx((175, 50, 185, 53))  # kept continuous
    g = search.parse_aoi(
        {
            "type": "Polygon",
            "coordinates": [[[240, 30], [241, 30], [241, 31], [240, 31], [240, 30]]],
        }
    )
    assert g.bounds == pytest.approx((-120, 30, -119, 31))
    with pytest.raises(search.SearchError, match="around"):
        search.parse_aoi(
            {
                "type": "Polygon",
                "coordinates": [[[-200, 0], [200, 0], [200, 1], [-200, 1], [-200, 0]]],
            }
        )


def test_layer_query_finds_unwrapped_outlines_from_either_side():
    layer = ol._Layer(
        ["F00001", "F00002"],
        [CUT, SPolygon([(-120, 35), (-119, 35), (-119, 36)])],
        [{"id": "F00001"}, {"id": "F00002"}],
    )
    assert layer.geoms[0].geom_type == "Polygon"  # stored unwrapped
    for bbox in [(-179, 51, -178.5, 52), (178.5, 51, 179, 52), (181, 51, 181.5, 52)]:
        feats, _ = layer.query(bbox, 10)
        assert [f["id"] for f in feats] == ["F00001"], bbox
        assert feats[0]["geometry"]["type"] == "Polygon"


def test_disp_frames_queries_each_side(monkeypatch):
    from opera_utils import burst_frame_db

    asked = []

    def fake(bounds):
        asked.append(tuple(bounds))
        return {
            "features": [
                {
                    "id": "1918",
                    "geometry": CUT.__geo_interface__,
                    "properties": {"orbit_pass": "ASCENDING"},
                }
            ]
        }

    monkeypatch.setattr(burst_frame_db, "get_intersecting_frames", fake)
    aoi = search.parse_aoi(
        {
            "type": "Polygon",
            "coordinates": [
                [[179, 51.2], [181, 51.2], [181, 51.8], [179, 51.8], [179, 51.2]]
            ],
        }
    )
    frames = search.disp_frames(aoi)
    assert len(asked) == 2  # one query per side of the line
    assert [f["frame_id"] for f in frames] == [1918]  # found once
    assert frames[0]["coverage"] == pytest.approx(1.0)
    assert frames[0]["geometry"]["type"] == "Polygon"


def test_nisar_gpkg_outlines_and_lookup(tmp_path, monkeypatch):
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("pyogrio")
    path = tmp_path / "NISAR_TrackFrame_L_test.gpkg"
    gdf = gpd.GeoDataFrame(
        {
            "track": [34, 1, 2],
            "frame": [20, 72, 5],
            "passDirection": ["Ascending", "Descending", "Ascending"],
            "hasLand": [True, True, False],
        },
        geometry=[
            MultiPolygon(
                [SPolygon([(-121, 35.2), (-118, 35.9), (-118.7, 38), (-121.4, 37.3)])]
            ),
            CUT,
            MultiPolygon([SPolygon([(0, 0), (1, 0), (1, 1)])]),
        ],
        crs="EPSG:4326",
    )
    gdf.to_file(path, layer="tfdb", driver="GPKG")
    monkeypatch.setenv("OPERA_STUDIO_NISAR_GPKG", str(path))
    assert ol.find_nisar_gpkg() == path
    layer = ol._load_nisar()
    assert layer.ids == ["034_020", "001_072"]  # sea-only frame dropped
    assert [p["direction"] for p in layer.props] == ["ascending", "descending"]
    assert layer.note is None
    assert layer.geoms[1].bounds == pytest.approx(
        (178, 51, 182, 52)
    )  # the dateline frame, whole


def test_worker_crop_box_continuous_across_line(monkeypatch, tmp_path):
    from opera_utils.studio import worker

    boxes = []
    monkeypatch.setattr(worker, "_session", lambda: None)
    monkeypatch.setattr(
        worker,
        "fetch",
        lambda _url, out, _session: (
            out.parent.mkdir(parents=True, exist_ok=True)
            or out.write_bytes(b"x")
            or out
        ),
    )
    import opera_utils.studio.subset as sub

    monkeypatch.setattr(
        sub,
        "subset_h5",
        lambda _raw, final, bbox: boxes.append(bbox) or final.write_bytes(b"y"),
    )

    class S:
        data: ClassVar[dict] = {"outputs": [], "groups": []}

        def write(self):
            pass

        def update(self, **kw):
            pass

        def group(self, *a, **k):
            pass

    aoi = {
        "type": "Polygon",
        "coordinates": [
            [[179, 51.2], [181, 51.2], [181, 51.8], [179, 51.8], [179, 51.2]]
        ],
    }
    job = {
        "aoi": aoi,
        "groups": ["T001_000001_IW1"],
        "items": [
            {"group": "T001_000001_IW1", "url": "https://x/a.h5", "name": "a.h5"}
        ],
        "footprints": [
            {
                "type": "Feature",
                "geometry": CUT.__geo_interface__,
                "properties": {"id": "T001_000001_IW1"},
            }
        ],
        "options": {"subset": True, "workers": 1},
    }
    worker.run_cslc(job, tmp_path, S())
    assert boxes
    assert boxes[0] == pytest.approx((179, 51.2, 181, 51.8))


def test_layer_query_truncates_to_nearest_view_centre():
    from shapely.geometry import box

    geoms = [box(i, 0, i + 0.5, 0.5) for i in range(10)]
    layer = ol._Layer(
        [f"F{i:05d}" for i in range(10)],
        geoms,
        [{"id": f"F{i:05d}"} for i in range(10)],
    )
    feats, truncated = layer.query((5, 0, 9.6, 1), limit=2)
    assert truncated
    assert sorted(f["id"] for f in feats) == ["F00007", "F00008"]


# -- hosted interface: CORS allowlist and Host check --------------------------------------
PAGES = "https://someone.github.io"


@pytest.fixture
def hosted_server(tmp_path, monkeypatch):
    from http.server import ThreadingHTTPServer

    from opera_utils.studio import server as srv

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = httpd.server_address[1]
    app = srv.Studio(
        tmp_path / "work",
        config_dir=tmp_path / "cfg",
        max_running=0,
        origins=srv.allowed_origins([PAGES + "/opera-utils/"]),
        hosts=srv.local_hosts("127.0.0.1", port),
    )
    httpd.RequestHandlerClass = srv.make_handler(app)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}", port
    app._stop.set()
    httpd.shutdown()
    httpd.server_close()


def _raw(port, method, path, headers):
    import http.client

    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, headers=headers)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, dict(r.getheaders()), body


def test_allowed_origins_normalised(monkeypatch):
    from opera_utils.studio import server as srv

    monkeypatch.setenv("OPERA_STUDIO_ALLOW_ORIGIN", "https://a.example/x, not-a-url")
    assert srv.allowed_origins(
        ["https://b.github.io/opera-utils/", "https://b.github.io"]
    ) == (
        "https://b.github.io",
        "https://a.example",
    )
    assert srv.local_hosts("127.0.0.1", 8765) == {
        "localhost:8765",
        "127.0.0.1:8765",
        "[::1]:8765",
    }
    assert srv.local_hosts("0.0.0.0", 8765) is None


def test_hosted_origin_gets_cors_and_private_network(hosted_server):
    _, port = hosted_server
    host = {"Host": f"localhost:{port}"}
    status, h, _ = _raw(
        port,
        "OPTIONS",
        "/api/search",
        {
            **host,
            "Origin": PAGES,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert status == 204
    assert h["Access-Control-Allow-Origin"] == PAGES
    assert h["Access-Control-Allow-Private-Network"] == "true"
    assert "POST" in h["Access-Control-Allow-Methods"]
    status, h, body = _raw(port, "GET", "/api/health", {**host, "Origin": PAGES})
    assert status == 200
    assert h["Access-Control-Allow-Origin"] == PAGES
    assert json.loads(body)["ok"]


def test_other_origins_and_rebinding_refused(hosted_server):
    _, port = hosted_server
    for headers in (
        {"Host": f"localhost:{port}", "Origin": "https://evil.example"},  # another site
        {
            "Host": f"evil.example:{port}",
            "Origin": f"http://evil.example:{port}",
        },  # DNS rebinding
        {"Host": f"evil.example:{port}"},
    ):
        status, h, _ = _raw(port, "GET", "/api/auth", headers)
        assert status == 403, headers
        assert "Access-Control-Allow-Origin" not in h
    status, _, _ = _raw(
        port,
        "OPTIONS",
        "/api/auth",
        {"Host": f"localhost:{port}", "Origin": "https://evil.example"},
    )
    assert status == 403
    # the studio's own page (same origin) and plain navigations still work
    assert (
        _raw(
            port,
            "GET",
            "/api/health",
            {"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}"},
        )[0]
        == 200
    )
    assert _raw(port, "GET", "/", {"Host": f"localhost:{port}"})[0] == 200


def test_cli_has_studio_subcommand():
    import inspect

    from opera_utils import cli

    sig = inspect.signature(cli.studio)
    assert sig.parameters["port"].default == 8765
    assert sig.parameters["allow_origin"].default == ()
