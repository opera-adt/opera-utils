"""Product searches for the studio, normalised for the UI.

Every search takes the area as a GeoJSON geometry (lon/lat) plus an optional
date range, and returns::

    {"product": ..., "kind": ..., "groups": [...], "items": [...],
     "footprints": FeatureCollection, "summary": {...}}

``groups`` are what the user picks from (a DISP frame, a CSLC burst, a NISAR
track/frame, a TROPO day); ``items`` are the granules a job will fetch, each
tagged with its ``group``.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, cast

from pyproj import Geod
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry

from . import geo

logger = logging.getLogger("opera_utils")

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
TROPO_COLLECTION = "OPERA_L4_TROPO-ZENITH_V1"
NISAR_COLLECTIONS = {
    "gslc": "NISAR_L2_GSLC_BETA_V1",
    "rslc": "NISAR_L1_RSLC_PROVISIONAL_V1",
    "gunw": "NISAR_L2_GUNW_BETA_V1",
}
PRODUCTS = ("disp", "cslc", "nisar", "tropo", "static")
_GEOD = Geod(ellps="WGS84")


class SearchError(ValueError):
    """The search request is not valid (shown to the user as-is)."""


# -- helpers -----------------------------------------------------------------
def area_km2(geom: BaseGeometry) -> float:
    """Geodesic area of a lon/lat geometry, in km²."""
    return abs(_GEOD.geometry_area_perimeter(geom)[0]) / 1e6


def parse_aoi(aoi: dict | None) -> BaseGeometry:
    """Read a GeoJSON geometry, Feature or FeatureCollection as a lon/lat area."""
    if not aoi:
        msg = "Draw an area on the map first."
        raise SearchError(msg)
    if aoi.get("type") == "Feature":
        aoi = aoi["geometry"]
    elif aoi.get("type") == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in aoi.get("features", [])]
        if not geoms:
            msg = "The area is empty."
            raise SearchError(msg)
        from shapely.ops import unary_union

        return unary_union(geoms)
    try:
        geom = shape(aoi)
    except Exception as e:
        msg = f"Could not read the area: {e}"
        raise SearchError(msg) from e
    if geom.is_empty or not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        msg = "The area is empty."
        raise SearchError(msg)
    west, south, east, north = geom.bounds
    if not (
        -540 <= west <= 540
        and -540 <= east <= 540
        and -90 <= south <= 90
        and -90 <= north <= 90
    ):
        msg = "The area must be in longitude/latitude degrees."
        raise SearchError(msg)
    if east - west >= 360:
        msg = "The area wraps all the way around the globe."
        raise SearchError(msg)
    # Map clicks can come from another world copy; an area may cross ±180°
    # (it then stays continuous, e.g. 175..185, see geo.py).
    return geo.normalize(geom)


def parse_day(value: str | None, end: bool = False) -> datetime | None:
    """Parse ``YYYY-MM-DD`` as UTC (the day's last second when ``end``)."""
    if not value:
        return None
    try:
        d = datetime.fromisoformat(value[:10])
    except ValueError as e:
        msg = f"Not a date: {value!r} (use YYYY-MM-DD)"
        raise SearchError(msg) from e
    if end:
        d = d.replace(hour=23, minute=59, second=59)
    return d.replace(tzinfo=timezone.utc)


def _iso(d: datetime | None) -> str | None:
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if d else None


def _umm_polygon(umm: dict) -> dict | None:
    try:
        geo = umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
    except KeyError:
        return None
    polys = []
    for gp in geo.get("GPolygons", []):
        pts = gp["Boundary"]["Points"]
        polys.append([[p["Longitude"], p["Latitude"]] for p in pts])
    if polys:
        return {"type": "MultiPolygon", "coordinates": [[ring] for ring in polys]}
    rects = geo.get("BoundingRectangles", [])
    if rects:
        r = rects[0]
        return mapping(
            box(
                r["WestBoundingCoordinate"],
                r["SouthBoundingCoordinate"],
                r["EastBoundingCoordinate"],
                r["NorthBoundingCoordinate"],
            )
        )
    return None


def _coverage(aoi: BaseGeometry, footprint: BaseGeometry) -> float:
    """Share of the area of interest inside ``footprint`` (0..1), across ±180° too."""
    return geo.coverage(aoi, geo.unwrap(footprint))


def _footprint(geometry: dict | BaseGeometry) -> dict:
    """GeoJSON for the map: one continuous polygon even across ±180°."""
    g = geometry if isinstance(geometry, BaseGeometry) else shape(geometry)
    return mapping(geo.unwrap(g))


def _fc(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def _parallel(fn: Callable, args: list, workers: int = 8) -> list:
    if not args:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(args))) as pool:
        return list(pool.map(fn, args))


# -- DISP-S1 -----------------------------------------------------------------
def disp_frames(aoi: BaseGeometry, direction: str = "both") -> list[dict]:
    """DISP-S1 frames whose footprint meets the area, with coverage of the area."""
    from opera_utils._types import Bbox
    from opera_utils.burst_frame_db import get_intersecting_frames

    seen: dict = {}
    for part in geo.wrap_parts(aoi):
        for f in get_intersecting_frames(Bbox(*part.bounds))["features"]:
            seen.setdefault(f["id"], f)
    out = []
    for f in seen.values():
        geom = shape(f["geometry"])
        cov = _coverage(aoi, geom)
        if cov <= 0:
            continue
        props = f.get("properties", {})
        d = str(props.get("orbit_pass", "")).lower()
        if direction in ("ascending", "descending") and d != direction:
            continue
        out.append(
            {
                "frame_id": int(f["id"]),
                "direction": d,
                "coverage": round(cov, 4),
                "geometry": _footprint(geom),
                "frame_km2": area_km2(geo.unwrap(geom)),
            }
        )
    return sorted(out, key=lambda r: (-r["coverage"], r["frame_id"]))


def search_disp(
    aoi: BaseGeometry, start: datetime | None, end: datetime | None, options: dict
) -> dict:
    """Find DISP-S1 frames over ``aoi`` and their products in the date range."""
    from opera_utils.disp._search import search

    frames = disp_frames(aoi, options.get("direction", "both"))
    version = options.get("version") or "1.0"
    aoi_km2 = area_km2(aoi)

    def one(fr: dict) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            products = search(
                frame_id=fr["frame_id"],
                product_version=version,
                start_datetime=start,
                end_datetime=end,
            )
        sizes = [p.size_in_bytes or 0 for p in products]
        frac = min(1.0, aoi_km2 * fr["coverage"] / max(fr["frame_km2"], 1e-9))
        return {
            **fr,
            "id": f"F{fr['frame_id']:05d}",
            "label": f"Frame {fr['frame_id']}",
            "n": len(products),
            "first": _iso(products[0].secondary_datetime) if products else None,
            "last": _iso(products[-1].secondary_datetime) if products else None,
            "n_reference_dates": len({p.reference_datetime for p in products}),
            "size_bytes": int(sum(sizes)),
            # Subsets keep a share of each file; metadata and compression make
            # this rough, so the UI labels it an estimate.
            "est_bytes": int(sum(sizes) * max(frac, 0.002)),
            "_products": products,
        }

    groups = _parallel(one, frames)
    items, features = [], []
    for g in groups:
        for p in g.pop("_products"):
            items.append(
                {
                    "group": g["id"],
                    "url": str(p.filename),
                    "date": _iso(p.secondary_datetime),
                    "reference_date": _iso(p.reference_datetime),
                    "size_bytes": p.size_in_bytes,
                }
            )
        features.append(
            {
                "type": "Feature",
                "id": g["frame_id"],
                "geometry": g.pop("geometry"),
                "properties": {
                    "id": g["id"],
                    "direction": g["direction"],
                    "label": g["label"],
                },
            }
        )
        g.pop("frame_km2", None)
    groups = [g for g in groups if g["n"] > 0] + [g for g in groups if g["n"] == 0]
    return {
        "groups": groups,
        "items": items,
        "footprints": _fc(features),
        "summary": {"n_groups": len(groups), "n_items": len(items)},
    }


# -- CSLC-S1 and CSLC-S1-STATIC (ASF) ----------------------------------------
def _asf_cslc(aoi: BaseGeometry, start, end, static: bool, direction: str) -> dict:
    from opera_utils.download import L2Product, search_l2

    product = L2Product.CSLC_STATIC if static else L2Product.CSLC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, names = [], set()
        for part in geo.wrap_parts(aoi):
            # check_missing_data=False: the plain ASFSearchResults, not a tuple.
            found = cast(
                "Any",
                search_l2(
                    start=None if static else start,
                    end=None if static else end,
                    aoi_polygon=part.wkt,
                    product=product,
                    max_results=20_000,
                ),
            )
            for r in found:
                if r.properties["fileName"] not in names:
                    names.add(r.properties["fileName"])
                    results.append(r)
    by_burst: dict[str, list] = defaultdict(list)
    for r in results:
        by_burst[r.properties["operaBurstID"]].append(r)
    # ASF's CSLC footprint is the whole burst grid, mostly nodata; the burst
    # database has the valid burst outline, which is what coverage should use.
    outlines = burst_outlines(list(by_burst))
    groups, items, features = [], [], []
    for burst, burst_results in sorted(by_burst.items()):
        d0 = str(burst_results[0].properties.get("flightDirection", "")).lower()
        if direction in ("ascending", "descending") and d0 != direction:
            continue
        rs = sorted(burst_results, key=lambda r: r.properties["startTime"])
        size = 0
        for r in rs:
            p = r.properties
            nbytes = (p.get("bytes") or {}).get(p["fileName"], {}).get("bytes")
            size += nbytes or 0
            items.append(
                {
                    "group": burst,
                    "url": p["url"],
                    "name": p["fileName"],
                    "date": p["startTime"],
                    "size_bytes": nbytes,
                }
            )
        geom = outlines.get(burst.lower()) or rs[0].geometry
        geom = _footprint(geom) if geom else None
        cov = _coverage(aoi, shape(geom)) if geom else 0
        groups.append(
            {
                "id": burst,
                "label": burst,
                "burst_id": burst,
                "direction": d0,
                "track": rs[0].properties.get("pathNumber"),
                "n": len(rs),
                "first": rs[0].properties["startTime"],
                "last": rs[-1].properties["startTime"],
                "size_bytes": size,
                "coverage": round(cov, 4),
            }
        )
        if geom:
            features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"id": burst, "direction": d0, "label": burst},
                }
            )
    groups.sort(key=lambda g: (-g["coverage"], g["id"]))
    return {
        "groups": groups,
        "items": items,
        "footprints": _fc(features),
        "summary": {"n_groups": len(groups), "n_items": len(items)},
    }


def burst_outlines(burst_ids: list[str]) -> dict[str, dict]:
    """Lower-case JPL burst id -> burst outline from the opera-utils burst database."""
    if not burst_ids:
        return {}
    try:
        from opera_utils.burst_frame_db import get_burst_id_geojson

        fc = get_burst_id_geojson([b.lower() for b in burst_ids])
    except Exception as e:
        logger.warning("burst outlines unavailable: %s", e)
        return {}
    return {f["properties"]["burst_id_jpl"]: f["geometry"] for f in fc["features"]}


def search_cslc(aoi, start, end, options: dict) -> dict:
    """Find CSLC-S1 bursts over ``aoi`` and their products in the date range."""
    return _asf_cslc(aoi, start, end, False, options.get("direction", "both"))


# -- static layers -------------------------------------------------------------
def search_static(aoi, start, end, options: dict) -> dict:  # noqa: ARG001
    """DISP-S1-STATIC per frame and/or CSLC-S1-STATIC per burst, in one result.

    ``options["kinds"]`` lists ``"disp"`` and/or ``"cslc"`` (both by default);
    each group carries its ``kind`` so a job can route it.
    """
    kinds = options.get("kinds") or (
        [options["kind"]] if options.get("kind") else ["disp", "cslc"]
    )
    direction = options.get("direction", "both")
    parts = []
    if "disp" in kinds:
        parts.append(_disp_static(aoi, direction))
    if "cslc" in kinds:
        part = _asf_cslc(aoi, None, None, True, direction)
        for g in part["groups"]:
            g["kind"] = "cslc"
            g["label"] = f"CSLC burst {g['id']}"
        for f in part["footprints"]["features"]:
            f["properties"]["kind"] = "cslc"
        parts.append(part)
    groups = [g for p in parts for g in p["groups"]]
    items = [i for p in parts for i in p["items"]]
    feats = [f for p in parts for f in p["footprints"]["features"]]
    return {
        "groups": groups,
        "items": items,
        "footprints": _fc(feats),
        "summary": {"n_groups": len(groups), "n_items": len(items)},
    }


def _disp_static(aoi, direction: str) -> dict:
    from opera_utils.disp._static import search_static as _search_static

    frames = disp_frames(aoi, direction)

    def one(fr: dict) -> dict:
        try:
            granules = _search_static(fr["frame_id"])
        except Exception as e:
            logger.warning("static search failed for %s: %s", fr["frame_id"], e)
            granules = []
        g = granules[0] if granules else None
        return {
            **fr,
            "id": f"F{fr['frame_id']:05d}",
            "label": f"DISP frame {fr['frame_id']}",
            "kind": "disp",
            "n": 1 if g else 0,
            "granule": g.granule if g else None,
            "layers": sorted(g.urls) if g else [],
            "_urls": g.urls if g else {},
        }

    groups = _parallel(one, frames)
    items, features = [], []
    for g in groups:
        for layer, url in g.pop("_urls").items():
            items.append({"group": g["id"], "url": url, "layer": layer})
        features.append(
            {
                "type": "Feature",
                "geometry": g.pop("geometry"),
                "properties": {
                    "id": g["id"],
                    "direction": g["direction"],
                    "label": g["label"],
                    "kind": "disp",
                },
            }
        )
        g.pop("frame_km2", None)
    return {"groups": groups, "items": items, "footprints": _fc(features)}


# -- NISAR -----------------------------------------------------------------------
def _nisar_footprints(short_name: str, bbox, start, end) -> dict[str, dict]:
    """GranuleUR -> footprint (a second, unverified CMR walk; geometry only)."""
    from opera_utils._cmr import fetch_cmr_pages

    params: dict[str, Any] = {
        "short_name": short_name,
        "provider": "ASF",
        "page_size": 500,
        "bounding_box": ",".join(map(str, bbox)),
    }
    if start or end:
        params["temporal"] = (
            f"{start.isoformat() if start else ''},{end.isoformat() if end else ''}"
        )
    out = {}
    try:
        for it in fetch_cmr_pages(CMR_GRANULES, params):
            umm = it["umm"]
            poly = _umm_polygon(umm)
            if poly:
                out[umm["GranuleUR"]] = poly
    except Exception as e:
        logger.warning("NISAR footprints: %s", e)
    return out


def search_nisar(aoi, start, end, options: dict) -> dict:
    """Find NISAR products (``options["kind"]``) over ``aoi`` in the date range."""
    from opera_utils.nisar import search, search_gunw

    kind = options.get("kind", "gslc")
    if kind not in NISAR_COLLECTIONS:
        msg = f"Unknown NISAR product {kind!r}"
        raise SearchError(msg)
    direction = options.get("direction", "both")
    od = {"ascending": "A", "descending": "D"}.get(direction)
    short_name = NISAR_COLLECTIONS[kind]
    products: list[Any] = []
    feet: dict = {}
    names: set[str] = set()
    for part in geo.wrap_parts(aoi):
        bbox = part.bounds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            found: list[Any]
            if kind == "gunw":
                found = search_gunw(
                    bbox=bbox,
                    start_datetime=start,
                    end_datetime=end,
                    orbit_direction=od,
                )
            else:
                found = search(
                    bbox=bbox,
                    start_datetime=start,
                    end_datetime=end,
                    orbit_direction=od,
                    short_name=short_name,
                )
        for p in found:
            if str(p.filename) not in names:
                names.add(str(p.filename))
                products.append(p)
        feet.update(_nisar_footprints(short_name, bbox, start, end))
    by_group: dict[str, list] = defaultdict(list)
    for p in products:
        key = (
            f"{p.relative_orbit_number:03d}_{str(p.orbit_direction)[:1]}_"
            f"{p.track_frame_number:03d}"
        )
        by_group[key].append(p)
    groups, items, features = [], [], []
    for key, ps in sorted(by_group.items()):
        rows = []
        for p in ps:
            name = str(p.filename).rsplit("/", 1)[-1]
            when = p.secondary_datetime if kind == "gunw" else p.start_datetime
            row = {
                "group": key,
                "url": str(p.filename),
                "name": name,
                "date": _iso(when),
                "size_bytes": p.size_in_bytes,
                "polarizations": p.polarizations,
                "bandwidth": p.bandwidth,
            }
            if kind == "gunw":
                row["reference_date"] = _iso(p.reference_datetime)
            rows.append(row)
        rows.sort(key=lambda r: r["date"] or "")
        items += rows
        d = (
            "ascending"
            if str(ps[0].orbit_direction).upper().startswith("A")
            else "descending"
        )
        geom = next(
            (
                feet[r["name"].removesuffix(".h5")]
                for r in rows
                if r["name"].removesuffix(".h5") in feet
            ),
            None,
        )
        geom = _footprint(geom) if geom else None
        cov = _coverage(aoi, shape(geom)) if geom else None
        groups.append(
            {
                "id": key,
                "label": (
                    f"Track {ps[0].relative_orbit_number} · frame"
                    f" {ps[0].track_frame_number}"
                ),
                "direction": d,
                "relative_orbit": ps[0].relative_orbit_number,
                "track_frame": ps[0].track_frame_number,
                "n": len(rows),
                "first": rows[0]["date"],
                "last": rows[-1]["date"],
                "size_bytes": int(sum(r["size_bytes"] or 0 for r in rows)),
                "coverage": round(cov, 4) if cov is not None else None,
                "polarizations": sorted({r["polarizations"] for r in rows}),
            }
        )
        if geom:
            features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"id": key, "direction": d, "label": key},
                }
            )
    return {
        "groups": groups,
        "items": items,
        "footprints": _fc(features),
        "summary": {"n_groups": len(groups), "n_items": len(items)},
    }


# -- TROPO -------------------------------------------------------------------------
def _tropo_time(granule_ur: str) -> datetime:
    # OPERA_L4_TROPO-ZENITH_<valid time>_<production time>_HRES_v1.0
    stamp = granule_ur.split("_")[3]
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def search_tropo(aoi, start, end, options: dict) -> dict:
    """List the global TROPO products in the date range, cropped to ``aoi``."""
    from opera_utils._cmr import fetch_cmr_pages, get_download_url

    if start is None or end is None:
        msg = "TROPO is global and 6-hourly: pick a start and an end date."
        raise SearchError(msg)
    if geo.crosses(aoi):
        msg = (
            "TROPO cropping cannot cross the antimeridian yet: split the area at 180°."
        )
        raise SearchError(msg)
    if (end - start).days > 3 * 366:
        msg = "Pick at most three years of TROPO at a time."
        raise SearchError(msg)
    params: dict[str, Any] = {
        "short_name": TROPO_COLLECTION,
        "page_size": 2000,
        "temporal": f"{start.isoformat()},{end.isoformat()}",
    }
    latest: dict[datetime, dict] = {}
    for it in fetch_cmr_pages(CMR_GRANULES, params):
        umm = it["umm"]
        try:
            t = _tropo_time(umm["GranuleUR"])
            url = get_download_url(umm, filename_suffix=".nc")
        except (ValueError, IndexError):
            continue
        if not (start <= t <= end):  # CMR also returns cubes whose range overlaps
            continue
        prev = latest.get(t)
        # Several productions of one valid time: keep the newest.
        if prev is None or umm["GranuleUR"] > prev["name"]:
            archive = umm.get("DataGranule", {}).get(
                "ArchiveAndDistributionInformation", []
            )
            latest[t] = {
                "url": url,
                "name": umm["GranuleUR"],
                "date": _iso(t),
                "size_bytes": archive[0].get("SizeInBytes") if archive else None,
            }
    hours = options.get("hours")  # e.g. [0, 12]; None keeps all four
    rows = [
        latest[t]
        for t in sorted(latest)
        if not hours or t.hour in {int(h) for h in hours}
    ]
    by_day: dict[str, list] = defaultdict(list)
    for r in rows:
        r["group"] = r["date"][:10]
        by_day[r["group"]].append(r)
    groups = [
        {
            "id": day,
            "label": day,
            "n": len(rs),
            "first": rs[0]["date"],
            "last": rs[-1]["date"],
            "size_bytes": int(sum(r["size_bytes"] or 0 for r in rs)),
            "hours": [int(r["date"][11:13]) for r in rs],
        }
        for day, rs in by_day.items()
    ]
    footprint = {
        "type": "Feature",
        "geometry": mapping(box(*aoi.buffer(0.3).bounds)),
        "properties": {
            "id": "tropo-crop",
            "label": "Crop (area + 0.3°)",
            "direction": "global",
        },
    }
    return {
        "groups": groups,
        "items": rows,
        "footprints": _fc([footprint]),
        "summary": {"n_groups": len(groups), "n_items": len(rows)},
    }


SEARCHES: dict[str, Callable] = {
    "disp": search_disp,
    "cslc": search_cslc,
    "nisar": search_nisar,
    "tropo": search_tropo,
    "static": search_static,
}


def run_search(request: dict) -> dict:
    """Run one product search from a UI request (see module docstring)."""
    product = request.get("product")
    if product not in SEARCHES:
        msg = f"Unknown product {product!r}; choose from {', '.join(PRODUCTS)}"
        raise SearchError(msg)
    aoi = parse_aoi(request.get("aoi"))
    start = parse_day(request.get("start"))
    end = parse_day(request.get("end"), end=True)
    if start and end and start > end:
        msg = "The start date is after the end date."
        raise SearchError(msg)
    options = request.get("options") or {}
    result = SEARCHES[product](aoi, start, end, options)
    result["product"] = product
    result["kind"] = options.get("kind")
    result["aoi_km2"] = round(area_km2(aoi), 2)
    return result
