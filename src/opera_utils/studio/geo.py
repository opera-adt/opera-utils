"""Longitude handling for areas and outlines near the antimeridian.

The burst, frame and NISAR databases store a polygon that crosses ±180° as a
MultiPolygon cut at the line (one part near +180, one near -180). On a map
that shows as a seam, or as a band across the whole world. Searches,
on the other hand, want every part inside [-180, 180].

* ``unwrap``: one continuous polygon (e.g. 175..185) for drawing and for
  computing crop windows (pyproj accepts longitudes past 180);
* ``wrap_parts``: the same area as parts inside [-180, 180], for databases,
  ASF and CMR;
* ``normalize``: move a drawn area (map clicks can be in another world copy)
  so its centre lies in [-180, 180);
* ``intersect_lon``: intersection of two geometries that may be unwrapped
  differently (one shifted by 360°).
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

WORLD = box(-180, -90, 180, 90)


def _shift(g: BaseGeometry, dx: float) -> BaseGeometry:
    return affinity.translate(g, xoff=dx) if dx else g


def crosses(g: BaseGeometry) -> bool:
    """Return True if the geometry spans the antimeridian (cut, jumping, unwrapped)."""
    minx, _, maxx, _ = g.bounds
    return maxx - minx > 180 or minx < -180 or maxx > 180


def unwrap(g: BaseGeometry) -> BaseGeometry:
    """Continuous longitudes for a geometry cut at, or jumping across, ±180°."""
    g = shapely.force_2d(g)
    minx, _, maxx, _ = g.bounds
    if maxx - minx <= 180:
        return g
    parts = list(getattr(g, "geoms", [g]))
    out = []
    for part in parts:
        p = part
        pminx, _, pmaxx, _ = p.bounds
        if pmaxx - pminx > 180:  # a ring whose vertices jump from +179 to -179
            p = shapely.transform(
                p,
                lambda c: np.column_stack(
                    [np.where(c[:, 0] < 0, c[:, 0] + 360, c[:, 0]), c[:, 1]]
                ),
            )
        elif pmaxx <= 0:  # the western part of a cut polygon
            p = _shift(p, 360)
        out.append(p)
    merged = unary_union(out)
    # The databases cut loosely: the two halves rarely meet vertex for vertex, and
    # one may stop short of the line (e.g. at 179.985°). The union then keeps a
    # zero-width slit or a thin wedge down the 180° line, drawn as a seam. A
    # mitred closing of 0.06° (~6 km) fills notches narrower than ~12 km and
    # keeps the convex frame corners where they are; it is only applied when
    # the area changes by less than 2%.
    eps = 0.06
    fused = merged.buffer(eps, join_style="mitre").buffer(-eps, join_style="mitre")
    fused = shapely.simplify(fused, 1e-7)
    if not fused.is_empty and abs(fused.area - merged.area) <= 0.02 * max(
        merged.area, 1e-12
    ):
        merged = fused
    return merged if merged.is_valid else merged.buffer(0)


def normalize(g: BaseGeometry) -> BaseGeometry:
    """Shift by whole turns so the centre lies in [-180, 180)."""
    k = round(g.centroid.x / 360)
    return _shift(g, -360 * k) if k else g


def wrap_parts(g: BaseGeometry) -> list[BaseGeometry]:
    """Split the geometry into polygons inside [-180, 180] (crossings give 2+)."""
    pieces: list[BaseGeometry] = []
    for dx in (0, 360, -360):
        p = _shift(g, dx).intersection(WORLD)
        if not p.is_empty and p.area > 0:
            pieces.extend(getattr(p, "geoms", [p]))
    return [p for p in pieces if p.geom_type == "Polygon"]


def intersect_lon(a: BaseGeometry, b: BaseGeometry) -> BaseGeometry:
    """Return ``a ∩ b``, retrying ``b`` shifted ±360° if unwrapped differently."""
    best = a.intersection(b)
    for dx in (360, -360):
        if not best.is_empty and best.area > 0:
            break
        best = a.intersection(_shift(b, dx))
    return best


def coverage(aoi: BaseGeometry, footprint: BaseGeometry) -> float:
    """Share of ``aoi`` inside ``footprint`` (0..1), across the antimeridian too."""
    if aoi.area == 0:
        return 1.0 if not intersect_lon(aoi, footprint).is_empty else 0.0
    return float(min(1.0, intersect_lon(aoi, footprint).area / aoi.area))
