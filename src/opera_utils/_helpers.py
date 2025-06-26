from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import chain, combinations
from typing import Any

from pyproj import Transformer

from ._types import Bbox

try:
    from rasterio.warp import transform_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


__all__ = ["reproject_bounds"]


def flatten(list_of_lists: Iterable[Iterable[Any]]) -> chain[Any]:
    """Flatten one level of nesting."""
    return chain.from_iterable(list_of_lists)


def sorted_deduped_values(in_mapping: Mapping[Any, list]):
    """Sort, dedupe, and concatenate all items in the lists of `in_mapping`'s values."""
    all_values = flatten(in_mapping.values())
    return sorted(set(all_values))


def powerset(iterable: Iterable[Any]) -> chain[tuple[Any, ...]]:
    """Generate the powerset of an iterable.

    Examples
    --------
    >>> list(powerset([1,2,3]))
    [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

    """
    s = list(iterable)
    return flatten(combinations(s, r) for r in range(len(s) + 1))


def reproject_bounds(bounds: Bbox, src_epsg: int, dst_epsg: int) -> Bbox:
    """Reproject the (left, bottom, right top) from `src_epsg to `dst_epsg`."""
    if not HAS_RASTERIO:
        msg = "rasterio must be installed to use this function"
        raise ImportError(msg)

    left, bottom, right, top = transform_bounds(src_epsg, dst_epsg, *bounds)
    return Bbox(left, bottom, right, top)


def reproject_coordinates(
    x: Sequence[float], y: Sequence[float], src_epsg: int, dst_epsg: int
) -> tuple[list[float], list[float]]:
    """Reproject the coordinates from `src_epsg` to `dst_epsg`."""
    t = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    return t.transform(x, y)
