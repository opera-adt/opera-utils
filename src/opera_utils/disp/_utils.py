from __future__ import annotations

import datetime
import itertools
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from opera_utils._dates import get_dates

PathOrStrT = TypeVar("PathOrStrT", Path, str)


def get_frame_coordinates(frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Get the UTM x, y coordinates for a frame.

    Parameters
    ----------
    frame_id : int
        The frame ID to get the coordinates for.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (x, y) arrays of UTM coordinates (in meters).
    """
    from opera_utils.burst_frame_db import get_frame_bbox

    _epsg, bbox = get_frame_bbox(frame_id)
    # 30 meter spacing, coords are on the pixel centers.
    x = np.arange(bbox[0], bbox[2], 30)
    # y-coordinates are in decreasing order (north first)
    y = np.arange(bbox[3], bbox[1], -30)
    # Now shift to pixel centers:
    return x + 15, y - 15


def flatten(list_of_lists: Iterable[Iterable[Any]]) -> itertools.chain[Any]:
    """Flatten one level of a nested iterable."""
    return itertools.chain.from_iterable(list_of_lists)


def _last_per_ministack(
    opera_file_list: Sequence[PathOrStrT],
) -> list[PathOrStrT]:
    def _get_generation_time(fname: PathOrStrT) -> datetime.datetime:
        return get_dates(fname)[2]

    last_per_ministack = []
    for d, cur_groupby in itertools.groupby(
        sorted(opera_file_list), key=_get_generation_time
    ):
        # cur_groupby is an iterable of all matching
        # Get the first one, and the last one. ignore the rest
        last_file = list(cur_groupby)[-1]
        last_per_ministack.append(last_file)
    return last_per_ministack


def utm_to_rowcol(
    utm_x: float,
    utm_y: float,
    geotransform: tuple[float, float, float, float, float, float],
) -> tuple[int, int]:
    """Convert UTM coordinates to pixel row and column indices.

    Parameters
    ----------
    utm_x : float
        The UTM x coordinate.
    utm_y : float
        The UTM y coordinate.
    geotransform : tuple
        Geotransform tuple of the form
        (xmin, pixel_width, 0, ymax, 0, pixel_height),
        where pixel_height is negative for top-down images.

    Returns
    -------
    tuple[int, int]
        (row, col) indices corresponding to the UTM coordinate.
    """
    xmin, pixel_width, _, ymax, _, pixel_height = geotransform
    col = int(round((utm_x - xmin) / pixel_width))
    row = int(round((ymax - utm_y) / abs(pixel_height)))
    return row, col


def round_mantissa(z: np.ndarray, keep_bits=10) -> None:
    """Zero out mantissa bits of elements of array in place.

    Drops a specified number of bits from the floating point mantissa,
    leaving an array more amenable to compression.

    Parameters
    ----------
    z : numpy.ndarray
        Real or complex array whose mantissas are to be zeroed out
    keep_bits : int, optional
        Number of bits to preserve in mantissa. Defaults to 10.
        Lower numbers will truncate the mantissa more and enable
        more compression.

    References
    ----------
    https://numcodecs.readthedocs.io/en/v0.12.1/_modules/numcodecs/bitround.html

    """
    max_bits = {
        "float16": 10,
        "float32": 23,
        "float64": 52,
    }
    # recurse for complex data
    if np.iscomplexobj(z):
        round_mantissa(z.real, keep_bits)
        round_mantissa(z.imag, keep_bits)
        return

    if not z.dtype.kind == "f" or z.dtype.itemsize > 8:
        raise TypeError("Only float arrays (16-64bit) can be bit-rounded")

    bits = max_bits[str(z.dtype)]
    # cast float to int type of same width (preserve endianness)
    a_int_dtype = np.dtype(z.dtype.str.replace("f", "i"))
    all_set = np.array(-1, dtype=a_int_dtype)
    if keep_bits == bits:
        return z
    if keep_bits > bits:
        raise ValueError("keep_bits too large for given dtype")
    b = z.view(a_int_dtype)
    maskbits = bits - keep_bits
    mask = (all_set >> maskbits) << maskbits
    half_quantum1 = (1 << (maskbits - 1)) - 1
    b += ((b >> maskbits) & 1) + half_quantum1
    b &= mask
    return b.view(z.dtype)
