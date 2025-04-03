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
