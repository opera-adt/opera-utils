from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from re import Pattern
from typing import overload

from ._types import Filename, PathLikeT
from .constants import OPERA_BURST_RE

logger = logging.getLogger(__name__)

__all__ = [
    "filter_by_burst_id",
    "get_burst_id",
    "group_by_burst",
    "normalize_burst_id",
    "sort_by_burst_id",
]


def normalize_burst_id(burst_id_str: str) -> str:
    """Normalize the OPERA S1 burst id to lowercase/underscores."""
    return burst_id_str.lower().replace("-", "_")


def get_burst_id(
    filename: Filename, burst_id_fmt: str | Pattern[str] = OPERA_BURST_RE
) -> str:
    """Extract the burst id from a filename.

    Matches either format of
        t087_185684_iw2 (which comes from COMPASS)
        T087-165495-IW3 (which is the official product naming scheme)

    Parameters
    ----------
    filename: Filename
        CSLC filename
    burst_id_fmt: str
        format of the burst id in the filename.
        Default is [`OPERA_BURST_RE`][opera_utils.OPERA_BURST_RE]

    Returns
    -------
    str
        burst id of the SLC acquisition, normalized to be in the format
            t087_185684_iw2

    """
    if not (m := re.search(burst_id_fmt, str(filename))):
        msg = f"Could not parse burst id from {filename}"
        raise ValueError(msg)
    return normalize_burst_id(m.group())


@overload
def group_by_burst(
    file_list: Iterable[str],
    burst_id_fmt: str | Pattern[str] = OPERA_BURST_RE,
) -> dict[str, list[str]]: ...


@overload
def group_by_burst(
    file_list: Iterable[PathLikeT],
    burst_id_fmt: str | Pattern[str] = OPERA_BURST_RE,
) -> dict[str, list[PathLikeT]]: ...


def group_by_burst(file_list, burst_id_fmt=OPERA_BURST_RE):
    """Group Sentinel CSLC files by burst.

    Parameters
    ----------
    file_list: Iterable[Filename]
        list of paths of CSLC files
    burst_id_fmt: str
        format of the burst id in the filename.
        Default is [`OPERA_BURST_RE`][opera_utils.OPERA_BURST_RE]

    Returns
    -------
    dict
        key is the burst id of the SLC acquisition
        Value is a list of inputs which correspond to that burst:
        {
            't087_185678_iw2': ['inputs/t087_185678_iw2_20200101.h5',...,],
            't087_185678_iw3': ['inputs/t087_185678_iw3_20200101.h5',...,],
        }

    """
    if not file_list:
        return {}

    sorted_file_list = sort_by_burst_id(list(file_list), burst_id_fmt)
    # Now collapse into groups, sorted by the burst_id
    grouped_images = {
        burst_id: list(g)
        for burst_id, g in itertools.groupby(
            sorted_file_list, key=lambda x: get_burst_id(x)
        )
    }
    return grouped_images


@overload
def sort_by_burst_id(file_list: Iterable[str], burst_id_fmt) -> list[str]: ...


@overload
def sort_by_burst_id(
    file_list: Iterable[PathLikeT], burst_id_fmt
) -> list[PathLikeT]: ...


def sort_by_burst_id(file_list, burst_id_fmt):
    """Sort files/paths by the burst ID in their names.

    Parameters
    ----------
    file_list : Iterable[PathLikeT]
        list of paths of CSLC files
    burst_id_fmt : str
        format of the burst id in the filename.
        Default is [`OPERA_BURST_RE`][opera_utils.OPERA_BURST_RE]

    Returns
    -------
    list[Path] or list[str]
        sorted list of files

    """
    file_burst_tuples = sorted(
        [(f, get_burst_id(f, burst_id_fmt)) for f in file_list],
        # use the date or dates as the key
        key=lambda f_b_tuple: f_b_tuple[1],
    )
    # Unpack the sorted pairs with new sorted values
    out_file_list = [f for f, _ in file_burst_tuples]
    return out_file_list


@overload
def filter_by_burst_id(
    files: Iterable[PathLikeT],
    burst_ids: str | Iterable[str],
) -> list[PathLikeT]: ...


@overload
def filter_by_burst_id(
    files: Iterable[str],
    burst_ids: str | Iterable[str],
) -> list[str]: ...


def filter_by_burst_id(files, burst_ids):
    """Keep only items from `files` which contain a burst ID in `burst_ids`.

    Searches only the burst ID in the base name, not the full path.

    Parameters
    ----------
    files : Iterable[PathLikeT] or Iterable[str]
        Iterable of files to filter
    burst_ids : str | Iterable[str]
        Burst ID/Iterable containing the of burst IDs to keep

    Returns
    -------
    list[PathLikeT] or list[str]
        filtered list of files

    """
    if isinstance(burst_ids, str):
        burst_ids = [burst_ids]

    burst_id_set = set(burst_ids)
    parsed_burst_ids = [get_burst_id(Path(f).name) for f in files]
    # Only search the burst ID in the name, not the full path
    return [f for (f, b) in zip(files, parsed_burst_ids) if b in burst_id_set]
