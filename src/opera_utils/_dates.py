from __future__ import annotations

import datetime
import itertools
import operator
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import overload

from ._types import DateOrDatetime, Filename, PathLikeT
from ._utils import _get_path_from_gdal_str

__all__ = [
    "DATETIME_FORMAT",
    "DATE_FORMAT",
    "filter_by_date",
    "get_dates",
    "group_by_date",
    "sort_files_by_date",
]

DATE_FORMAT = "%Y%m%d"
DATETIME_FORMAT = "%Y%m%dT%H%M%S"


def get_dates(filename: Filename, fmt: str = DATE_FORMAT) -> list[datetime.datetime]:
    """Search for dates in the stem of `filename` matching `fmt`.

    Excludes dates that are not in the stem of `filename` (in the directories).

    Parameters
    ----------
    filename : str or PathLike
        Filename to search for dates.
    fmt : str, optional
        Format of date to search for. Default is %Y%m%d

    Returns
    -------
    list[datetime.datetime]
        list of dates found in the stem of `filename` matching `fmt`.

    Examples
    --------
    >>> get_dates("/path/to/20191231.slc.tif")
    [datetime.datetime(2019, 12, 31, 0, 0)]
    >>> get_dates("S1A_IW_SLC__1SDV_20191231T000000_20191231T000000_032123_03B8F1_1C1D.nc")
    [datetime.datetime(2019, 12, 31, 0, 0), datetime.datetime(2019, 12, 31, 0, 0)]
    >>> get_dates("/not/a/date_named_file.tif")
    []

    """  # noqa: E501
    path = _get_path_from_gdal_str(filename)
    pattern = _date_format_to_regex(fmt)
    date_list = re.findall(pattern, path.name)
    if not date_list:
        return []
    return [_parse_date(d, fmt) for d in date_list]


@overload
def filter_by_date(
    files: Iterable[PathLikeT],
    dates: Iterable[DateOrDatetime],
    fmt: str = DATE_FORMAT,
) -> list[PathLikeT]: ...


@overload
def filter_by_date(
    files: Iterable[str],
    dates: Iterable[DateOrDatetime],
    fmt: str = DATE_FORMAT,
) -> list[str]: ...


def filter_by_date(files, dates, fmt=DATE_FORMAT):
    """Keep only items in `files` which have a date in `dates`.

    Parameters
    ----------
    files : Iterable[PathLikeT] or Iterable[str]
        Iterable of files to filter
    dates : Iterable[datetime.date]
        Iterable of dates to filter by
    fmt : str, optional
        Format of date to search for. Default is %Y%m%d

    Returns
    -------
    list[PathLikeT]
        Items in `files`

    """
    date_set = set(dates)
    out = []
    for f in list(files):
        date_tuple = get_dates(f, fmt)
        if any(d in date_set for d in date_tuple):
            out.append(f)
    return out


def _parse_date(datestr: str, fmt: str = DATE_FORMAT) -> datetime.datetime:
    return datetime.datetime.strptime(datestr, fmt)


def _date_format_to_regex(date_format: str) -> re.Pattern:
    r"""Convert a python date format string to a regular expression.

    Parameters
    ----------
    date_format : str
        Date format string, e.g. DATE_FORMAT

    Returns
    -------
    re.Pattern
        Regular expression that matches the date format string.

    Examples
    --------
    >>> pat2 = _date_format_to_regex("%Y%m%d").pattern
    >>> pat2 == re.compile(r'\d{4}\d{2}\d{2}').pattern
    True
    >>> pat = _date_format_to_regex("%Y-%m-%d").pattern
    >>> pat == re.compile(r'\d{4}\-\d{2}\-\d{2}').pattern
    True

    """
    # Escape any special characters in the date format string
    date_format = re.escape(date_format)

    # Replace each format specifier with a regular expression that matches it
    date_format = date_format.replace("%Y", r"\d{4}")
    date_format = date_format.replace("%y", r"\d{2}")
    date_format = date_format.replace("%m", r"\d{2}")
    date_format = date_format.replace("%d", r"\d{2}")
    date_format = date_format.replace("%H", r"\d{2}")
    date_format = date_format.replace("%M", r"\d{2}")
    date_format = date_format.replace("%S", r"\d{2}")
    date_format = date_format.replace("%j", r"\d{3}")

    # Return the resulting regular expression
    return re.compile(date_format)


def group_by_date(
    files: Iterable[PathLikeT],
    file_date_fmt: str = DATE_FORMAT,
    date_idx: int | None = None,
) -> dict[tuple[datetime.datetime, ...], list[PathLikeT]]:
    """Combine files by date into a dict.

    Parameters
    ----------
    files : Iterable[Filename]
        Path to folder containing files with dates in the filename.
    file_date_fmt : str
        Format of the date in the filename.
        Default is [dolphin.DEFAULT_DATETIME_FORMAT][]
    date_idx : int, optional
        If provided, uses only this index of the dates found in each filename.
        For example, if `file_date_fmt='%Y%m%d'`, and the files have pairs of
        these date strings but you only wish to group by the first, use
        `date_idx=0`.

    Returns
    -------
    dict
        key is a list of dates in the filenames.
        Value is a list of Paths on that date.
        E.g.:
        {(datetime.datetime(2017, 10, 13),
          [Path(...)
            Path(...),
            ...]),
         (datetime.datetime(2017, 10, 25),
          [Path(...)
            Path(...),
            ...]),
        }

    """
    # collapse into groups of dates
    # Use a `defaultdict` so we dont have to sort the files by date in advance,
    # but rather just extend the list each time there's a new group
    grouped_images: dict[tuple[datetime.datetime, ...], list[PathLikeT]] = defaultdict(
        list
    )

    for dates, g in itertools.groupby(
        files, key=lambda x: tuple(get_dates(x, fmt=file_date_fmt))
    ):
        key = dates if date_idx is None else (dates[date_idx],)
        grouped_images[key].extend(list(g))
    return grouped_images


def sort_files_by_date(
    files: Iterable[Filename], file_date_fmt: str = "%Y%m%d"
) -> tuple[list[Filename], list[list[datetime.date | datetime.datetime]]]:
    """Sort a list of files by date.

    If some files have multiple dates, the files with the most dates are sorted
    first. Within each group of files with the same number of dates, the files
    with the earliest dates are sorted first.

    The multi-date files are placed first so that compressed SLCs are sorted
    before the individual SLCs that make them up.

    Parameters
    ----------
    files : Iterable[Filename]
        list of files to sort.
    file_date_fmt : str, optional
        Datetime format passed to `strptime`, by default "%Y%m%d"

    Returns
    -------
    file_list : list[Filename]
        list of files sorted by date.
    dates : list[list[datetime.date | datetime.datetime]]
        Sorted list, where each entry has all the dates from the corresponding file.

    """
    file_date_tuples = [(f, get_dates(f, fmt=file_date_fmt)) for f in files]
    # Get the second item in the tuple for the key
    file_dates = sorted(file_date_tuples, key=operator.itemgetter(1))

    # Unpack the sorted pairs with new sorted values
    file_list, dates = zip(*file_dates)
    return list(file_list), list(dates)
