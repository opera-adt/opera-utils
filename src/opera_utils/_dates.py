from __future__ import annotations

import datetime
import itertools
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, overload

from ._types import DateOrDatetime, Filename, PathLikeT

__all__ = [
    "get_dates",
    "filter_by_date",
    "group_by_date",
    "DATE_FORMAT",
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
    date_list = re.findall(pattern, path.stem)
    if not date_list:
        return []
    return [_parse_date(d, fmt) for d in date_list]


@overload
def filter_by_date(
    files: Iterable[PathLikeT],
    dates: Iterable[DateOrDatetime],
    fmt: str = DATE_FORMAT,
) -> list[PathLikeT]:
    ...


@overload
def filter_by_date(
    files: Iterable[str],
    dates: Iterable[DateOrDatetime],
    fmt: str = DATE_FORMAT,
) -> list[str]:
    ...


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


def group_by_date(
    files: Iterable[PathLikeT], file_date_fmt: str = DATE_FORMAT
) -> dict[tuple[datetime.datetime, ...], list[PathLikeT]]:
    """Combine files by date into a dict.

    Parameters
    ----------
    files: Iterable[Filename]
        Path to folder containing files with dates in the filename.
    file_date_fmt: str
        Format of the date in the filename.
        Default is [dolphin.DEFAULT_DATETIME_FORMAT][]

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
        grouped_images[dates].extend(list(g))
    return grouped_images


def _parse_date(datestr: str, fmt: str = DATE_FORMAT) -> datetime.datetime:
    return datetime.datetime.strptime(datestr, fmt)


def _get_path_from_gdal_str(name: Filename) -> Path:
    s = str(name)
    if s.upper().startswith("DERIVED_SUBDATASET"):
        # like DERIVED_SUBDATASET:AMPLITUDE:slc_filepath.tif
        p = s.split(":")[-1].strip('"').strip("'")
    elif ":" in s and (s.upper().startswith("NETCDF") or s.upper().startswith("HDF")):
        # like NETCDF:"slc_filepath.nc":subdataset
        p = s.split(":")[1].strip('"').strip("'")
    else:
        # Whole thing is the path
        p = str(name)
    return Path(p)


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
    date_format = date_format.replace("%m", r"\d{2}")
    date_format = date_format.replace("%d", r"\d{2}")
    date_format = date_format.replace("%H", r"\d{2}")
    date_format = date_format.replace("%M", r"\d{2}")
    date_format = date_format.replace("%S", r"\d{2}")

    # Return the resulting regular expression
    return re.compile(date_format)
