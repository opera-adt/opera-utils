from __future__ import annotations

import datetime
import itertools
import re
from pathlib import Path
from typing import Iterable, overload

from ._types import Filename, PathLikeT

__all__ = [
    "get_dates",
    "filter_by_date",
    "DATE_FORMAT",
]

DATE_FORMAT = "%Y%m%d"
DATETIME_FORMAT = "%Y%m%dT%H%M%S"

__all__ = [
    "group_by_date",
]


def get_dates(filename: Filename, fmt: str = DATE_FORMAT) -> list[datetime.date]:
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
    list[datetime.date]
        list of dates found in the stem of `filename` matching `fmt`.

    Examples
    --------
    >>> get_dates("/path/to/20191231.slc.tif")
    [datetime.date(2019, 12, 31)]
    >>> get_dates("S1A_IW_SLC__1SDV_20191231T000000_20191231T000000_032123_03B8F1_1C1D.nc")
    [datetime.date(2019, 12, 31), datetime.date(2019, 12, 31)]
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
    dates: Iterable[datetime.date],
    fmt: str = DATE_FORMAT,
) -> list[PathLikeT]:
    ...


@overload
def filter_by_date(
    files: Iterable[str],
    dates: Iterable[datetime.date],
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
    """
    date_set = set(dates)
    out = []
    for f in list(files):
        date_tuple = get_dates(f, fmt)
        if any(d in date_set for d in date_tuple):
            out.append(f)
    return out


def _parse_date(datestr: str, fmt: str = DATE_FORMAT) -> datetime.date:
    """Parse a date string into a datetime.date object.

    Parameters
    ----------
    datestr : str
        Date string to be parsed.
    fmt : str, optional
        Format of the date string. Default is %Y%m%d.

    Returns
    -------
    datetime.date
        Parsed date object.
    """
    return datetime.datetime.strptime(datestr, fmt).date()


def _get_path_from_gdal_str(name: Filename) -> Path:
    """Extract a Path object from a GDAL-formatted string.

    Parameters
    ----------
    name : Filename
        GDAL-formatted string.

    Returns
    -------
    Path
        Extracted Path object.
    """
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

    Useful for Year, month, date date formats.

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

    # Return the resulting regular expression
    return re.compile(date_format)


def group_by_date(
    file_list: Iterable[Filename], file_date_fmt: str = "%Y%m%d"
) -> dict[datetime.date, list[Filename]]:
    """Combine files by date into a dict.

    Parameters
    ----------
    file_list: Iterable[Filename]
        Path to folder containing files with dates in the filename.
    file_date_fmt: str
        Format of the date in the filename.
        Default is [dolphin.io.DEFAULT_DATETIME_FORMAT][]

    Returns
    -------
    dict
        key is a list of dates in the filenames.
        Value is a list of Paths on that date.
        E.g.:
        {(datetime.date(2017, 10, 13),
          [Path(...)
            Path(...),
            ...]),
         (datetime.date(2017, 10, 25),
          [Path(...)
            Path(...),
            ...]),
        }
    """
    sorted_file_list, _ = sort_files_by_date(file_list, file_date_fmt=file_date_fmt)

    # Now collapse into groups, sorted by the date
    grouped_images = {
        dates: list(g)
        for dates, g in itertools.groupby(
            sorted_file_list, key=lambda x: get_dates(x)[0]
        )
    }
    return grouped_images


def sort_files_by_date(
    files: Iterable[Filename], file_date_fmt: str = "%Y%m%d"
) -> tuple[list[Filename], list[list[datetime.date]]]:
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
    dates : list[list[datetime.date,...]]
        Sorted list, where each entry has all the dates from the corresponding file.
    """

    def sort_key(file_date_tuple):
        # Key for sorting:
        # To sort the files with the most dates first (the compressed SLCs which
        # span a date range), sort the longer date lists first.
        # Then, within each group of dates of the same length, use the date/dates
        _, dates = file_date_tuple
        try:
            return (-len(dates), dates)
        except TypeError:
            return (-1, dates)

    file_date_tuples = [(f, get_dates(f, fmt=file_date_fmt)) for f in files]
    file_dates = sorted([fd_tuple for fd_tuple in file_date_tuples], key=sort_key)

    # Unpack the sorted pairs with new sorted values
    file_list, dates = zip(*file_dates)  # type: ignore
    return list(file_list), list(dates)
