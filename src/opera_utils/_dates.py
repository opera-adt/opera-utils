from __future__ import annotations

import datetime
import re
from pathlib import Path

from ._types import Filename

__all__ = [
    "get_dates",
]


def get_dates(filename: Filename, fmt: str = "%Y%m%d") -> list[datetime.date]:
    """Search for dates in the stem of `filename` matching `fmt`.

    Excludes dates that are not in the stem of `filename` (in the directories).

    Parameters
    ----------
    filename : str or PathLike
        Filename to search for dates.
    fmt : str, optional
        Format of date to search for. Default is "%Y%m%d".

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


def _parse_date(datestr: str, fmt: str = "%Y%m%d") -> datetime.date:
    return datetime.datetime.strptime(datestr, fmt).date()


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


def _date_format_to_regex(date_format):
    r"""Convert a python date format string to a regular expression.

    Useful for Year, month, date date formats.

    Parameters
    ----------
    date_format : str
        Date format string, e.g. "%Y%m%d"

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
