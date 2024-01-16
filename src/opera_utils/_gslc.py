from __future__ import annotations

import datetime
from typing import Any, Callable

import h5py

from ._types import Filename

__all__ = [
    "get_zero_doppler_time",
    "get_radar_wavelength",
]


def get_zero_doppler_time(
    filename: Filename, type_: str = "start"
) -> datetime.datetime:
    """Get the full acquisition time from the CSLC product.

    Uses `/identification/zero_doppler_{type_}_time` from the CSLC product.

    Parameters
    ----------
    filename : Filename
        Path to the CSLC product.
    type_ : str, optional
        Either "start" or "stop", by default "start".

    Returns
    -------
    str
        Full acquisition time.
    """
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

    def get_dt(in_str):
        return datetime.datetime.strptime(in_str.decode("utf-8"), DATETIME_FORMAT)

    dset = f"/identification/zero_doppler_{type_}_time"
    value = _get_dset_and_attrs(filename, dset, parse_func=get_dt)[0]
    return value


def _get_dset_and_attrs(
    filename: Filename,
    dset_name: str,
    parse_func: Callable = lambda x: x,
) -> tuple[Any, dict[str, Any]]:
    """Get one dataset's value and attributes from the CSLC product.

    Parameters
    ----------
    filename : Filename
        Path to the CSLC product.
    dset_name : str
        Name of the dataset.
    parse_func : Callable, optional
        Function to parse the dataset value, by default lambda x: x
        For example, could be parse_func=lambda x: x.decode("utf-8") to decode,
        or getting a datetime object from a string.

    Returns
    -------
    dset : Any
        The value of the scalar
    attrs : dict
        Attributes.
    """
    with h5py.File(filename, "r") as hf:
        dset = hf[dset_name]
        attrs = dict(dset.attrs)
        value = parse_func(dset[()])
        return value, attrs


def get_radar_wavelength(filename: Filename):
    """Get the radar wavelength from the CSLC product.

    Parameters
    ----------
    filename : Filename
        Path to the CSLC product.

    Returns
    -------
    wavelength : float
        Radar wavelength in meters.
    """
    dset = "/metadata/processing_information/input_burst_metadata/wavelength"
    value = _get_dset_and_attrs(filename, dset)[0]
    return value
