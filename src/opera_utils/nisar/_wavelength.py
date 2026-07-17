"""Radar wavelength extraction for NISAR GSLC products."""

from __future__ import annotations

import re

import numpy as np

from opera_utils._remote import open_h5
from opera_utils._types import PathOrStr
from opera_utils.constants import SPEED_OF_LIGHT

__all__ = ["get_nisar_wavelength"]

CENTER_FREQUENCY_DATASET = "centerFrequency"

# Match a full "frequencyA"/"frequencyB" path segment (not a substring of a
# longer segment name).
_FREQUENCY_GROUP_REGEX = re.compile(r"(?:^|/)frequency[AB](?=/|$)")


def get_nisar_wavelength(filename: PathOrStr, subdataset: str) -> float:
    """Get the radar wavelength for one frequency group of a NISAR GSLC product.

    The frequency group is located from `subdataset`, so the wavelength always
    corresponds to the same radar band and frequency sub-band as the data being
    processed (e.g. the subdataset ``/science/LSAR/GSLC/grids/frequencyA/HH``
    reads ``/science/LSAR/GSLC/grids/frequencyA/centerFrequency``).

    Parameters
    ----------
    filename : PathOrStr
        Path or URL of the NISAR GSLC HDF5 file.
        Remote (HTTPS/S3) URLs are opened via `opera_utils.open_h5`.
    subdataset : str
        HDF5 path to the dataset being processed, or to its frequency group.
        Must contain exactly one ``frequencyA`` or ``frequencyB`` path segment.

    Returns
    -------
    float
        Radar wavelength in meters, computed as the speed of light divided by
        the frequency group's ``centerFrequency``.

    Raises
    ------
    ValueError
        If `subdataset` does not contain exactly one frequency group segment,
        or if ``centerFrequency`` is missing or is not a positive, finite,
        real-valued scalar.

    """
    group_path = _get_frequency_group_path(subdataset)
    return SPEED_OF_LIGHT / _read_center_frequency(filename, group_path)


def _get_frequency_group_path(subdataset: str) -> str:
    """Extract the path up to the frequency group from a subdataset path."""
    matches = list(_FREQUENCY_GROUP_REGEX.finditer(subdataset))
    if len(matches) != 1:
        msg = (
            "Expected exactly one 'frequencyA' or 'frequencyB' path segment in"
            f" subdataset {subdataset!r}, found {len(matches)}"
        )
        raise ValueError(msg)
    return subdataset[: matches[0].end()]


def _read_center_frequency(filename: PathOrStr, frequency_group_path: str) -> float:
    """Read and validate the center frequency (in Hz) of one frequency group.

    Parameters
    ----------
    filename : PathOrStr
        Path or URL of the NISAR GSLC HDF5 file.
    frequency_group_path : str
        HDF5 path of the frequency group,
        e.g. ``/science/LSAR/GSLC/grids/frequencyA``.

    Returns
    -------
    float
        Center frequency in Hz.

    """
    dataset_path = f"{frequency_group_path}/{CENTER_FREQUENCY_DATASET}"
    err_prefix = (
        f"Invalid {CENTER_FREQUENCY_DATASET} at {dataset_path!r} in {str(filename)!r}"
    )

    with open_h5(str(filename)) as hf:
        if dataset_path not in hf:
            msg = f"{err_prefix}: dataset not found"
            raise ValueError(msg)
        dset = hf[dataset_path]
        # A group named `centerFrequency` has no shape and fails here too
        if getattr(dset, "shape", None) != ():
            msg = f"{err_prefix}: expected a scalar dataset"
            raise ValueError(msg)
        if not (
            np.issubdtype(dset.dtype, np.floating)
            or np.issubdtype(dset.dtype, np.integer)
        ):
            msg = f"{err_prefix}: expected a real-valued dtype, got {dset.dtype}"
            raise ValueError(msg)
        center_frequency = float(dset[()])

    if not np.isfinite(center_frequency) or center_frequency <= 0:
        msg = f"{err_prefix}: expected a positive, finite value, got {center_frequency}"
        raise ValueError(msg)
    return center_frequency
