from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from ._dates import get_dates
from ._types import Filename
from .bursts import group_by_burst

logger = logging.getLogger(__name__)

__all__ = [
    "BurstSubsetOption",
    "get_burst_id_to_dates",
    "get_missing_data_options",
    "get_burst_id_date_incidence",
]


@dataclass(frozen=True)
class BurstSubsetOption:
    """Dataclass for a possible subset of SLC data."""

    num_dates: int
    """Number of dates used in this subset"""
    num_burst_ids: int
    """Number of burst IDs used in this subset."""
    total_num_bursts: int
    """Total number of bursts used in this subset."""
    burst_id_list: list[str]
    """List of burst IDs used in this subset."""


def get_burst_id_to_dates(
    slc_files: Optional[Iterable[Filename]] = None,
    burst_id_date_tuples: Optional[Iterable[tuple[str, date]]] = None,
) -> dict[str, list[date]]:
    """Get a mapping of burst ID to list of dates.

    Assumes that the `slc_files` have only one date in the name, or
    that the first date in the `burst_id_date_tuples` is the relevant
    one (as is the case for OPERA CSLCs).


    Parameters
    ----------
    slc_files : Optional[Iterable[Filename]]
        List of OPERA CSLC filenames.
    burst_id_date_tuples : Optional[Iterable[tuple[str, date]]]
        Alternative input: list of all existing (burst_id, date) tuples.

    Returns
    -------
    dict[str, list[date]]
        Mapping of burst ID to list of dates.
    """
    if slc_files is not None:
        return _burst_id_mapping_from_files(slc_files)
    elif burst_id_date_tuples is not None:
        return _burst_id_mapping_from_tuples(burst_id_date_tuples)
    else:
        raise ValueError("Must provide either slc_files or burst_id_date_tuples")


def get_missing_data_options(
    slc_files: Optional[Iterable[Filename]] = None,
    burst_id_date_tuples: Optional[Iterable[tuple[str, date]]] = None,
) -> list[BurstSubsetOption]:
    """Get a list of possible data subsets for a set of burst SLCs.

    The default optimization criteria for choosing among these subsets is

        maximize        total number of bursts used
        subject to      dates used for each burst ID are all equal

    The constraint that the same dates are used for each burst ID is to
    avoid spatial discontinuities the estimated displacement/velocity,
    which can occur if different dates are used for different burst IDs.

    Parameters
    ----------
    slc_files : Optional[Iterable[Filename]]
        list of OPERA CSLC filenames.
    burst_id_date_tuples : Optional[Iterable[tuple[str, date]]]
        Alternative input: list of all existing (burst_id, date) tuples.

    Returns
    -------
    list[BurstSubsetOption]
        List of possible subsets of the given SLC data.
        The options will be sorted by the total number of bursts used, so
        that the first option is the one that uses the most data.
    """
    burst_id_to_dates = get_burst_id_to_dates(
        slc_files=slc_files, burst_id_date_tuples=burst_id_date_tuples
    )

    all_burst_ids = list(burst_id_to_dates.keys())

    B = get_burst_id_date_incidence(burst_id_to_dates)
    # In this matrix,
    # - Each row corresponds to one of the possible burst IDs
    # - Each column corresponds to one of the possible dates
    # Getting the unique rows of this matrix will give us the possible
    # combinations of dates that can be used for each burst ID.
    unique_date_idxs, burst_id_counts = np.unique(B, axis=0, return_counts=True)
    out = []

    for date_idxs in unique_date_idxs:
        required_num_dates = date_idxs.sum()
        keep_burst_idxs = np.array(
            [required_num_dates == burst_row[date_idxs].sum() for burst_row in B]
        )
        # B.shape: (num_burst_ids, num_dates)
        cur_burst_total = B[keep_burst_idxs, :][:, date_idxs].sum()

        cur_burst_id_list = np.array(all_burst_ids)[keep_burst_idxs].tolist()
        out.append(
            BurstSubsetOption(
                num_dates=required_num_dates,
                num_burst_ids=len(cur_burst_id_list),
                total_num_bursts=cur_burst_total,
                burst_id_list=cur_burst_id_list,
            )
        )
    return sorted(out, key=lambda x: x.total_num_bursts, reverse=True)


def get_burst_id_date_incidence(
    burst_id_to_dates: Mapping[str, list[date]]
) -> NDArray[np.bool]:
    """Create a matrix of burst ID vs. date incidence.

    Parameters
    ----------
    burst_id_to_dates : Mapping[str, list[date]]
        Mapping of burst ID to list of dates.

    Returns
    -------
    NDArray[bool]
        Matrix of burst ID vs. date incidence.
        Rows correspond to burst IDs, columns correspond to dates.
        A value of True indicates that the burst ID was acquired on that date.
    """
    all_dates = _sorted_deduped_values(burst_id_to_dates)

    # Construct the incidence matrix of dates vs. burst IDs
    burst_id_to_date_incidence = {}
    for burst_id, date_list in burst_id_to_dates.items():
        cur_incidences = np.zeros(len(all_dates), dtype=bool)
        idxs = np.searchsorted(all_dates, date_list)
        cur_incidences[idxs] = True
        burst_id_to_date_incidence[burst_id] = cur_incidences

    return np.array(list(burst_id_to_date_incidence.values()))


def _sorted_deduped_values(in_mapping: Mapping[Any, list]):
    """Sort, dedupe, and concatenate all items in the lists of `in_mapping`'s values."""
    all_values = itertools.chain.from_iterable(in_mapping.values())
    return sorted(set(all_values))


def _burst_id_mapping_from_tuples(
    burst_id_date_tuples: Iterable[tuple[str, date]]
) -> dict[str, list[date]]:
    """Create a {burst_id -> [date,...]} (burst_id, date) tuples."""
    # Don't exhaust the iterator for multiple groupings
    burst_id_date_tuples = list(burst_id_date_tuples)

    # Group the possible SLC files by their date and by their Burst ID
    return {
        burst_id: [date for burst_id, date in g]
        for burst_id, g in itertools.groupby(burst_id_date_tuples, key=lambda x: x[0])
    }


def _burst_id_mapping_from_files(
    slc_files: Iterable[Filename],
) -> dict[str, list[date]]:
    """Create a {burst_id -> [date,...]} mapping from filenames."""
    # Don't exhaust the iterator for multiple groupings
    slc_file_list = list(map(str, slc_files))

    # Group the possible SLC files by their date and by their Burst ID
    burst_id_to_files = group_by_burst(slc_file_list)

    date_tuples = [get_dates(f) for f in slc_file_list]
    assert all(len(tup) == 1 for tup in date_tuples)

    return {
        burst_id: [get_dates(f)[0] for f in file_list]
        for (burst_id, file_list) in burst_id_to_files.items()
    }
