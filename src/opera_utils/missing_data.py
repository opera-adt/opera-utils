from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import groupby
from typing import Any

import numpy as np

from ._dates import filter_by_date, get_dates
from ._helpers import flatten, powerset, sorted_deduped_values
from .bursts import filter_by_burst_id, group_by_burst

logger = logging.getLogger(__name__)

__all__ = [
    "BurstSubsetOption",
    "get_burst_id_date_incidence",
    "get_burst_id_to_dates",
    "get_missing_data_options",
]


@dataclass
class BurstSubsetOption:
    """Dataclass for a possible subset of SLC data."""

    total_num_bursts: int
    """Total number of bursts used in this subset."""
    burst_ids: tuple[str, ...]
    """Burst IDs used in this subset."""
    dates: tuple[datetime, ...]
    """Dates used in this subset."""
    # subset_selected: list[bool]
    num_candidate_bursts: int
    """The number of (burst_id, datetime) pairs that were passed as options."""
    inputs: list[Any] = field(default_factory=list)
    """The corresponding subset of inputs used to create this option."""

    @property
    def num_dates(self) -> int:
        return len(self.dates)

    @property
    def num_burst_ids(self) -> int:
        return len(self.burst_ids)


def get_missing_data_options(
    slc_files: Iterable[str] | None = None,
    burst_id_date_tuples: Iterable[tuple[str, datetime]] | None = None,
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
    slc_files : Optional[Iterable[str]]
        list of OPERA CSLC filenames/urls.
    burst_id_date_tuples : Optional[Iterable[tuple[str, datetime]]]
        Alternative input: list of all existing (burst_id, datetime) tuples.

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
    dupes = _duplicated_bursts(burst_id_to_dates)
    if dupes:
        s = "\n".join(f"{b}:{d.strftime('%Y%m%d')}" for (b, d) in dupes)
        msg = f"Duplicated (burst_id, datetime) pairs passed:\n{s}."
        # TODO: is there a better way to echo this back to them?
        raise ValueError(msg)

    all_burst_ids = list(burst_id_to_dates.keys())
    all_dates = sorted_deduped_values(burst_id_to_dates)

    B = get_burst_id_date_incidence(burst_id_to_dates)
    # In this matrix,
    # - Each row corresponds to one of the possible burst IDs
    # - Each column corresponds to one of the possible dates
    options = generate_burst_subset_options(B, all_burst_ids, all_dates)
    # If they gave strings/files/urls, pick those out
    if slc_files is not None:
        for option in options:
            cur_burst_ids = option.burst_ids
            # The selected files are those which match the selected date + burst IDs
            valid_date_files = filter_by_date(slc_files, option.dates)
            option.inputs = filter_by_burst_id(valid_date_files, cur_burst_ids)

    return options


def get_burst_id_to_dates(
    slc_files: Iterable[str] | None = None,
    burst_id_date_tuples: Iterable[tuple[str, datetime]] | None = None,
) -> dict[str, list[datetime]]:
    """Get a mapping of burst ID to list of dates.

    Assumes that the `slc_files` have only one datetime in the name, or
    that the first datetime in the `burst_id_date_tuples` is the relevant
    one (as is the case for OPERA CSLCs).


    Parameters
    ----------
    slc_files : Optional[Iterable[str]]
        List of OPERA CSLC filenames.
    burst_id_date_tuples : Optional[Iterable[tuple[str, datetime]]]
        Alternative input: list of all existing (burst_id, datetime) tuples.

    Returns
    -------
    dict[str, list[datetime]]
        Mapping of burst ID to list of dates.

    """
    if slc_files is not None:
        return _burst_id_mapping_from_files(slc_files)
    elif burst_id_date_tuples is not None:
        return _burst_id_mapping_from_tuples(burst_id_date_tuples)
    else:
        msg = "Must provide either slc_files or burst_id_date_tuples"
        raise ValueError(msg)


def _duplicated_bursts(burst_id_to_dates: Mapping[str, Sequence[datetime]]):
    from collections import Counter

    counts: Counter = Counter()
    for burst_id, d_list in burst_id_to_dates.items():
        for d in d_list:
            counts[(burst_id, d)] += 1
    return [pair for pair, count in counts.items() if count > 1]


def get_burst_id_date_incidence(
    burst_id_to_dates: Mapping[str, list[datetime]],
) -> np.ndarray:
    """Create a matrix of burst ID vs. datetime incidence.

    Parameters
    ----------
    burst_id_to_dates : Mapping[str, list[datetime]]
        Mapping of burst ID to list of dates.

    Returns
    -------
    np.ndarray[bool]
        Matrix of burst ID vs. datetime incidence.
        Rows correspond to burst IDs, columns correspond to dates.
        A value of True indicates that the burst ID was acquired on that datetime.

    """
    all_dates = sorted_deduped_values(burst_id_to_dates)

    # Construct the incidence matrix of dates vs. burst IDs
    burst_id_to_date_incidence = {}
    for burst_id, date_list in burst_id_to_dates.items():
        cur_incidences = np.zeros(len(all_dates), dtype=bool)
        idxs = np.searchsorted(all_dates, date_list)
        cur_incidences[idxs] = True
        burst_id_to_date_incidence[burst_id] = cur_incidences

    return np.array(list(burst_id_to_date_incidence.values()))


def _burst_id_mapping_from_tuples(
    burst_id_date_tuples: Iterable[tuple[str, datetime]],
) -> dict[str, list[datetime]]:
    """Create a {burst_id -> [datetime,...]} (burst_id, datetime) tuples."""
    # Don't exhaust the iterator for multiple groupings
    burst_id_date_tuples = sorted(burst_id_date_tuples)

    # Group the possible SLC files by their datetime and by their Burst ID
    return {
        burst_id: [d for burst_id, d in g]
        for burst_id, g in groupby(burst_id_date_tuples, key=lambda x: x[0])
    }


def _burst_id_mapping_from_files(
    slc_files: Iterable[str],
) -> dict[str, list[datetime]]:
    """Create a {burst_id -> [datetime,...]} mapping from filenames.

    Assumes the first datetime in the filename is the relevant one.
    """
    # Don't exhaust the iterator for multiple groupings
    slc_file_list = list(map(str, slc_files))

    # Group the possible SLC files by their datetime and by their Burst ID
    burst_id_to_files = group_by_burst(slc_file_list)

    return {
        burst_id: [get_dates(f)[0] for f in file_list]
        for (burst_id, file_list) in burst_id_to_files.items()
    }


def generate_burst_subset_options(
    B: np.ndarray,  # noqa: N803
    burst_ids: Sequence[str],
    dates: Sequence[datetime],
) -> list[BurstSubsetOption]:
    """Generate possible valid subsets of the given SLC data.

    Parameters
    ----------
    B : NDArray[np.bool]
        Matrix of burst ID vs. datetime incidence.
        Rows correspond to burst IDs, columns correspond to dates.
        A value of True indicates that the burst ID was acquired on that datetime.
    burst_ids : Sequence[str]
        List of all burst IDs.
    dates : Sequence[datetime]
        List of all dates.

    Returns
    -------
    list[BurstSubsetOption]
        List of possible subsets of the given SLC data.
        The options will be sorted by the total number of bursts used, so
        that the first option is the one that uses the most data.

    """
    options = []
    num_candidate_bursts = B.sum()
    logger.debug("Number of candidates: %s", num_candidate_bursts)
    # Get the idxs where there are any missing dates for each burst
    # We're going to try all possible combinations of these *groups*,
    # not all possible combinations of the individual missing dates
    missing_date_idxs = set()
    for row in B:
        missing_date_idxs.add(tuple(np.where(~row)[0]))

    # Generate all unique combinations of idxs to exclude
    date_idxs_to_exclude_combinations = []
    # NOTE: if `missing_date_idxs` is larger than ~25, this blows up
    # Since most cases take milliseconds, we'll set a cap at considering
    # a million (more than we need)
    for i, combo in enumerate(powerset(missing_date_idxs)):
        if i > 1e4:
            break
        date_idxs_to_exclude_combinations.append(set(flatten(combo)))

    all_column_idxs = set(range(B.shape[1]))
    all_row_idxs = set(range(B.shape[0]))

    # Track the row/col combinations that we've already
    tested_combinations = set()
    # Now iterate over these combinations
    for idxs_to_exclude in date_idxs_to_exclude_combinations:
        valid_col_idxs = all_column_idxs - idxs_to_exclude

        # Create sub-matrix with the remaining columns
        col_selector = sorted(valid_col_idxs)
        B_sub = B[:, col_selector]

        # We've decided which columns to exclude
        # Now we have to decide if we're throwing away rows
        # We'll get rid of any row that's not fully populated
        rows_to_exclude = set()
        for i, row in enumerate(B_sub):
            if not row.all():
                rows_to_exclude.add(i)

        # Get which indexes we're keeping
        valid_row_idxs = all_row_idxs - rows_to_exclude

        # Check if we've already tested this combination
        combo = (tuple(valid_row_idxs), tuple(valid_col_idxs))
        if combo in tested_combinations:
            continue
        tested_combinations.add(combo)

        # Remove the rows that we're excluding
        row_selector = sorted(valid_row_idxs)
        B_sub2 = B_sub[row_selector, :]

        # Check if all rows have the same pattern in the remaining columns
        if not (B_sub2.size > 0):
            logger.debug("No remaining entries in B_sub2")
            continue
        if not np.all(B_sub2 == B_sub2[[0]]):
            logger.debug("Not all rows have the same pattern in the remaining columns")
            continue
        # Create a BurstSubsetOption if we have at least one burst and one datetime
        assert np.all(B_sub2)

        selected_burst_ids = tuple(burst_ids[i] for i in valid_row_idxs)
        selected_dates = tuple(dates[i] for i in valid_col_idxs)
        total_num_bursts = B_sub2.sum()
        options.append(
            BurstSubsetOption(
                total_num_bursts=total_num_bursts,
                burst_ids=selected_burst_ids,
                dates=selected_dates,
                num_candidate_bursts=num_candidate_bursts,
            )
        )

    return sorted(
        options, key=lambda x: (x.total_num_bursts, x.num_burst_ids), reverse=True
    )


def print_with_rich(options: Iterable[BurstSubsetOption]) -> None:
    """Print a summary of the burst options using `rich.Table`."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Option", style="dim", width=6)
    table.add_column("# Dates", style="dim", width=10)
    table.add_column("# Burst IDs", style="dim", width=14)
    table.add_column("Selected Bursts", style="dim", width=15)
    table.add_column("Excluded Bursts", style="dim", width=15)

    for idx, option in enumerate(options, start=1):
        table.add_row(
            str(idx),
            str(option.num_dates),
            str(option.num_burst_ids),
            str(option.total_num_bursts),
            str(option.num_candidate_bursts - option.total_num_bursts),
        )
    console.print(table)
