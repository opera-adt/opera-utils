"""Command-line interface for opera-utils."""

from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

import tyro

from opera_utils import (
    burst_frame_db,
    filter_by_burst_id,
    filter_by_date,
    get_missing_data_options,
)
from opera_utils.missing_data import print_plain, print_with_rich

from ._types import Bbox
from .burst_frame_db import get_frame_bbox


# Define disp-s1 subcommands
def frame_bbox(
    frame_id: int, /, latlon: bool = False, bounds_only: bool = False
) -> None:
    """Look up the DISP-S1 EPSG/bounding box for FRAME_ID.

    Outputs as JSON string to stdout like
    {"epsg": 32618, "bbox": [157140.0, 4145220.0, 440520.0, 4375770.0]}

    Unless `--bounds-only` is given
    """
    epsg, bounds = get_frame_bbox(frame_id=frame_id)
    if latlon:
        from opera_utils._helpers import reproject_bounds

        bounds = reproject_bounds(bounds, epsg, 4326)
    if bounds_only:
        print(list(bounds))
    else:
        obj = dict(epsg=epsg, bbox=bounds)
        print(json.dumps(obj))


def intersects(
    *,  # keyword-only arguments
    bbox: Optional[Tuple[float, float, float, float]] = None,
    point: Optional[Tuple[float, float]] = None,
    ids_only: bool = False,
) -> None:
    """Get the DISP-S1 frames that intersect with the given bounding box."""
    if bbox is None and point is None:
        raise ValueError("Either bbox or point must be provided")

    if point is not None:
        geom = Bbox(point[0], point[1], point[0], point[1])
    else:
        assert bbox is not None
        geom = Bbox(*bbox)
    frames = burst_frame_db.get_intersecting_frames(geom)
    if ids_only:
        print("\n".join([f["id"] for f in frames["features"]]))
    else:
        print(json.dumps(frames))


def missing_data_options(
    namelist: str,
    /,
    write_options: bool = True,
    output_prefix: str = "option_",
    max_options: int = 5,
) -> None:
    """Get a list of options for how to handle missing S1 data.

    Prints a table of options to stdout, and writes the subset
    of files to disk for each option with names like

    option_1_bursts_1234_burst_ids_27_dates_10.txt
    """
    with open(namelist, "r") as f:
        file_list = [line.strip() for line in f.read().splitlines()]

    options = get_missing_data_options(file_list)[:max_options]

    try:
        print_with_rich(options)
    except ImportError:
        print_plain(options)
    if not write_options:
        return

    # Now filter the files by the burst ids in each, and output to separate
    for idx, option in enumerate(options, start=1):
        cur_burst_ids = option.burst_ids
        # The selected files are those which match the selected date + burst IDs
        valid_date_files = filter_by_date(file_list, option.dates)
        valid_files = filter_by_burst_id(valid_date_files, cur_burst_ids)

        cur_output = (
            f"{output_prefix}{idx}"
            f"_bursts_{option.total_num_bursts}"
            f"_burst_ids_{len(cur_burst_ids)}"
            f"_dates_{option.num_dates}.txt"
        )
        print(f"Writing {len(valid_files)} files to {cur_output}")
        with open(cur_output, "w") as f:
            f.write("\n".join(valid_files))
            f.write("\n")


# Define sub-commands for each top-level command
def cmd(debug: bool = False) -> None:
    """Tools for working with DISP-S1 data."""
    tyro.extras.subcommand_cli_from_dict(
        {
            "frame-bbox": frame_bbox,
            "intersects": intersects,
            "missing-data-options": missing_data_options,
        },
        prog="opera-utils disp-s1",
        description="Tools for working with DISP-S1 data.",
    )


# Main entry point
def cli_app() -> None:
    """opera-utils command-line interface."""
    # Use subcommand_cli_from_dict to handle the top-level commands
    handler = logging.StreamHandler()
    logger = logging.getLogger("opera_utils")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    tyro.extras.subcommand_cli_from_dict(
        {
            "disp-s1-frame-bbox": frame_bbox,
            "disp-s1-intersects": intersects,
            "disp-s1-missing-data-options": missing_data_options,
        },
        prog="opera-utils",
        description="opera-utils command-line interface.",
    )


if __name__ == "__main__":
    cli_app()
