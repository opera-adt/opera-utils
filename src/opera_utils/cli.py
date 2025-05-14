"""Command-line interface for opera-utils."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from functools import partial

import tyro

from opera_utils import (
    burst_frame_db,
    filter_by_burst_id,
    filter_by_date,
    get_missing_data_options,
)
from opera_utils.missing_data import print_with_rich

from ._types import Bbox
from .burst_frame_db import get_frame_bbox


def frame_bbox(
    frame_id: int, /, latlon: bool = False, bounds_only: bool = False
) -> None:
    """Print the DISP-S1 EPSG/bounding box for FRAME_ID.

    Outputs as JSON string to stdout like
    {"epsg": 32618, "bbox": [157140.0, 4145220.0, 440520.0, 4375770.0]}

    unless `--bounds-only` is given, which prints a JSON of 4 numbers
        [left, bottom, right, top]

    Parameters
    ----------
    frame_id : int
        The ID of the frame to get the bounding box for.
    latlon : bool
        Print the bounds at latitude/longitude (in degrees).
        Default is False, meaning bounds are printed in UTM coordinates.
    bounds_only : bool
        Print only a JSON array of the bounds, not the EPSG code.

    """
    epsg, bounds = get_frame_bbox(frame_id=frame_id)
    if latlon:
        from opera_utils._helpers import reproject_bounds

        bounds = reproject_bounds(bounds, epsg, 4326)
    if bounds_only:
        print(list(bounds))
    else:
        obj = {"epsg": epsg, "bbox": bounds}
        print(json.dumps(obj))


def intersects(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    point: tuple[float, float] | None = None,
    ids_only: bool = False,
) -> None:
    """Get the DISP-S1 frames that intersect with the given bounding box.

    Parameters
    ----------
    bbox : tuple[float, float, float, float], optional
        Bounding box (in degrees longitude/latitude) to search for intersection.
        The four numbers are (west, south, east, north).
    point : tuple[float, float], optional
        Point as (longitude, latitude), in degrees, to search for intersection.
        Mututally exclusive with `bbox`.
    ids_only : bool
        Print only the Frame IDs as newline-separate ints.
        By default False, which returns a GeoJSON string of Frame geometries.

    """
    if bbox is None and point is None:
        msg = "Either bbox or point must be provided"
        raise ValueError(msg)

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
    with open(namelist) as f:
        file_list = [line.strip() for line in f.read().splitlines()]

    options = get_missing_data_options(file_list)[:max_options]

    print_with_rich(options)
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


def cli_app() -> None:
    """opera-utils command-line interface."""
    # Use subcommand_cli_from_dict to handle the top-level commands
    handler = logging.StreamHandler()
    logger = logging.getLogger("opera_utils")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    cli_dict: dict[str, Callable] = {
        "disp-s1-frame-bbox": frame_bbox,
        "disp-s1-intersects": intersects,
        "disp-s1-missing-data-options": missing_data_options,
    }
    try:
        from opera_utils.disp._search import search

        cli_dict["disp-s1-search"] = partial(search, print_urls=True)

    except ImportError:
        pass
    tyro.extras.subcommand_cli_from_dict(
        cli_dict,
        prog="opera-utils",
        description="opera-utils command-line interface.",
    )


if __name__ == "__main__":
    cli_app()
