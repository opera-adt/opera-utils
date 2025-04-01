"""Command-line interface for opera-utils."""

from __future__ import annotations

import json
import logging

import click

from opera_utils import burst_frame_db

from ._types import Bbox
from .burst_frame_db import get_frame_bbox


@click.group()
@click.version_option()
@click.option("--debug", is_flag=True, default=False)
@click.pass_context
def cli_app(ctx, debug):
    """opera-utils command-line interface."""
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler()
    logger = logging.getLogger("opera_utils")
    logger.setLevel(level)
    logger.addHandler(handler)


@cli_app.group()
@click.pass_context
def disp_s1(ctx):
    """Tools for working with DISP-S1 data."""
    pass


@cli_app.group()
@click.pass_context
def disp_nisar(ctx):
    """Tools for working with DISP-NISAR data."""


@disp_s1.command()
@click.argument("frame_id")
@click.option(
    "--latlon",
    "-l",
    is_flag=True,
    help="Output the bounding box in latitude/longitude (EPSG:4326)",
)
@click.option(
    "--bounds-only",
    "-b",
    is_flag=True,
    help="Output only (left, bottom, right, top) and omit EPSG",
)
def frame_bbox(frame_id, latlon: bool, bounds_only: bool):
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
        click.echo(list(bounds))
    else:
        obj = dict(epsg=epsg, bbox=bounds)
        click.echo(json.dumps(obj))


@disp_s1.command()
@click.option("--bbox", type=float, nargs=4)
@click.option("--point", type=float, nargs=2)
@click.option("--ids-only", is_flag=True, default=False)
def intersects(
    bbox: tuple[float, float, float, float], point: tuple[float, float], ids_only: bool
):
    """Get the frames that intersect with the given bounding box."""
    geom = Bbox(point[0], point[1], point[0], point[1]) if point else Bbox(*bbox)
    frames = burst_frame_db.get_intersecting_frames(geom, ids_only=ids_only)
    if ids_only:
        click.echo("\n".join(map(str, frames)))
    else:
        click.echo(frames)


@disp_s1.command()
@click.argument("namelist", type=click.File("r"))
@click.option(
    "--write-options/--no-write-options",
    default=True,
    help="Write out each option to a text file.",
)
@click.option(
    "--output-prefix",
    type=str,
    default="option_",
    help="Prefix for output filenames.",
    show_default=True,
)
@click.option(
    "--max-options", type=int, default=5, help="Maximum number of options to show"
)
def missing_data_options(
    namelist, write_options: bool, output_prefix: str, max_options: int
):
    """Get a list of options for how to handle missing S1 data.

    Prints a table of options to stdout, and writes the subset
    of files to disk for each option with names like

    option_1_bursts_1234_burst_ids_27_dates_10.txt
    """
    from opera_utils import filter_by_burst_id, filter_by_date, get_missing_data_options
    from opera_utils.missing_data import print_plain, print_with_rich

    file_list = [f.strip() for f in namelist.read().splitlines()]
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
        click.echo(f"Writing {len(valid_files)} files to {cur_output}")
        with open(cur_output, "w") as f:
            f.write("\n".join(valid_files))
            f.write("\n")
