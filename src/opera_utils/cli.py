"""Command-line interface for opera-utils."""

from __future__ import annotations

import json
import logging

import click

from .burst_frame_db import get_frame_bbox


@click.group()
@click.version_option()
@click.option("--debug", is_flag=True, default=False)
@click.pass_context
def cli_app(ctx, debug):
    """opera-utils command-line interface."""
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler()
    logging.basicConfig(level=level, handlers=[handler])


@cli_app.command()
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
    """Look up the EPSG/bounding box for FRAME_ID.

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


@cli_app.command()
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
    "--verbose", is_flag=True, help="Print full list of burst IDs for each option"
)
def missing_data_options(
    namelist, write_options: bool, output_prefix: str, verbose: bool
):
    """Get a list of options for how to handle missing data.

    Prints a table of options to stdout, and writes the subset
    of files to disk for each option with names like

    option_1_bursts_1234_burst_ids_27_dates_10.txt
    """
    from opera_utils import filter_by_burst_id, filter_by_date, get_missing_data_options

    def print_plain(options):
        header = f"|{'Option':<6}| {'# Dates':<10}| {'# Burst IDs':<14}|"
        header += f" {'Selected Bursts':<15}| {'Excluded Bursts':<15}| "
        print(header)
        print("-" * len(header))

        for idx, option in enumerate(options, start=1):
            excluded = option.num_candidate_bursts - option.total_num_bursts
            row = f"|{idx:<6}| {option.num_dates:<10}| {option.num_burst_ids:<14}|"
            row += f" {option.total_num_bursts:<15}| {excluded:<15}|"
            print(row)
        print()

    def print_with_rich(options):
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

    file_list = [f.strip() for f in namelist.read().splitlines()]
    options = get_missing_data_options(file_list)

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
