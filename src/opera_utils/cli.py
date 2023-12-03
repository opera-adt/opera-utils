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
    """Orca command-line interface."""
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
