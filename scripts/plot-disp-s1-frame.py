#!/usr/bin/env python
# /// script
# dependencies = [
#  "matplotlib"
#  "pykdtree"
#  "cartopy"
#  "geopandas"
#  "shapely"
#  "opera_utils"
#  "tyro"
# ]
# ///
"""Plot a DISP-S1 frame on a background map.

Examples
--------
    # Plot frame 11115 on a satellite background, interactive figure
    python scripts/plot-disp-s1-frame.py 11115

    # Plot frame 11115 on a satellite background, saving to "frame.png"
    python scripts/plot-disp-s1-frame.py 11115 --output frame.png

    # Plot frame 11115 with states outlines (no satellite imagery), more zoom
    python scripts/plot-disp-s1-frame.py 11115 --no-use-satellite --pad-degrees 2

"""

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from cartopy.mpl.geoaxes import GeoAxes
from shapely.ops import unary_union

import opera_utils


def map_background(
    bbox: tuple[float, float, float, float],
    use_satellite: bool = True,
    zoom_level: int = 8,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a DISP-S1 frame on a background base map.

    Parameters
    ----------
    bbox : tuple of float
        (west, south, east, north) bounding box for display.
    use_satellite : bool, optional
        If True, fetch satellite tiles; otherwise, just use a blank background
        with state outlines, by default True.
    zoom_level : int, optional
        Zoom level for background satellite tiles, by default 8.
    figsize : tuple of float, optional
        Size of the figure (width, height) in inches, by default (8, 6).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Figure.
    ax : matplotlib.axes.Axes
        The GeoAxes for further plotting.

    """
    fig = plt.figure(figsize=figsize)
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    west, south, east, north = bbox
    ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())

    if use_satellite:
        # Example: a Google or Mapbox tiles source.
        # Requires cartopy >= 0.20 and an internet connection.
        from cartopy.io.img_tiles import GoogleTiles

        tiler = GoogleTiles(style="satellite")
        ax.add_image(tiler, zoom_level)

    # Add states, coastlines
    shapename = "admin_1_states_provinces_lakes"
    states_shp = shpreader.natural_earth(
        resolution="10m", category="cultural", name=shapename
    )
    ax.add_geometries(
        shpreader.Reader(states_shp).geometries(),
        ccrs.PlateCarree(),
        facecolor="lightgray" if not use_satellite else "none",
        edgecolor="white",
        linewidth=2,
        zorder=5,
    )
    # ax.coastlines(resolution="10m", zorder=2)

    # Optionally add gridlines / ticks
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)
    gl.top_labels = gl.right_labels = False

    return fig, ax


def plot_frames_on_background(
    frame_ids: Sequence[int],
    /,
    output: Optional[Path] = None,
    use_satellite: bool = True,
    pad_degrees: float = 1.5,
    zoom_level: int = 8,
) -> plt.Figure:
    """Plot a single DISP-S1 frame geometry on a background map.

    Parameters
    ----------
    frame_ids : Sequence[int]
        DISP-S1 frame IDs to plot.
    output : Path or None, optional
        If provided, path to save the resulting plot.
        If None, an interactive window will be shown.
    use_satellite : bool, optional
        If True, fetch satellite tiles as background.
        If False, show states as plain background.
        By default True.
    pad_degrees : float, optional
        Padding in degrees around the frame geometry, by default 1.5.
    zoom_level : int, optional
        Zoom level for satellite tiles, by default 8.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plotted frame.

    """
    # Fetch all frames
    gdf_frames = opera_utils.get_frame_geodataframe()

    # Filter for the one we want
    gdf_this = gdf_frames.loc[list(frame_ids)]
    # If the geometry can be multi-part, unify them for plotting
    geometry_union = unary_union(gdf_this.geometry.values)

    # Get bounding box with a small padding around the polygon
    minx, miny, maxx, maxy = geometry_union.bounds
    bbox = (
        minx - pad_degrees,
        miny - pad_degrees,
        maxx + pad_degrees,
        maxy + pad_degrees,
    )

    # Prepare background
    fig, ax = map_background(
        bbox=bbox,
        use_satellite=use_satellite,
        zoom_level=zoom_level,
        figsize=(8, 6),
    )

    # Plot the frame polygon
    for frame_id, cur_geom in zip(frame_ids, gdf_this.geometry.values, strict=True):
        if cur_geom.geom_type == "Polygon":
            xs, ys = cur_geom.exterior.xy
            ax.plot(xs, ys, color="red", linewidth=2)
        else:
            # If multipolygon, plot each
            for part in cur_geom.geoms:
                xs, ys = part.exterior.xy
                ax.plot(xs, ys, color="red", linewidth=2)

        # Add an annotation in the centroid
        centroid = cur_geom.centroid
        ax.text(
            centroid.x,
            centroid.y,
            f"Frame {frame_id}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.3", "fc": "red", "ec": "none", "alpha": 0.8},
            transform=ccrs.PlateCarree(),
        )
    ax.set_title(f"DISP-S1 Frame {','.join(map(str, frame_ids))}")

    if output:
        fig.savefig(output, bbox_inches="tight", dpi=150)
    else:
        plt.show(block=True)

    return fig


if __name__ == "__main__":
    import tyro

    tyro.cli(plot_frames_on_background)
