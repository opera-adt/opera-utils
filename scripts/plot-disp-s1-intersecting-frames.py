#!/usr/bin/env python
# /// script
# dependencies = ["matplotlib", "shapely", "geopandas", "opera_utils", "tyro"]
# ///
"""Make a plot of DISP-S1 frames that intersect a bounding box.

Examples
--------
    python scripts/plot-intersecting-frames.py -104 31 -103 32  # interactive plot
    python scripts/plot-intersecting-frames.py -104 31 -103 32  --output frames.png

"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from shapely import box, from_wkt
from shapely import plotting as splot

import opera_utils


def _get_track_colors(gdf: gpd.GeoDataFrame, cmap_name: str = "tab10") -> dict:
    """Create a color mapping for unique track numbers in the GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        A GeoDataFrame with frame_id as index.
    cmap_name : str, optional
        The name of the matplotlib colormap to use, by default 'tab10'.

    Returns
    -------
    dict
        A dictionary mapping track numbers to colors from the specified colormap.

    """
    track_numbers = [_get_track(idx) for idx in gdf.index]
    unique_tracks = sorted(set(track_numbers))

    # Create a color mapping for each unique track
    cmap = get_cmap(cmap_name)
    colors = {
        track: cmap(i / len(unique_tracks)) for i, track in enumerate(unique_tracks)
    }
    return colors


def _get_track(frame_id):
    # Extract unique track numbers (first 2 digits of frame_id)
    first_burst = opera_utils.get_burst_ids_for_frame(frame_id)[0]
    # Example: "t042_088913_iw2"
    return int(first_burst[1:4])


def plot_frames_with_labels(
    bounds: tuple[float, float, float, float] | None = None,
    wkt: str | None = None,
    ascending: bool = True,
    descending: bool = True,
    cmap_name: str = "tab10",
    output: Path | str | None = None,
) -> plt.Figure:
    """Plot frame geometries with colored track labels.

    Parameters
    ----------
    bounds : tuple, optional
        A tuple of lat/lon (west, south, east, north) defining the area of interest.
    wkt : str, optional
        Alternative to bounds: a well-known text string of the area of interest in
        degrees lat/lon.
    ascending : bool
        Whether to include ascending frames in the plot.
        Default is True
    descending : bool
        Whether to include descending frames in the plot.
        Default is True
    cmap_name : str, optional
        The name of the matplotlib colormap to use, by default 'tab10'.
    output : Path | str, optional
        If provided, name to save plot to.
        If None, plot is shown interactively.

    Returns
    -------
    plt.Figure
        The figure containing the plotted frames.

    """
    if bounds is None:
        if wkt is None:
            msg = "Must provide wkt or bounds"
            raise ValueError(msg)
        bounds = from_wkt(wkt).bounds
    poly = box(*bounds)

    fig, ax = plt.subplots()
    gdf_frames = opera_utils.get_frame_geodataframe()
    gdf_intersect = gdf_frames[gdf_frames.intersects(poly)]
    if not ascending:
        gdf_intersect = gdf_intersect[gdf_intersect.orbit_pass != "ASCENDING"]
    if not descending:
        gdf_intersect = gdf_intersect[gdf_intersect.orbit_pass != "DESCENDING"]
    gdf_intersect.plot(ax=ax, facecolor="none")

    track_to_color = _get_track_colors(gdf_intersect, cmap_name=cmap_name)
    # Plot each polygon with color based on track number
    for frame_id, row in gdf_intersect.iterrows():
        track = _get_track(frame_id)
        color = track_to_color[track]

        # Plot polygon with track-specific color
        ax.plot(*row.geometry.exterior.xy, color=color, alpha=0.7, linewidth=2)

        # Determine rotation angle based on orbit_pass
        if row["orbit_pass"] == "ASCENDING":
            rotation = 10  # 10 degrees counterclockwise
        else:  # DESCENDING
            rotation = -10  # 10 degrees clockwise

        # Get the centroid of the polygon for text placement
        centroid = row.geometry.centroid

        # Add text annotation with the frame_id at the centroid
        ax.annotate(
            text=str(frame_id),
            xy=(centroid.x, centroid.y),
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            weight="bold",
            rotation=rotation,
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": color,
                "ec": "black",
                "alpha": 0.7,
            },
        )

    # Plot the bounding box
    splot.plot_polygon(poly, ax=ax, zorder=-1, color="red", alpha=0.2, linewidth=2)

    # Add a legend for track numbers
    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=2, label=f"Track {track}")
        for track, color in track_to_color.items()
    ]
    ax.legend(handles=legend_elements, loc="best", title="Track Numbers")

    fig.suptitle(f"DISP-S1 Frames intersecting {bounds}")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    if output:
        fig.savefig(output)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    import tyro

    tyro.cli(plot_frames_with_labels)
