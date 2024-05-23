from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, Normalize
except ImportError:
    print("matplotlib is not installed: unable to use `missing_data_plots`")

import opera_utils
from opera_utils._types import PathOrStr


def plot_burst_id_date_incidence(
    slc_files: Optional[Iterable[PathOrStr]] = None,
    burst_id_date_tuples: Optional[Iterable[tuple[str, datetime]]] = None,
    ax=None,
    output_file: Optional[PathOrStr] = None,
) -> tuple[plt.figure, plt.Axes]:
    """Plot a matrix of burst ID vs. datetime incidence.

    Make a scatter plot of burst ID vs. datetime, where the x-axis spacing is uniform
    over time (dots may be irregularly spaced).

    Parameters
    ----------
    slc_files : Optional[Iterable[Filename]]
        List of OPERA CSLC filenames.
    burst_id_date_tuples : Optional[Iterable[tuple[str, datetime]]]
        Alternative input: list of all existing (burst_id, datetime) tuples.
    ax : Optional[matplotlib.axes.Axes]
        Axes to plot on. If None, a new figure and axes will be created.
    output_file : Optional[PathOrStr]
        Name of file to save figure to.

    Returns
    -------
    fig : matplotlib.Figure
        Matplotlib figure object of plot
    axes : matplotlib.pyploy.Axes
        Axes holding plot
    """
    burst_id_to_dates = opera_utils.missing_data.get_burst_id_to_dates(
        slc_files=slc_files, burst_id_date_tuples=burst_id_date_tuples
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7.5))
    else:
        fig = ax.figure

    B = opera_utils.get_burst_id_date_incidence(burst_id_to_dates)
    all_dates = opera_utils.missing_data.sorted_deduped_values(burst_id_to_dates)
    burst_ids = list(burst_id_to_dates.keys())
    ylabels = burst_ids
    cbar_labels = ["Missing", "Present"]

    # Both use the same yticks
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_ylabel("Burst ID")

    fig, ax = _plot_incidence_per_date(
        B,
        dates=all_dates,
        ax=ax,
        fig=fig,
        cbar_labels=cbar_labels,
    )
    fig.tight_layout()
    if output_file:
        _save_fig(fig, output_file=output_file, title="Acquisition history")
    return fig, ax


def _save_fig(fig, output_file: PathOrStr, title: str):
    fig.suptitle(title)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_file}")


def _plot_incidence_per_date(
    matrix: NDArray,
    *,
    dates: Sequence[datetime],
    fig,
    ax,
    colors: Sequence[str] = ["#a2cffe", "#fdae61"],
    cbar_labels: Sequence[str] = ["False", "True"],
):
    """Make a scatterplot with fixed datetime xaxis and variable spacing between items."""
    # Define a color map with two colors: one for True, one for False
    date_nums = mdates.date2num(dates)

    # Plot each point in the matrix
    xs, ys, cs = [], [], []
    for (i, j), value in np.ndenumerate(matrix):
        color = colors[int(value)]
        xs.append(date_nums[j])
        ys.append(i)
        cs.append(color)

    # Square marker
    marker = "s"
    ax.scatter(xs, ys, c=cs, s=80, marker=marker, edgecolors="k")

    # Define the colormap for the binary data and create a ScalarMappable object
    cmap = ListedColormap(colors)  # Colorblind-friendly colors
    norm = Normalize(vmin=0, vmax=1)  # Binary data, so only 0 and 1
    mappable = ScalarMappable(norm=norm, cmap=cmap)

    # Create the colorbar
    cbar = plt.colorbar(mappable, ticks=[0, 1])
    # Set the tick location to be in the middle of the color segment
    cbar.set_ticks([0.25, 0.75])
    cbar.ax.set_yticklabels(cbar_labels)  # Set the tick labels

    # Format the x-axis to display dates
    ax.xaxis_date()
    fig.autofmt_xdate()

    # Set the x-axis with the actual dates
    ax.set_xticks(date_nums)
    ax.set_xticklabels(dates, rotation="vertical", ha="center")
    fixed_dates = _generate_fixed_spacing_dates(dates)
    fixed_date_nums = mdates.date2num(fixed_dates)

    # Draw gridlines for each fixed datetime
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xticks(fixed_date_nums, minor=True)
    return fig, ax


def _generate_fixed_spacing_dates(dates: Sequence[datetime]) -> list[datetime]:
    """Generate a datetime list with fixed spacing using the given dates.

    Sets the interval to be the smallest timedelta from `dates`.

    Parameters
    ----------
    dates : list
        A list of datetime strings.
    date_format : str
        The format of the datetime strings in the list.

    Returns
    -------
    list[datetime]
        A list of datetime objects with fixed spacing.
    """
    date_list = list(dates)
    date_list.sort()

    # Calculate the smallest interval between consecutive dates
    min_interval = min(
        (date_list[i + 1] - date_list[i] for i in range(len(date_list) - 1)),
        default=timedelta(days=1),
    )

    # Generate new date_list array with the fixed smallest interval
    start_date, end_date = date_list[0], date_list[-1]
    num_days = (end_date - start_date) // min_interval
    return [start_date + i * min_interval for i in range(num_days + 1)]
