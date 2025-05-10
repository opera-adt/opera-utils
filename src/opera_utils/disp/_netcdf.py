from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5netcdf
import numpy as np
import pyproj

from ._product import DispProductStack

__all__ = ["save_data"]

CHUNK_SIZE = (15, 256, 256)
DEFAULT_HDF5_OPTIONS = {
    "chunks": CHUNK_SIZE,
    "compression": "gzip",
    "compression_opts": 4,
    "shuffle": True,
}
GRID_MAPPING_DSET = "spatial_ref"


def save_data(
    data: np.ndarray,
    product_stack: DispProductStack,
    output: Path | str,
    *,
    dataset_name: str,
    rows: slice | None = None,
    cols: slice | None = None,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Save displacement data to a NetCDF file.

    Parameters
    ----------
    data : np.ndarray
        3D array of displacement values [time, height, width]
    product_stack : DispProductStack
        Stack of products used to create `data`.
    output : Path | str
        Name of file to save results to
    dataset_name : str
        Name of dataset used to create `data`.
    rows : slice | None, optional
        The subset of rows of the full product frame used to make `data`.
    cols : slice | None, optional
        The subset of columns of the full product frame used to make `data`.
    attrs : Optional[dict[str, Any]], optional
        Attributes to save with the dataset.

    """
    if attrs:
        long_name = attrs.get("long_name", dataset_name)
        description = attrs.get("description", dataset_name)
    with h5netcdf.File(output, "w") as f:
        _create_grid_mapping(
            group=f, crs=product_stack.crs, gt=product_stack.transform.to_gdal()
        )

        full_y, full_x = product_stack.y, product_stack.x
        y = full_y[rows] if rows is not None else full_y
        x = full_x[cols] if cols is not None else full_x
        _create_dimension_variables(
            group=f, datetimes=product_stack.secondary_dates, y=y, x=x
        )
        _create_geo_dataset(
            group=f,
            name=dataset_name,
            data=data,
            long_name=long_name,
            description=description,
            fillvalue=np.nan,
            attrs=attrs,
        )


def _create_geo_dataset(
    *,
    group: h5netcdf.Group,
    name: str,
    data: np.ndarray,
    long_name: str,
    description: str,
    fillvalue: float,
    attrs: dict[str, Any] | None,
    grid_mapping_dset_name=GRID_MAPPING_DSET,
) -> h5netcdf.Variable:
    if data.ndim != 3:
        msg = "Data must be 3D"
        raise ValueError(msg)
    dimensions = ["time", "y", "x"]
    if attrs is None:
        attrs = {}

    attrs.update(description=description)
    if long_name:
        attrs["long_name"] = long_name

    # Make sure the chunks set are not larger than the data:
    chunks = tuple(min(CHUNK_SIZE[i], data.shape[i]) for i in range(3))
    DEFAULT_HDF5_OPTIONS["chunks"] = chunks

    dset = group.create_variable(
        name,
        dimensions=dimensions,
        data=data,
        fillvalue=fillvalue,
        **DEFAULT_HDF5_OPTIONS,
    )
    dset.attrs.update(attrs)
    dset.attrs["grid_mapping"] = grid_mapping_dset_name
    return dset


def _create_dimension_variables(
    group: h5netcdf.Group, datetimes: Sequence[datetime], y: np.ndarray, x: np.ndarray
) -> tuple[h5netcdf.Variable, h5netcdf.Variable, h5netcdf.Variable]:
    """Create the y, x, and coordinate datasets."""
    ny = len(y)
    nx = len(x)
    nt = len(datetimes)

    if not group.dimensions:
        dims = {"y": ny, "x": nx, "time": nt}
        group.dimensions = dims

    # Create the x/y datasets
    y_ds = group.create_variable("y", ("y",), data=y, dtype=float)
    x_ds = group.create_variable("x", ("x",), data=x, dtype=float)

    for name, ds in zip(["y", "x"], [y_ds, x_ds]):
        ds.attrs["standard_name"] = f"projection_{name}_coordinate"
        ds.attrs["long_name"] = f"{name.replace('_', ' ')} coordinate of projection"
        ds.attrs["units"] = "m"

    # Create the time coordinate dataset."""
    times, calendar, units = _create_time_array(datetimes)
    t_ds = group.create_variable("time", ("time",), data=times, dtype=float)
    t_ds.attrs["standard_name"] = "time"
    t_ds.attrs["long_name"] = "time"
    t_ds.attrs["calendar"] = calendar
    t_ds.attrs["units"] = units

    return t_ds, y_ds, x_ds


def _create_time_array(times: Sequence[datetime]):
    """Set up the CF-compliant time array and dimension metadata.

    References
    ----------
    http://cfconventions.org/cf-conventions/cf-conventions.html#time-coordinate

    """
    since_time = datetime(2010, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    time = np.array(
        [(t.astimezone(timezone.utc) - since_time).total_seconds() for t in times]
    )
    calendar = "standard"
    units = f"seconds since {since_time.strftime('%Y-%m-%d %H:%M:%S.%f')}"
    return time, calendar, units


def _create_grid_mapping(
    group: h5netcdf.Group,
    crs: pyproj.CRS,
    gt: list[float],
    name: str = GRID_MAPPING_DSET,
) -> h5netcdf.Variable:
    """Set up the grid mapping variable."""
    # https://github.com/corteva/rioxarray/blob/21284f67db536d9c104aa872ab0bbc261259e59e/rioxarray/rioxarray.py#L34
    dset = group.create_variable(name, (), data=0, dtype=int)

    dset.attrs.update(crs.to_cf())
    # Also add the GeoTransform
    gt_string = " ".join([str(x) for x in gt])
    dset.attrs.update(
        {
            "GeoTransform": gt_string,
            "units": "unitless",
            "long_name": "Dummy variable with geo-referencing metadata in attributes",
        }
    )

    return dset
