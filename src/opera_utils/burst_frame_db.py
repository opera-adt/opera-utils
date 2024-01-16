from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Optional, Sequence

from . import datasets
from ._types import Bbox, PathOrStr


def read_zipped_json(filename: PathOrStr):
    """Read a zipped JSON file and returns its contents as a dictionary.

    Parameters
    ----------
    filename : PathOrStr
        The path to the zipped JSON file.

    Returns
    -------
    dict
        The contents of the zipped JSON file as a dictionary.
    """
    if Path(filename).suffix == ".zip":
        with zipfile.ZipFile(filename) as zf:
            bytes = zf.read(str(Path(filename).name).replace(".zip", ""))
            return json.loads(bytes.decode())
    else:
        with open(filename) as f:
            return json.load(f)


def get_frame_to_burst_mapping(
    frame_id: int, json_file: Optional[PathOrStr] = None
) -> dict:
    """Get the frame data for one frame ID.

    Parameters
    ----------
    frame_id : int
        The ID of the frame to get the bounding box for.
    json_file : PathOrStr, optional
        The path to the JSON file containing the frame-to-burst mapping.
        If `None`, uses the zip file contained in `data/`
    Returns
    -------
    dict
        The frame data for the given frame ID.
    """
    if json_file is None:
        json_file = datasets.fetch_frame_to_burst_mapping_file()
    js = read_zipped_json(json_file)
    return js["data"][str(frame_id)]


def get_frame_geojson(
    as_geodataframe: bool = False,
    columns: Optional[Sequence[str]] = None,
    frame_ids: Optional[Sequence[str]] = None,
) -> dict:
    """Get the GeoJSON for the frame geometries."""
    where = _form_where_in_query(frame_ids, "frame_id") if frame_ids else None
    return _get_geojson(
        datasets.fetch_frame_geometries_simple(),
        as_geodataframe=as_geodataframe,
        columns=columns,
        where=where,
        index_name="frame_id",
    )


def get_burst_id_geojson(
    as_geodataframe: bool = False,
    columns: Optional[Sequence[str]] = None,
    burst_ids: Optional[Sequence[str]] = None,
) -> dict:
    """Get the GeoJSON for the burst_id geometries."""
    where = _form_where_in_query(burst_ids, "burst_id_jpl") if burst_ids else None
    return _get_geojson(
        datasets.fetch_burst_id_geometries_simple(),
        as_geodataframe=as_geodataframe,
        columns=columns,
        where=where,
        index_name="burst_id_jpl",
    )


def _form_where_in_query(values: Sequence[str], column_name):
    # Example:
    # "burst_id_jpl in ('t005_009471_iw2','t007_013706_iw2','t008_015794_iw1')"
    where_in_str = ",".join(f"'{b}'" for b in values)
    return f"{column_name} IN ({where_in_str})"


def _get_geojson(
    f,
    as_geodataframe: bool = False,
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    index_name: Optional[str] = None,
) -> dict:
    # https://gdal.org/user/ogr_sql_dialect.html#where
    # https://pyogrio.readthedocs.io/en/latest/introduction.html#filter-records-by-attribute-value
    if as_geodataframe:
        from pyogrio import read_dataframe

        # import geopandas as gpd
        # return gpd.read_file(f)
        gdf = read_dataframe(f, columns=columns, where=where, fid_as_index=True)
        if index_name:
            gdf.index.name = index_name
        return gdf

    return read_zipped_json(f)


def get_frame_bbox(
    frame_id: int, json_file: Optional[PathOrStr] = None
) -> tuple[int, Bbox]:
    """Get the bounding box of a frame from a JSON file.

    Parameters
    ----------
    frame_id : int
        The ID of the frame to get the bounding box for.
    json_file : PathOrStr, optional
        The path to the JSON file containing the frame-to-burst mapping.
        If `None`, fetches the remote zip file from `datasets`

    Returns
    -------
    epsg : int
        EPSG code for the bounds coordinates
    tuple[float, float, float, float]
        bounding box coordinates (xmin, ymin, xmax, ymax)
    """
    frame_dict = get_frame_to_burst_mapping(frame_id=frame_id, json_file=json_file)
    epsg = int(frame_dict["epsg"])
    bounds = (
        float(frame_dict["xmin"]),
        float(frame_dict["ymin"]),
        float(frame_dict["xmax"]),
        float(frame_dict["ymax"]),
    )
    return epsg, Bbox(*bounds)


def get_burst_ids_for_frame(
    frame_id: int, json_file: Optional[PathOrStr] = None
) -> list[str]:
    """Get the burst IDs for one frame ID.

    Parameters
    ----------
    frame_id : int
        The ID of the frame to get the bounding box for.
    json_file : PathOrStr, optional
        The path to the JSON file containing the frame-to-burst mapping.
        If `None`, fetches the remote zip file from `datasets`

    Returns
    -------
    list[str]
        The burst IDs for the given frame ID.
    """
    frame_data = get_frame_to_burst_mapping(frame_id, json_file)
    return frame_data["burst_ids"]
