from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from . import datasets
from ._types import Bbox, PathOrStr
from .bursts import normalize_burst_id

if TYPE_CHECKING:
    import geopandas


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
    frame_ids: Optional[Sequence[int | str]] = None,
    as_geodataframe: bool = False,
) -> dict | geopandas.GeoDataFrame:
    """Get the GeoJSON for the frame geometries."""
    data = _get_geojson(
        datasets.fetch_frame_geometries_simple(),
        as_geodataframe=as_geodataframe,
        fids=frame_ids,
        index_name="frame_id",
    )
    if as_geodataframe or not frame_ids:
        # `as_geodataframe` means it's already filtered
        return data

    # Manually filter for the case of no geopandas
    return {
        **data,
        "features": [
            f for f in data["features"] if f.get("id") in set(map(int, frame_ids))
        ],
    }


def get_burst_id_geojson(
    burst_ids: Optional[Sequence[str]] = None,
    as_geodataframe: bool = False,
) -> dict | geopandas.GeoDataFrame:
    """Get the GeoJSON for the burst_id geometries."""
    data = _get_geojson(
        datasets.fetch_burst_id_geometries_simple(),
        as_geodataframe=as_geodataframe,
        fids=burst_ids,
        index_name="burst_id_jpl",
    )
    if not burst_ids:
        return data

    if isinstance(burst_ids, str):
        burst_ids = [burst_ids]
    if as_geodataframe:
        assert isinstance(data, geopandas.GeoDataFrame)
        return data[data.burst_id_jpl.isin(tuple(burst_ids))]
    # Manually filter for the case of no geopandas
    return {
        **data,
        "features": [
            f
            for f in data["features"]
            if f["properties"]["burst_id_jpl"] in set(burst_ids)
        ],
    }


def _get_geojson(
    f: PathOrStr,
    as_geodataframe: bool = False,
    fids: Sequence[str | int] | None = None,
    index_name: Optional[str] = None,
) -> dict | geopandas.GeoDataFrame:
    # https://gdal.org/user/ogr_sql_dialect.html#where
    # https://pyogrio.readthedocs.io/en/latest/introduction.html#filter-records-by-attribute-value
    if as_geodataframe:
        return _get_frame_geodataframe(fids)

    return read_zipped_json(f)


def _get_frame_geodataframe(
    frame_ids: Optional[Sequence[int | str]] = None,
    json_file: Optional[PathOrStr] = None,
    index_name: Optional[str] = None,
) -> geopandas.GeoDataFrame:
    from pyogrio import read_dataframe

    if json_file is None:
        json_file = datasets.fetch_frame_geometries_simple()

    gdf = read_dataframe(json_file, layer=None, fid_as_index=True, fids=frame_ids)
    if index_name:
        gdf.index.name = index_name
    return gdf


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


def get_burst_to_frame_mapping(
    burst_id: str, json_file: Optional[PathOrStr] = None
) -> dict:
    """Get the burst data for one burst ID.

    Parameters
    ----------
    burst_id : str
        The ID of the burst to get the frame IDs for.
    json_file : PathOrStr, optional
        The path to the JSON file containing the burst-to-frame mapping.
        If `None`, uses the zip file fetched from `datasets`

    Returns
    -------
    dict
        The burst data for the given burst ID.
    """
    if json_file is None:
        json_file = datasets.fetch_burst_to_frame_mapping_file()
    js = read_zipped_json(json_file)
    return js["data"][normalize_burst_id(burst_id)]


def get_frame_ids_for_burst(
    burst_id: str, json_file: Optional[PathOrStr] = None
) -> list[int]:
    """Get the frame IDs for one burst ID.

    Parameters
    ----------
    burst_id : str
        The ID of the burst to get the frame IDs for.
    json_file : PathOrStr, optional
        The path to the JSON file containing the burst-to-frame mapping.
        If `None`, fetches the remote zip file from `datasets`

    Returns
    -------
    list[int]
        The frame IDs for the given burst ID.
        Most burst IDs have 1, but burst IDs in the overlap are in
        2 frames.
    """
    burst_data = get_burst_to_frame_mapping(burst_id, json_file)
    return burst_data["frame_ids"]


def get_intersecting_frames(bounds: Bbox, ids_only: bool = False) -> list[int]:
    """Get the frame IDs that intersect with the given bounds."""
    from shapely.geometry import box

    try:
        # gdf[gdf.geometry.intersects(Point(-123, 32))]
        # gdf = opera_utils.get_frame_geojson(as_geodataframe=True)
        gdf = _get_frame_geodataframe(index_name="frame_id")
    except ImportError as e:
        # TODO: decide if this is worth supporting without geopandas
        raise ImportError("geopandas is required for this function") from e

    frames = gdf[gdf.geometry.intersects(box(*bounds))]
    if ids_only:
        return frames.index.tolist()
    return frames.to_json()
