from __future__ import annotations

import importlib.util
import json
import zipfile
from collections.abc import Sequence
from enum import Enum
from functools import cache
from os import fsdecode
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union, overload

from . import datasets
from ._types import Bbox, PathOrStr
from .bursts import normalize_burst_id

if TYPE_CHECKING:
    import geopandas

GeojsonOrGdf = Union[dict, "geopandas.GeoDataFrame"]

# Check if geopandas is available
_has_geopandas = (
    importlib.util.find_spec("pyogrio") is not None
    and importlib.util.find_spec("geopandas") is not None
)


class OrbitPass(str, Enum):
    """Choices for the orbit direction of a granule."""

    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"

    def __str__(self) -> str:
        return str(self.value)


@cache
def read_zipped_json(filename: Path | str) -> dict:
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
            b = zf.read(str(Path(filename).name).replace(".zip", ""))
            return json.loads(b.decode())
    else:
        with open(filename) as f:
            return json.load(f)


def get_frame_to_burst_mapping(
    frame_id: int, json_file: PathOrStr | None = None
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
    js = read_zipped_json(fsdecode(json_file))
    return js["data"][str(frame_id)]


@overload
def get_frame_geojson(
    frame_ids: Sequence[int | str] | None = None,
    as_geodataframe: Literal[True] = True,
    json_file: PathOrStr | None = None,
) -> geopandas.GeoDataFrame: ...


@overload
def get_frame_geojson(
    frame_ids: Sequence[int | str] | None = None,
    as_geodataframe: Literal[False] = False,
    json_file: PathOrStr | None = None,
) -> dict: ...


def get_frame_geojson(
    frame_ids: Sequence[int | str] | None = None,
    as_geodataframe: bool = False,
    json_file: PathOrStr | None = None,
) -> GeojsonOrGdf:
    """Get the GeoJSON or GeoDataFrame for the frame geometries.

    Parameters
    ----------
    frame_ids : Sequence[int | str], optional
        Frame IDs to filter by. If None, returns all frames.
    as_geodataframe : bool, default=False
        If True, returns a GeoDataFrame. If False, returns a GeoJSON dict.
    json_file : PathOrStr, optional
        The path to the JSON file containing the frame geometries.
        If None, uses the default file from datasets.

    Returns
    -------
    dict or geopandas.GeoDataFrame
        Frame geometries as GeoJSON or GeoDataFrame.

    """
    if as_geodataframe:
        return get_frame_geodataframe(frame_ids, json_file=json_file)

    data = read_zipped_json(datasets.fetch_frame_geometries_simple())
    if not frame_ids:
        return data

    # Manually filter for the case of no geopandas
    return {
        **data,
        "features": [
            f for f in data["features"] if f.get("id") in set(map(int, frame_ids))
        ],
    }


@overload
def get_burst_id_geojson(
    burst_ids: Sequence[str] | None = None,
    as_geodataframe: Literal[True] = True,
) -> geopandas.GeoDataFrame: ...


@overload
def get_burst_id_geojson(
    burst_ids: Sequence[str] | None = None,
    as_geodataframe: Literal[False] = False,
) -> dict: ...


def get_burst_id_geojson(
    burst_ids: Sequence[str] | None = None,
    as_geodataframe: bool = False,
) -> GeojsonOrGdf:
    """Get the GeoJSON or GeoDataFrame for the burst_id geometries.

    Parameters
    ----------
    burst_ids : Sequence[str], optional
        Burst IDs to filter by. If None, returns all bursts.
    as_geodataframe : bool, default=False
        If True, returns a GeoDataFrame. If False, returns a GeoJSON dict.

    Returns
    -------
    dict or geopandas.GeoDataFrame
        Burst geometries as GeoJSON or GeoDataFrame.

    """
    if as_geodataframe:
        return get_burst_geodataframe(burst_ids)

    data = read_zipped_json(datasets.fetch_burst_id_geometries_simple())
    if not burst_ids:
        return data

    if isinstance(burst_ids, str):
        burst_ids = [burst_ids]
    # Manually filter for the case of no geopandas
    return {
        **data,
        "features": [
            f
            for f in data["features"]
            if f["properties"]["burst_id_jpl"] in set(burst_ids)
        ],
    }


def get_frame_geodataframe(
    frame_ids: Sequence[int | str] | None = None,
    json_file: PathOrStr | None = None,
) -> geopandas.GeoDataFrame:
    """Get frame geometries as a GeoDataFrame.

    Parameters
    ----------
    frame_ids : Sequence[int | str], optional
        Frame IDs to filter by. If None, returns all frames.
    json_file : PathOrStr, optional
        The path to the JSON file containing the frame geometries.
        If None, uses the default file from datasets.

    Returns
    -------
    geopandas.GeoDataFrame
        Frame geometries as a GeoDataFrame.

    """
    try:
        from pyogrio import read_dataframe
    except ImportError as e:
        msg = "geopandas and pyogrio are required for GeoDataFrame support"
        raise ImportError(msg) from e

    if json_file is None:
        json_file = datasets.fetch_frame_geometries_simple()

    gdf = read_dataframe(json_file, layer=None, fid_as_index=True, fids=frame_ids)
    gdf.index.name = "frame_id"
    return gdf


def get_burst_geodataframe(
    burst_ids: Sequence[str] | None = None,
    json_file: PathOrStr | None = None,
) -> geopandas.GeoDataFrame:
    """Get burst geometries as a GeoDataFrame.

    Parameters
    ----------
    burst_ids : Sequence[str], optional
        Burst IDs to filter by. If None, returns all bursts.
    json_file : PathOrStr, optional
        The path to the JSON file containing the burst geometries.
        If None, uses the default file from datasets.

    Returns
    -------
    geopandas.GeoDataFrame
        Burst geometries as a GeoDataFrame.

    """
    try:
        from pyogrio import read_dataframe
    except ImportError as e:
        msg = "geopandas and pyogrio are required for GeoDataFrame support"
        raise ImportError(msg) from e

    if json_file is None:
        json_file = datasets.fetch_burst_id_geometries_simple()

    gdf = read_dataframe(json_file, layer=None)
    gdf.index.name = "burst_id_jpl"

    if burst_ids:
        if isinstance(burst_ids, str):
            burst_ids = [burst_ids]
        return gdf[gdf.burst_id_jpl.isin(tuple(burst_ids))]

    return gdf


def get_frame_bbox(
    frame_id: int, json_file: PathOrStr | None = None
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
    frame_id: int, json_file: PathOrStr | None = None
) -> list[str]:
    """Get the burst IDs for one frame ID.

    Parameters
    ----------
    frame_id : int
        The ID of the frame to get the burst IDs for.
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
    burst_id: str, json_file: PathOrStr | None = None
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
    js = read_zipped_json(fsdecode(json_file))
    return js["data"][normalize_burst_id(burst_id)]


def get_frame_ids_for_burst(
    burst_id: str, json_file: PathOrStr | None = None
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


def get_intersecting_frames(bounds: Bbox) -> dict:
    """Get the frame IDs that intersect with the given bounds.

    Parameters
    ----------
    bounds : Bbox
        Bounding box to check for intersection

    Returns
    -------
    dict
        Returns a GeoJSON dict object.

    """
    try:
        import geopandas  # noqa: F401
        from shapely.geometry import box
    except ImportError as e:
        msg = "geopandas and shapely are required for this function"
        raise ImportError(msg) from e

    gdf = get_frame_geodataframe()
    frames = gdf[gdf.geometry.intersects(box(*bounds))]

    return frames.to_geo_dict()


def get_frame_orbit_pass(
    frame_ids: int | Sequence[int], json_file: PathOrStr | None = None
) -> list[OrbitPass]:
    """Return the orbit pass direction for `frame_id`.

    Parameters
    ----------
    frame_ids : int | Sequence[int]
        Frame ID (or multuple frame IDs) to query.
    as_geodataframe : bool, default=False
        If True, returns a GeoDataFrame. If False, returns a GeoJSON dict.
    json_file : PathOrStr, optional
        The path to the JSON file containing the frame geometries.
        If None, uses the default file from datasets.


    Returns
    -------
    OrbitPass
        The orbit direction for the requested frame_id.

    """
    frame_list = [frame_ids] if isinstance(frame_ids, int) else frame_ids
    features = get_frame_geojson(
        frame_list, as_geodataframe=False, json_file=json_file
    )["features"]
    if not features:
        msg = "No Frame {frame_id} found"
        raise ValueError(msg)
    return [OrbitPass(f["properties"]["orbit_pass"]) for f in features]
