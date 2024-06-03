from __future__ import annotations

import itertools
import json
import logging
import re
import subprocess
import tempfile
from os import fspath
from pathlib import Path
from typing import Iterable, Pattern, Sequence, Union, overload

import h5py
from shapely import geometry, ops, wkt

from ._types import Filename, PathLikeT
from .constants import OPERA_BURST_RE, OPERA_DATASET_NAME, OPERA_IDENTIFICATION

logger = logging.getLogger(__name__)

__all__ = [
    "get_burst_id",
    "group_by_burst",
    "sort_by_burst_id",
    "filter_by_burst_id",
    "get_cslc_polygon",
    "get_union_polygon",
    "make_nodata_mask",
]


def get_burst_id(
    filename: Filename, burst_id_fmt: Union[str, Pattern[str]] = OPERA_BURST_RE
) -> str:
    """Extract the burst id from a filename.

    Matches either format of
        t087_185684_iw2 (which comes from COMPASS)
        T087-165495-IW3 (which is the official product naming scheme)

    Parameters
    ----------
    filename: Filename
        CSLC filename
    burst_id_fmt: str
        format of the burst id in the filename.
        Default is [`OPERA_BURST_RE`][opera_utils.OPERA_BURST_RE]

    Returns
    -------
    str
        burst id of the SLC acquisition, normalized to be in the format
            t087_185684_iw2
    """
    if not (m := re.search(burst_id_fmt, str(filename))):
        raise ValueError(f"Could not parse burst id from {filename}")
    burst_str = m.group()
    # Normalize
    return burst_str.lower().replace("-", "_")


@overload
def group_by_burst(
    file_list: Iterable[str],
    burst_id_fmt: Union[str, Pattern[str]] = OPERA_BURST_RE,
) -> dict[str, list[str]]: ...


@overload
def group_by_burst(
    file_list: Iterable[PathLikeT],
    burst_id_fmt: Union[str, Pattern[str]] = OPERA_BURST_RE,
) -> dict[str, list[PathLikeT]]: ...


def group_by_burst(file_list, burst_id_fmt=OPERA_BURST_RE):
    """Group Sentinel CSLC files by burst.

    Parameters
    ----------
    file_list: Iterable[Filename]
        list of paths of CSLC files
    burst_id_fmt: str
        format of the burst id in the filename.
        Default is [`OPERA_BURST_RE`][opera_utils.OPERA_BURST_RE]

    Returns
    -------
    dict
        key is the burst id of the SLC acquisition
        Value is a list of inputs which correspond to that burst:
        {
            't087_185678_iw2': ['inputs/t087_185678_iw2_20200101.h5',...,],
            't087_185678_iw3': ['inputs/t087_185678_iw3_20200101.h5',...,],
        }
    """
    if not file_list:
        return {}

    sorted_file_list = sort_by_burst_id(list(file_list), burst_id_fmt)
    # Now collapse into groups, sorted by the burst_id
    grouped_images = {
        burst_id: list(g)
        for burst_id, g in itertools.groupby(
            sorted_file_list, key=lambda x: get_burst_id(x)
        )
    }
    return grouped_images


@overload
def sort_by_burst_id(file_list: Iterable[str], burst_id_fmt) -> list[str]: ...


@overload
def sort_by_burst_id(
    file_list: Iterable[PathLikeT], burst_id_fmt
) -> list[PathLikeT]: ...


def sort_by_burst_id(file_list, burst_id_fmt):
    """Sort files/paths by the burst ID in their names.

    Parameters
    ----------
    file_list : Iterable[PathLikeT]
        list of paths of CSLC files
    burst_id_fmt : str
        format of the burst id in the filename.
        Default is [`OPERA_BURST_RE`][opera_utils.OPERA_BURST_RE]

    Returns
    -------
    list[Path] or list[str]
        sorted list of files
    """
    file_burst_tuples = sorted(
        [(f, get_burst_id(f, burst_id_fmt)) for f in file_list],
        # use the date or dates as the key
        key=lambda f_b_tuple: f_b_tuple[1],  # type: ignore
    )
    # Unpack the sorted pairs with new sorted values
    out_file_list = [f for f, _ in file_burst_tuples]
    return out_file_list


@overload
def filter_by_burst_id(
    files: Iterable[PathLikeT],
    burst_ids: str | Iterable[str],
) -> list[PathLikeT]: ...


@overload
def filter_by_burst_id(
    files: Iterable[str],
    burst_ids: str | Iterable[str],
) -> list[str]: ...


def filter_by_burst_id(files, burst_ids):
    """Keep only items from `files` which contain a burst ID in `burst_ids`.

    Searches only the burst ID in the base name, not the full path.

    Parameters
    ----------
    files : Iterable[PathLikeT] or Iterable[str]
        Iterable of files to filter
    burst_ids : str | Iterable[str]
        Burst ID/Iterable containing the of burst IDs to keep

    Returns
    -------
    list[PathLikeT] or list[str]
        filtered list of files
    """
    if isinstance(burst_ids, str):
        burst_ids = [burst_ids]

    burst_id_set = set(burst_ids)
    parsed_burst_ids = [get_burst_id(Path(f).name) for f in files]
    # Only search the burst ID in the name, not the full path
    return [f for (f, b) in zip(files, parsed_burst_ids) if b in burst_id_set]


def get_cslc_polygon(
    opera_file: Filename, buffer_degrees: float = 0.0
) -> Union[geometry.Polygon, None]:
    """Get the union of the bounding polygons of the given files.

    Parameters
    ----------
    opera_file : list[Filename]
        list of COMPASS SLC filenames.
    buffer_degrees : float, optional
        Buffer the polygons by this many degrees, by default 0.0
    """
    dset_name = f"{OPERA_IDENTIFICATION}/bounding_polygon"
    with h5py.File(opera_file) as hf:
        if dset_name not in hf:
            logger.debug(f"Could not find {dset_name} in {opera_file}")
            return None
        wkt_str = hf[dset_name][()].decode("utf-8")
    return wkt.loads(wkt_str).buffer(buffer_degrees)


def get_union_polygon(
    opera_file_list: Sequence[Filename], buffer_degrees: float = 0.0
) -> geometry.Polygon:
    """Get the union of the bounding polygons of the given files.

    Parameters
    ----------
    opera_file_list : list[Filename]
        list of COMPASS SLC filenames.
    buffer_degrees : float, optional
        Buffer the polygons by this many degrees, by default 0.0
    """
    polygons = [get_cslc_polygon(f, buffer_degrees) for f in opera_file_list]
    polygons = [p for p in polygons if p is not None]

    if len(polygons) == 0:
        raise ValueError("No polygons found in the given file list.")
    # Union all the polygons
    return ops.unary_union(polygons)


def make_nodata_mask(
    opera_file_list: Sequence[Filename],
    out_file: Filename,
    buffer_pixels: int = 0,
    overwrite: bool = False,
):
    """Make a boolean raster mask from the union of nodata polygons using GDAL.

    Parameters
    ----------
    opera_file_list : list[Filename]
        list of COMPASS SLC filenames.
    out_file : Filename
        Output filename.
    buffer_pixels : int, optional
        Number of pixels to buffer the union polygon by, by default 0.
        Note that buffering will *decrease* the numba of pixels marked as nodata.
        This is to be more conservative to not mask possible valid pixels.
    overwrite : bool, optional
        Overwrite the output file if it already exists, by default False
    """
    if Path(out_file).exists():
        if not overwrite:
            logger.debug(f"Skipping {out_file} since it already exists.")
            return
        else:
            logger.info(f"Overwriting {out_file} since overwrite=True.")
            Path(out_file).unlink()

    # Check these are the right format to get nodata polygons
    try:
        test_f = f"NETCDF:{opera_file_list[0]}:{OPERA_DATASET_NAME}"
        # convert pixels to degrees lat/lon
        gt = _get_raster_gt(test_f)
        # TODO: more robust way to get the pixel size... this is a hack
        # maybe just use pyproj to warp lat/lon to meters and back?
        dx_meters = gt[1]
        dx_degrees = dx_meters / 111000
        buffer_degrees = buffer_pixels * dx_degrees
    except RuntimeError:
        raise ValueError(f"Unable to open {test_f}")

    # Get the union of all the polygons and convert to a temp geojson
    union_poly = get_union_polygon(opera_file_list, buffer_degrees=buffer_degrees)
    # convert shapely polygon to geojson

    # Make a dummy raster from the first file with all 0s
    # This will get filled in with the polygon rasterization
    cmd = (
        f"gdal_calc.py --quiet --outfile {out_file} --type Byte  -A"
        f" NETCDF:{opera_file_list[0]}:{OPERA_DATASET_NAME} --calc 'numpy.nan_to_num(A)"
        " * 0' --creation-option COMPRESS=LZW --creation-option TILED=YES"
        " --creation-option BLOCKXSIZE=256 --creation-option BLOCKYSIZE=256"
    )
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_vector_file = Path(tmpdir) / "temp.geojson"
        with open(temp_vector_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "geometry": geometry.mapping(union_poly),
                        "properties": {"id": 1},
                    }
                )
            )

        # Now burn in the union of all polygons
        cmd = f"gdal_rasterize -q -burn 1 {temp_vector_file} {out_file}"
        logger.info(cmd)
        subprocess.check_call(cmd, shell=True)


def _get_raster_gt(filename: Filename) -> list[float]:
    """Get the geotransform from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.

    Returns
    -------
    List[float]
        6 floats representing a GDAL Geotransform.
    """
    from osgeo import gdal

    ds = gdal.Open(fspath(filename))
    gt = ds.GetGeoTransform()
    return gt
