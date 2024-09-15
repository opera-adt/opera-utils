from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import rasterio as rio
from osgeo import gdal

from opera_utils import get_burst_ids_for_frame, stitching
from opera_utils._types import PathOrStr
from opera_utils._utils import format_nc_filename, scratch_directory
from opera_utils.download import download_cslc_static_layers

logger = logging.getLogger(__name__)

EXTRA_COMPRESSED_TIFF_OPTIONS = (
    "COMPRESS=DEFLATE",
    "ZLEVEL=4",
    "TILED=YES",
    "BLOCKXSIZE=128",
    "BLOCKYSIZE=128",
    # Note: we're dropping mantissa bits before we do not
    # need prevision for LOS rasters (or incidence)
    "NBITS=16",
    "PREDICTOR=2",
)


class Layer(Enum):
    """Names of available datasets in CSLC static layers HDF5 files."""

    LOS_EAST = "los_east"
    LOS_NORTH = "los_north"
    LAYOVER_SHADOW_MASK = "layover_shadow_mask"
    LOCAL_INCIDENCE_ANGLE = "local_incidence_angle"


# Layover shadow mask. 0=no layover, no shadow; 1=shadow; 2=layover; 3=shadow and layover.
DEFAULT_LAYERS = list(Layer)[:3]  # Skip the local incidence, much less compressible
DEFAULT_STRIDES = {"x": 6, "y": 3}
LAYER_TO_NODATA = {
    Layer.LOS_EAST: 0,
    Layer.LOS_NORTH: 0,
    Layer.LOCAL_INCIDENCE_ANGLE: 0,
    # layover_shadow_mask is Int8 with 127 meaning nodata
    Layer.LAYOVER_SHADOW_MASK: 127,
}


def create_geometry_files(
    *,
    frame_id: int | None = None,
    burst_ids: Sequence[str] | None = None,
    output_dir: PathOrStr = Path("."),
    download_dir: PathOrStr | None = None,
    save_hdf5_files: bool = False,
    layers: Sequence[Layer | str] = DEFAULT_LAYERS,
    strides: Mapping[str, int] = DEFAULT_STRIDES,
    max_download_jobs: int = 3,
) -> list[Path]:
    """Create merged geometry files for a frame of list of burst IDs.

    Parameters
    ----------
    frame_id : int | None, optional
        DISP frame ID to create, by default None
    burst_ids : Sequence[str] | None, optional
        Alternative to `frame_id`, manually specify CSLC burst IDs.
    output_dir : PathOrStr, optional
        Directory to store output geotiffs, by default Path(".")
    download_dir : PathOrStr | None, optional
        Directory to save files, by default None
    save_hdf5_files : bool, optional
        Flag to retain HDF5 files. Defaults to False.
    layers : Sequence[Layer | str], optional
        Layers to process. Defaults to DEFAULT_LAYERS.
    strides : Mapping[str, int], optional
        Stride values for merging images. Defaults to DEFAULT_STRIDES.
    max_download_jobs : int, optional
        Maximum number of download jobs to run in parallel. Defaults to 3.

    Returns
    -------
    list[Path]
        List of output files with paths.

    Raises
    ------
    ValueError
        If neither `frame_id` nor `burst_ids` are provided.
    """
    if frame_id is not None:
        burst_ids = get_burst_ids_for_frame(frame_id=frame_id)

    if not burst_ids:
        raise ValueError("Must provide frame_id or burst_ids")
    logger.debug("Using burst IDs: %s", burst_ids)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    if download_dir is None:
        download_dir = output_path / "hdf5"

    with scratch_directory(download_dir, delete=not save_hdf5_files) as sd:
        local_hdf5_files = download_cslc_static_layers(
            burst_ids=burst_ids, output_dir=sd, max_jobs=max_download_jobs
        )
        stitch_geometry_layers(
            local_hdf5_files=local_hdf5_files,
            layers=layers,
            strides=strides,
            output_dir=output_path,
        )

    _make_los_up(output_path)
    output_files = _make_3band_los(output_path)
    return output_files


def _make_los_up(output_path: Path):
    """Create los_up.tif from east/north."""
    los_east_path = output_path / "los_east.tif"
    los_north_path = output_path / "los_north.tif"
    los_up_path = output_path / "los_up.tif"

    with rio.open(los_east_path) as src_east, rio.open(los_north_path) as src_north:
        profile = src_east.profile
        los_east = src_east.read(1)
        los_north = src_north.read(1)

        los_up = np.sqrt(1 - los_east**2 - los_north**2)

        profile.update(
            dtype=rio.float32,
            count=1,
            compress="deflate",
            zlevel=4,
            tiled=True,
            blockxsize=128,
            blockysize=128,
            predictor=2,
        )

        with rio.open(los_up_path, "w", **profile) as dst:
            dst.write(los_up.astype(rio.float32), 1)


def _make_3band_los(output_path: Path):
    """Combine the 3 TIFFs into one, 3-band TIFF."""
    los_east_path = output_path / "los_east.tif"
    los_north_path = output_path / "los_north.tif"
    los_up_path = output_path / "los_up.tif"
    combined_los_path = output_path / "los_combined.tif"

    with (
        rio.open(los_east_path) as src_east,
        rio.open(los_north_path) as src_north,
        rio.open(los_up_path) as src_up,
    ):
        profile = src_east.profile
        profile.update(
            count=3,
            compress="deflate",
            zlevel=4,
            tiled=True,
            blockxsize=128,
            blockysize=128,
            predictor=2,
            interleave="band",
        )
        desc_base = "{} component of line of sight unit vector (ground to satellite)"
        with rio.open(combined_los_path, "w", **profile) as dst:
            dst.write(src_east.read(1), 1)
            dst.set_band_description(1, desc_base.format("East"))

            dst.write(src_north.read(1), 2)
            dst.set_band_description(2, desc_base.format("North"))

            dst.write(src_up.read(1), 3)
            dst.set_band_description(3, desc_base.format("Up"))

    # Delete old single-band TIFFs
    Path(los_east_path).unlink()
    Path(los_north_path).unlink()
    Path(los_up_path).unlink()

    # Create VRT files for each band
    for band, name in enumerate(["los_east", "los_north", "los_up"], start=1):
        vrt_path = output_path / f"{name}.vrt"
        gdal.BuildVRT(str(vrt_path), [str(combined_los_path)], bandList=[band])

    return [combined_los_path] + [
        output_path / f"{name}.vrt" for name in ["los_east", "los_north", "los_up"]
    ]


def stitch_geometry_layers(
    local_hdf5_files: list[Path],
    layers: Sequence[Layer | str] = DEFAULT_LAYERS,
    strides: Mapping[str, int] = DEFAULT_STRIDES,
    output_dir: PathOrStr = Path("."),
) -> list[Path]:
    """Stitch geometry layers from downloaded HDF5 files.

    Parameters
    ----------
    local_hdf5_files : list[Path]
        List of paths to the downloaded HDF5 files.
    layers : Sequence[Layer | str]
        Layers to be processed.
    strides : Mapping[str, int]
        Stride values for merging images.
    output_dir : PathOrStr
        Directory to store output Geotiffs.

    Returns
    -------
    list[Path]
        List of output files with paths.
    """
    output_files: list[Path] = []

    for layer in layers:
        layer_enum = Layer(layer)
        name = layer_enum.value
        gdal_strings = [
            format_nc_filename(f, ds_name=f"data/{name}") for f in local_hdf5_files
        ]
        nodata = LAYER_TO_NODATA[Layer(layer)]
        cur_outfile = Path(output_dir) / f"{name}.tif"
        output_files.append(cur_outfile)
        logger.info(f"Merging images for {name}")
        stitching.merge_images(
            file_list=gdal_strings,
            outfile=cur_outfile,
            strides=strides,
            driver="GTIff",
            options=EXTRA_COMPRESSED_TIFF_OPTIONS,
            resample_alg="nearest",
            in_nodata=nodata,
            out_nodata=nodata,
        )
    return output_files
