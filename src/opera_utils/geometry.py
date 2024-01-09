from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from opera_utils import get_burst_ids_for_frame, stitching
from opera_utils._types import PathOrStr
from opera_utils._utils import format_nc_filename, scratch_directory
from opera_utils.constants import EXTRA_COMPRESSED_TIFF_OPTIONS
from opera_utils.download import download_cslc_static_layers

logger = logging.getLogger(__name__)


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
        _description_, by default False
    layers : Sequence[Layer  |  str], optional
        _description_, by default DEFAULT_LAYERS
    strides : Mapping[str, int], optional
        _description_, by default DEFAULT_STRIDES
    max_download_jobs : int, optional
        _description_, by default 3

    Returns
    -------
    list[Path]
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    if frame_id is not None:
        burst_ids = get_burst_ids_for_frame(frame_id=frame_id)

    if not burst_ids:
        raise ValueError("Must provide frame_id or burst_ids")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    if download_dir is None:
        download_dir = output_path / "hdf5"
    output_files: list[Path] = []

    with scratch_directory(download_dir, delete=not save_hdf5_files) as sd:
        local_hdf5_files = download_cslc_static_layers(
            burst_ids=burst_ids, output_dir=sd, max_jobs=max_download_jobs
        )
        for layer in layers:
            layer_enum = Layer(layer)
            name = layer_enum.value
            gdal_strings = [
                format_nc_filename(f, ds_name=f"data/{name}") for f in local_hdf5_files
            ]
            nodata = LAYER_TO_NODATA[Layer(layer)]
            cur_outfile = output_path / f"{name}.tif"
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
