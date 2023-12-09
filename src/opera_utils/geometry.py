from __future__ import annotations

import logging
import netrc
from functools import cache
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import asf_search as asf
from asf_search import ASFSearchResults

from opera_utils import get_burst_ids_for_frame, stitching
from opera_utils._types import PathOrStr
from opera_utils._utils import format_nc_filename, scratch_directory

logger = logging.getLogger(__name__)
DEFAULT_TIFF_OPTIONS = (
    "COMPRESS=DEFLATE",
    "ZLEVEL=4",
    "TILED=YES",
    "BLOCKXSIZE=128",
    "BLOCKYSIZE=128",
    "DISCARD_LSB=6",
    "PREDICTOR=2",
)


LAYER_NAME = Literal[
    "los_east", "los_north", "layover_shadow_mask", "local_incidence_angle"
]
# Layover shadow mask. 0=no layover, no shadow; 1=shadow; 2=layover; 3=shadow and layover.
DEFAULT_LAYERS = ["los_east", "los_north", "layover_shadow_mask"]
DEFAULT_STRIDES = {"x": 6, "y": 3}
LAYER_TO_NODATA = {
    "los_east": 0,
    "los_north": 0,
    "local_incidence_angle": 0,
    # layover_shadow_mask is Int8 with 127 meaning nodata
    "layover_shadow_mask": 127,
}


def create_geometry_files(
    frame_id: int | None = None,
    burst_ids: Sequence[str] | None = None,
    output_dir: PathOrStr = Path("."),
    layers: Sequence[LAYER_NAME] = DEFAULT_LAYERS,
    strides: Mapping[str, int] = DEFAULT_STRIDES,
    save_hdf5_files=True,
    max_download_jobs: int = 3,
) -> list[Path]:
    if frame_id is not None:
        burst_ids = get_burst_ids_for_frame(frame_id=frame_id)

    if not burst_ids:
        raise ValueError("Must provide frame_id or burst_ids")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    download_dir = output_path / "hdf5"
    output_files: list[Path] = []

    with scratch_directory(download_dir, delete=not save_hdf5_files) as sd:
        local_hdf5_files = download_static_layers(
            burst_ids=burst_ids, output_dir=sd, max_jobs=max_download_jobs
        )
        for layer in layers:
            gdal_strings = [
                format_nc_filename(f, ds_name=f"data/{layer}") for f in local_hdf5_files
            ]
            nodata = LAYER_TO_NODATA[layer]
            cur_outfile = output_dir / f"{layer}.tif"
            output_files.append(cur_outfile)
            logger.info
            stitching.merge_images(
                file_list=gdal_strings,
                outfile=cur_outfile,
                strides=strides,
                driver="GTIff",
                options=DEFAULT_TIFF_OPTIONS,
                resample_alg="nearest",
                in_nodata=nodata,
                out_nodata=nodata,
            )

    return output_files


@cache
def _search(burst_ids: Iterable[str]) -> ASFSearchResults:
    return asf.search(operaBurstID=burst_ids, processingLevel="CSLC-STATIC")


def _get_urls(results: ASFSearchResults, type_: Literal["https", "s3"]):
    if type_ == "https":
        return [r.properties["url"] for r in results]
    elif type_ == "s3":
        # TODO: go through .umm, find s3 url
        raise NotImplementedError()
    else:
        raise ValueError(f"type_ must be 'https' or 's3'. Got {type_}")
    # r.umm
    # 'RelatedUrls': [...
    #     {'URL': 's3://asf-cumulus-prod-opera-products/OPERA_L2_CSLC
    #    'Type': 'GET DATA VIA DIRECT ACCESS',
    #    'Description': 'This link provides direct download access vi
    #    'Format': 'HDF5'},


def download_static_layers(
    burst_ids: Iterable[str], output_dir: Path, max_jobs: int = 3
) -> list[Path]:
    results = _search(burst_ids=burst_ids)
    session = _get_auth_session()
    urls = _get_urls(results)
    asf.download_urls(urls=urls, path=output_dir, session=session, processes=max_jobs)
    return [output_dir / r.properties["fileName"] for r in results]


def _get_auth_session():
    username, _, password = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
    return asf.ASFSession().auth_with_creds(username, password)
