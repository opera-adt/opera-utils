#!/usr/bin/env python
"""Process a single DISP-S1 frame to compute a line-of-sight velocity.

Examples
--------
# defaults: https URLs, 4 download workers, outputs under ./work
python disp_process_frame.py --frame-id 11116 --end-datetime 2020-01-01

"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import rasterio as rio
import requests

from opera_utils.disp import rebase_reference, search
from opera_utils.disp._rebase import NaNPolicy
from opera_utils.disp._search import UrlType


def _download(url: str, out_dir: Path) -> Path:
    """Download *url* into *out_dir* if needed and return the local path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / Path(url).name
    if dest.exists():
        return dest

    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=2**20):
            f.write(chunk)
    return dest


def process_frame(
    frame_id: int,
    work_dir: Path = Path("work"),
    url_type: UrlType = UrlType.HTTPS,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    apply_solid_earth_corrections: bool = True,
    apply_ionospheric_corrections: bool = True,
    apply_mask: bool = True,
    nan_policy: str | NaNPolicy = NaNPolicy.propagate,
    reference_point: tuple[int, int] | None = None,
    num_workers: int = 4,
) -> None:
    """Search, download, rebase one DISP-S1 frame, then create a velocity map.

    Parameters
    ----------
    frame_id
        DISP-S1 frame ID to process.
    work_dir
        Where outputs (and temp `.nc` files) are written.
    url_type : str, choices = ["https", "s3"]
        Whether to use HTTPS or S3 URLs when downloading from ASF.
    start_datetime
        Start datetime for search.
    end_datetime
        End datetime for search.
    apply_solid_earth_corrections : bool, optional
        Whether to apply solid earth corrections to the data, by default True
    apply_ionospheric_corrections : bool, optional
        Whether to apply ionospheric corrections to the data, by default True
    apply_mask : bool, optional
        Whether to apply the recommended mask to the data, by default True
    nan_policy : choices = ["propagate", "omit"]
        Whether to propagate or omit (zero out) NaNs in the data.
        By default "propagate", which means any ministack, or any "reference crossover"
        product, with nan at a pixel causes all subsequent data to be nan.
        If "omit", then any nan causes the pixel to be zeroed out, which is
        equivalent to assuming that 0 displacement occurred during that time.
    reference_point : tuple[int, int] | None, optional
        The (row, column) of the reference point to use when rebasing /displacement.
        If None, finds a point with the highest harmonic mean of temporal coherence.
        Default is None.
    num_workers
        Number of parallel download and reference-rebasing workers.

    """
    work_dir = work_dir.resolve()
    nc_dir = work_dir / "ncs"
    aligned_dir = work_dir / "aligned"

    products = search(
        frame_id=frame_id,
        url_type=UrlType(url_type),
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )
    if not products:
        msg = f"No granules found for frame {frame_id}"
        raise RuntimeError(msg)
    urls = [p.filename for p in products]

    # 2. download
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        nc_paths = list(pool.map(lambda u: _download(u, nc_dir), urls))

    # 3. reference alignment
    aligned_dir.mkdir(parents=True, exist_ok=True)
    rebase_reference.main(
        nc_files=nc_paths,
        output_dir=aligned_dir,
        apply_solid_earth_corrections=apply_solid_earth_corrections,
        apply_ionospheric_corrections=apply_ionospheric_corrections,
        apply_mask=apply_mask,
        nan_policy=nan_policy,
        reference_point=reference_point,
        num_workers=num_workers,
    )

    try:
        from dolphin import ReferencePoint, timeseries, utils
    except ImportError as e:
        msg = "dolphin is required for velocity creation."
        raise ImportError(msg) from e

    utils.disable_gpu()
    disp_files = sorted(aligned_dir.glob("displacement*.tif"))
    if not disp_files:
        msg = "No displacement TIFFs produced by rebase step."
        raise RuntimeError(msg)
    vel_file = aligned_dir / "velocity.tif"
    mask_files = sorted(aligned_dir.glob("recommended_mask*20*.tif"))
    timeseries.create_velocity(
        unw_file_list=disp_files,
        output_file=vel_file,
        reference=ReferencePoint(*reference_point) if reference_point else None,
        cor_file_list=mask_files,
        num_threads=num_workers,
    )
    # Copy with more compression options
    temp_file = aligned_dir / "velocity_temp.tif"
    options = {
        "nbits": "16",
        "predictor": "2",
        "tiled": "256x256",
        "compress": "deflate",
    }
    with (
        rio.open(vel_file, "r") as src,
        rio.open(temp_file, "w", **(src.profile | options)) as dst,
    ):
        dst.write(src.read(1), 1)
    temp_file.replace(vel_file)


if __name__ == "__main__":
    import tyro

    tyro.cli(process_frame)
