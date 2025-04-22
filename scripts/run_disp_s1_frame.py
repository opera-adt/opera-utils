#!/usr/bin/env python
"""Process a single DISP-S1 frame to compute a line-of-sight velocity.

Examples
--------
# defaults: https URLs, 4 download workers, outputs under ./work
python disp_process_frame.py --frame-id 11116 --start-datetime 2018-01-01 --end-datetime 2020-01-01
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests
import tyro

from opera_utils.disp import rebase_reference, search
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
    num_workers: int = 4,
    apply_corrections: bool = True,
) -> None:
    """Search, download, align one DISP-S1 frame, then create a velocity map.

    Parameters
    ----------
    frame_id
        DISP-S1 frame ID to process.
    work_dir
        Where outputs (and temp `.nc` files) are written.
    url_type
        "https" (default) or "s3".
    start_datetime
        Start datetime for search.
    end_datetime
        End datetime for search.
    num_workers
        Number of parallel download and reference-rebasing workers.
    apply_corrections
        Pass-through flag for `rebase_reference`.
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
        raise RuntimeError(f"No granules found for frame {frame_id}")
    urls = [p.filename for p in products]

    # 2. download
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        nc_paths = list(pool.map(lambda u: _download(u, nc_dir), urls))

    # 3. reference alignment
    aligned_dir.mkdir(parents=True, exist_ok=True)
    rebase_reference.main(
        nc_files=nc_paths,
        output_dir=aligned_dir,
        apply_corrections=apply_corrections,
        reference_point=None,
        num_workers=num_workers,
    )

    try:
        from dolphin import timeseries, utils
    except ImportError as e:
        raise ImportError("dolphin is required for velocity creation.") from e

    utils.disable_gpu()
    disp_files = sorted(aligned_dir.glob("displacement*.tif"))
    if not disp_files:
        raise RuntimeError("No displacement TIFFs produced by rebase step.")
    vel_file = aligned_dir / "velocity.tif"
    timeseries.create_velocity(
        unw_file_list=disp_files,
        output_file=vel_file,
        reference=None,
        num_threads=num_workers,
    )
    print(f"✓ Velocity written to {vel_file}")


if __name__ == "__main__":
    tyro.cli(process_frame)
