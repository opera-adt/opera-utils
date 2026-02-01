"""Download and subset NISAR GSLC products."""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import requests
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from opera_utils.constants import (
    NISAR_FREQUENCIES,
    NISAR_GSLC_GRIDS,
    NISAR_GSLC_IDENTIFICATION,
    NISAR_GSLC_METADATA,
    NISAR_POLARIZATIONS,
)
from opera_utils.credentials import get_earthdata_username_password

from ._product import GslcProduct, UrlType
from ._remote import open_file
from ._search import search

logger = logging.getLogger("opera_utils")

__all__ = ["run_download", "process_file"]


def process_file(
    url: str,
    rows: slice | None,
    cols: slice | None,
    output_dir: Path,
    frequency: str = "A",
    polarizations: list[str] | None = None,
    session: requests.Session | None = None,
) -> Path:
    """Download and subset a single NISAR GSLC product.

    Parameters
    ----------
    url : str
        URL of the GSLC product to download.
    rows : slice | None
        Rows of the product to subset.
    cols : slice | None
        Columns of the product to subset.
    output_dir : Path
        Directory to save the downloaded and subsetted product to.
    frequency : str
        Frequency band to extract ("A" or "B"). Default is "A".
    polarizations : list[str] | None
        List of polarizations to extract (e.g. ["HH", "VV"]).
        If None, extracts all available polarizations.
    session : requests.Session | None
        Authenticated Session to use for downloading the product.

    Returns
    -------
    Path
        Path to the downloaded and subsetted product.

    """
    outname = f"{output_dir}/{Path(url).name}"
    outpath = Path(outname)
    if outpath.exists():
        logger.info(f"Skipped (exists): {outname}")
        return outpath

    if Path(url).exists():
        # For local files, directly extract subset
        _extract_subset(
            url, outpath=outpath, rows=rows, cols=cols,
            frequency=frequency, polarizations=polarizations
        )
    elif url.startswith("s3://"):
        # For S3 urls, use fsspec
        with open_file(url) as in_f:
            _extract_subset(
                in_f, outpath=outpath, rows=rows, cols=cols,
                frequency=frequency, polarizations=polarizations
            )
    else:
        # HTTPS: download to temp file first for better performance
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
            temp_path = Path(tf.name)
            if session is None:
                session = requests.Session()
                username, password = get_earthdata_username_password()
                session.auth = (username, password)
            logger.info(f"Downloading {url}...")
            response = session.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with (
                open(temp_path, "wb") as out_f,
                tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    out_f.write(chunk)
                    pbar.update(len(chunk))

            _extract_subset(
                temp_path, outpath=outpath, rows=rows, cols=cols,
                frequency=frequency, polarizations=polarizations
            )
            temp_path.unlink()  # Clean up temp file

    logger.debug(f"Done: {outname}")
    return outpath


def _extract_subset(
    input_obj: Path | str,
    outpath: Path | str,
    rows: slice | None,
    cols: slice | None,
    frequency: str = "A",
    polarizations: list[str] | None = None,
    chunks: tuple[int, int] = (256, 256),
) -> None:
    """Extract a spatial subset from a GSLC HDF5 file.

    Parameters
    ----------
    input_obj : Path | str
        Input HDF5 file path or file-like object.
    outpath : Path | str
        Output HDF5 file path.
    rows : slice | None
        Row slice for subsetting. None means all rows.
    cols : slice | None
        Column slice for subsetting. None means all cols.
    frequency : str
        Frequency band to extract ("A" or "B"). Default is "A".
    polarizations : list[str] | None
        List of polarizations to extract. None means all available.
    chunks : tuple[int, int]
        Chunk size for output datasets.

    """
    row_slice = rows if rows is not None else slice(None)
    col_slice = cols if cols is not None else slice(None)

    with h5py.File(input_obj, "r") as src, h5py.File(outpath, "w") as dst:
        # Copy identification metadata
        if NISAR_GSLC_IDENTIFICATION in src:
            src.copy(NISAR_GSLC_IDENTIFICATION, dst, name=NISAR_GSLC_IDENTIFICATION)

        # Copy metadata if present
        if NISAR_GSLC_METADATA in src:
            src.copy(NISAR_GSLC_METADATA, dst, name=NISAR_GSLC_METADATA)

        # Get available polarizations for the frequency
        freq_path = f"{NISAR_GSLC_GRIDS}/frequency{frequency}"
        if freq_path not in src:
            msg = f"Frequency {frequency} not found in {input_obj}"
            raise ValueError(msg)

        available_pols = [
            name for name in src[freq_path].keys()
            if name in NISAR_POLARIZATIONS
        ]

        if polarizations is None:
            pols_to_extract = available_pols
        else:
            pols_to_extract = [p for p in polarizations if p in available_pols]
            missing = set(polarizations) - set(available_pols)
            if missing:
                logger.warning(f"Polarizations not found: {missing}")

        if not pols_to_extract:
            msg = f"No polarizations to extract from {input_obj}"
            raise ValueError(msg)

        # Create the grids structure
        dst.create_group(NISAR_GSLC_GRIDS)
        dst_freq_group = dst.create_group(freq_path)

        # Copy coordinate datasets if present (x, y coordinates)
        for coord_name in ["xCoordinates", "yCoordinates", "xCoordinateSpacing", "yCoordinateSpacing"]:
            if coord_name in src[freq_path]:
                coord_data = src[freq_path][coord_name]
                if coord_name in ["xCoordinates", "yCoordinates"]:
                    # These need to be subsetted
                    if coord_name == "xCoordinates":
                        subset_data = coord_data[col_slice]
                    else:
                        subset_data = coord_data[row_slice]
                    dst_freq_group.create_dataset(coord_name, data=subset_data)
                else:
                    # Scalar values, just copy
                    dst_freq_group.create_dataset(coord_name, data=coord_data[()])

        # Copy projection info if present
        for proj_name in ["projection", "epsg"]:
            if proj_name in src[freq_path]:
                proj_data = src[freq_path][proj_name]
                if isinstance(proj_data, h5py.Dataset):
                    dst_freq_group.create_dataset(proj_name, data=proj_data[()])

        # Extract each polarization
        for pol in pols_to_extract:
            pol_path = f"{freq_path}/{pol}"
            src_dset = src[pol_path]

            # Get the subset
            subset_data = src_dset[row_slice, col_slice]

            # Determine chunk size (don't exceed data dimensions)
            out_shape = subset_data.shape
            actual_chunks = (
                min(chunks[0], out_shape[0]),
                min(chunks[1], out_shape[1]),
            )

            # Create output dataset with compression
            dst.create_dataset(
                pol_path,
                data=subset_data,
                chunks=actual_chunks,
                compression="gzip",
                compression_opts=4,
            )

            # Copy attributes
            for attr_name, attr_val in src_dset.attrs.items():
                dst[pol_path].attrs[attr_name] = attr_val

            logger.debug(f"Extracted {pol}: {src_dset.shape} -> {out_shape}")

        # Store subset metadata
        dst.attrs["subset_rows"] = str(row_slice)
        dst.attrs["subset_cols"] = str(col_slice)
        dst.attrs["source_file"] = str(input_obj)


def _run_file(
    args: tuple[str, slice | None, slice | None, Path, str, list[str] | None, requests.Session | None],
) -> Path:
    """Worker function for parallel processing."""
    url, rows, cols, output_dir, frequency, polarizations, session = args
    return process_file(url, rows, cols, output_dir, frequency, polarizations, session)


def run_download(
    track_frame: str | None = None,
    cycle_number: int | None = None,
    relative_orbit_number: int | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    rows: tuple[int, int] | None = None,
    cols: tuple[int, int] | None = None,
    frequency: str = "A",
    polarizations: list[str] | None = None,
    url_type: UrlType = UrlType.HTTPS,
    short_name: str = "NISAR_L2_GSLC_BETA_V1",
    num_workers: int = 4,
    output_dir: Path = Path("./gslc_subsets"),
) -> list[Path]:
    """Download and subset NISAR GSLC products.

    This function downloads and subsets NISAR GSLC HDF5 products from
    Earthdata and saves them to the local file system.

    Optionally, the files can be cropped using the `rows` and `cols` parameters.

    Parameters
    ----------
    track_frame : str | None
        Track/frame identifier to search for (e.g. "004_076_A_022").
    cycle_number : int | None
        Cycle number to filter by.
    relative_orbit_number : int | None
        Relative orbit number to filter by.
    start_datetime : datetime | None
        Start datetime of the product search.
    end_datetime : datetime | None
        End datetime of the product search.
    rows : tuple[int, int] | None
        Row range to subset as (start, stop). None means all rows.
    cols : tuple[int, int] | None
        Column range to subset as (start, stop). None means all cols.
    frequency : str
        Frequency band to extract ("A" or "B"). Default is "A".
    polarizations : list[str] | None
        List of polarizations to extract (e.g. ["HH", "VV"]).
        If None, extracts all available polarizations.
    url_type : UrlType
        Type of URL to use for the product.
        Default is HTTPS.
    short_name : str
        CMR collection short name.
        Default is "NISAR_L2_GSLC_BETA_V1" (beta products).
    num_workers : int
        Number of workers to use for downloading and subsetting.
        Default is 4.
    output_dir : Path
        Directory to save the downloaded and subsetted products to.
        Default is "./gslc_subsets".

    Returns
    -------
    list[Path]
        List of paths to the downloaded and subsetted products.

    """
    if num_workers == 1:
        session = requests.session()
        username, password = get_earthdata_username_password()
        session.auth = (username, password)
    else:
        session = None

    # Search for products
    results = search(
        track_frame=track_frame,
        cycle_number=cycle_number,
        relative_orbit_number=relative_orbit_number,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        url_type=url_type,
        short_name=short_name,
    )
    n_urls = len(results)
    logger.info(f"Found {n_urls} GSLC products")

    if n_urls == 0:
        logger.warning("No products found matching search criteria")
        return []

    # Convert row/col tuples to slices
    row_slice = slice(*rows) if rows is not None else None
    col_slice = slice(*cols) if cols is not None else None

    logger.info(f"Subsetting to rows: {row_slice}, cols: {col_slice}")

    # Build job arguments
    jobs = [
        (str(product.filename), row_slice, col_slice, output_dir, frequency, polarizations, session)
        for product in results
    ]
    output_dir.mkdir(exist_ok=True, parents=True)

    if num_workers == 1:
        return [_run_file(job) for job in tqdm(jobs, desc="Processing")]

    return process_map(_run_file, jobs, max_workers=num_workers)
