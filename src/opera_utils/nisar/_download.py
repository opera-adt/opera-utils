"""Download and subset NISAR GSLC products."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import h5py
from shapely import from_wkt
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from opera_utils.constants import (
    NISAR_GSLC_GRIDS,
    NISAR_POLARIZATIONS,
)

from ._product import GslcProduct, UrlType
from ._remote import open_h5
from ._search import search

logger = logging.getLogger("opera_utils")

__all__ = ["process_file", "run_download"]


def process_file(
    url: str,
    rows: slice | None,
    cols: slice | None,
    output_dir: Path,
    frequency: str = "A",
    polarizations: list[str] | None = None,
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
            url,
            outpath=outpath,
            rows=rows,
            cols=cols,
            frequency=frequency,
            polarizations=polarizations,
        )
    else:
        # For remote urls (S3 or HTTPS), use open_h5 with cloud-optimized settings
        with open_h5(url) as src:
            _extract_subset_from_h5(
                src,
                outpath=outpath,
                rows=rows,
                cols=cols,
                frequency=frequency,
                polarizations=polarizations,
            )

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
    """Extract a spatial subset from a local GSLC HDF5 file.

    Parameters
    ----------
    input_obj : Path | str
        Input HDF5 file path.
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
    with h5py.File(input_obj, "r") as src:
        _extract_subset_from_h5(
            src, outpath, rows, cols, frequency, polarizations, chunks, str(input_obj)
        )


def _extract_subset_from_h5(
    src: h5py.File,
    outpath: Path | str,
    rows: slice | None,
    cols: slice | None,
    frequency: str = "A",
    polarizations: list[str] | None = None,
    chunks: tuple[int, int] = (256, 256),
    source_name: str = "remote",
) -> None:
    """Extract a spatial subset from an open HDF5 file handle.

    Parameters
    ----------
    src : h5py.File
        Open HDF5 file handle.
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
    source_name : str
        Name of source file for metadata.

    """
    row_slice = rows if rows is not None else slice(None)
    col_slice = cols if cols is not None else slice(None)

    with h5py.File(outpath, "w") as dst:
        # Note: We skip copying identification/metadata groups for remote files
        # because h5py's copy() doesn't work well with fsspec byte-range access.
        # The essential georeferencing info is in the frequency group.

        # Get available polarizations for the frequency
        freq_path = f"{NISAR_GSLC_GRIDS}/frequency{frequency}"
        if freq_path not in src:
            msg = f"Frequency {frequency} not found in {source_name}"
            raise ValueError(msg)

        available_pols = [
            name for name in src[freq_path] if name in NISAR_POLARIZATIONS
        ]

        if polarizations is None:
            pols_to_extract = available_pols
        else:
            pols_to_extract = [p for p in polarizations if p in available_pols]
            missing = set(polarizations) - set(available_pols)
            if missing:
                logger.warning(f"Polarizations not found: {missing}")

        if not pols_to_extract:
            msg = f"No polarizations to extract from {source_name}"
            raise ValueError(msg)

        # Create the grids structure
        dst.create_group(NISAR_GSLC_GRIDS)
        dst_freq_group = dst.create_group(freq_path)

        # Copy coordinate datasets if present (x, y coordinates)
        for coord_name in [
            "xCoordinates",
            "yCoordinates",
            "xCoordinateSpacing",
            "yCoordinateSpacing",
        ]:
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
        dst.attrs["source_file"] = source_name


def _get_rowcol_slice(
    product: GslcProduct,
    bbox: tuple[float, float, float, float] | None,
    frequency: str = "A",
) -> tuple[slice | None, slice | None]:
    """Convert a bounding box to row/col slices.

    Parameters
    ----------
    product : GslcProduct
        A GSLC product to use for coordinate conversion.
    bbox : tuple[float, float, float, float] | None
        Bounding box as (west, south, east, north) in degrees lon/lat.
    frequency : str
        Frequency band. Default is "A".

    Returns
    -------
    tuple[slice | None, slice | None]
        Row and column slices for subsetting.

    """
    if bbox is None:
        return None, None

    lon_left, lat_bottom, lon_right, lat_top = bbox

    # Need to open the file to get coordinate info
    with open_h5(str(product.filename)) as h5f:
        row_start, col_start = product.lonlat_to_rowcol(
            h5f, lon_left, lat_top, frequency
        )
        row_stop, col_stop = product.lonlat_to_rowcol(
            h5f, lon_right, lat_bottom, frequency
        )

    # Ensure start < stop
    if row_start > row_stop:
        row_start, row_stop = row_stop, row_start
    if col_start > col_stop:
        col_start, col_stop = col_stop, col_start

    return slice(row_start, row_stop), slice(col_start, col_stop)


def _run_file(
    args: tuple[
        str,
        slice | None,
        slice | None,
        Path,
        str,
        list[str] | None,
    ],
) -> Path:
    """Worker function for parallel processing."""
    url, rows, cols, output_dir, frequency, polarizations = args
    return process_file(url, rows, cols, output_dir, frequency, polarizations)


def run_download(
    bbox: tuple[float, float, float, float] | None = None,
    wkt: str | None = None,
    track_frame: str | None = None,
    track_frame_number: int | None = None,
    orbit_direction: str | None = None,
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

    The `bbox` parameter serves dual purposes: it filters the CMR search to find
    products that intersect the bounding box, AND subsets the downloaded files
    to that region.

    Parameters
    ----------
    bbox : tuple[float, float, float, float] | None
        Bounding box as (west, south, east, north) in degrees lon/lat.
        Used for both CMR spatial search and subsetting downloaded files.
        Cannot be used with `rows`/`cols`.
    wkt : str | None
        Well-known text representation of the region of interest.
        The bounding box of the geometry will be used.
        Cannot be used with `bbox` or `rows`/`cols`.
    track_frame : str | None
        Track/frame identifier (e.g. "004_076_A_022"). Includes cycle number,
        so only useful for finding a specific granule.
    track_frame_number : int | None
        Track frame number (e.g., 8). Stays constant for repeat passes.
    orbit_direction : str | None
        Orbit direction: "A" for ascending, "D" for descending.
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
        Cannot be used with `bbox`/`wkt`.
    cols : tuple[int, int] | None
        Column range to subset as (start, stop). None means all cols.
        Cannot be used with `bbox`/`wkt`.
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

    Examples
    --------
    Download by bounding box (searches CMR and subsets to that region):

    >>> run_download(  # doctest: +SKIP
    ...     bbox=(40.62, 13.56, 40.72, 13.64), polarizations=["HH"]
    ... )

    """
    # Validate mutually exclusive subsetting options
    if (bbox is not None or wkt is not None) and (rows is not None or cols is not None):
        msg = "Cannot specify both bbox/wkt and rows/cols. Use one or the other."
        raise ValueError(msg)
    if bbox is not None and wkt is not None:
        msg = "Cannot specify both bbox and wkt."
        raise ValueError(msg)

    # Convert WKT to bbox if provided
    if wkt is not None:
        poly = from_wkt(wkt)
        bbox = poly.bounds  # (minx, miny, maxx, maxy) = (west, south, east, north)

    # Search for products (bbox is used for both CMR search and subsetting)
    results = search(
        bbox=bbox,
        track_frame=track_frame,
        track_frame_number=track_frame_number,
        orbit_direction=orbit_direction,
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

    # Convert bbox or row/col tuples to slices
    if bbox is not None:
        # Use first product to convert bbox to row/col (all products share same grid)
        row_slice, col_slice = _get_rowcol_slice(results[0], bbox, frequency)
        logger.info(f"Converted bbox {bbox} to rows: {row_slice}, cols: {col_slice}")
    else:
        row_slice = slice(*rows) if rows is not None else None
        col_slice = slice(*cols) if cols is not None else None
        logger.info(f"Subsetting to rows: {row_slice}, cols: {col_slice}")

    # Build job arguments
    jobs = [
        (
            str(product.filename),
            row_slice,
            col_slice,
            output_dir,
            frequency,
            polarizations,
        )
        for product in results
    ]
    output_dir.mkdir(exist_ok=True, parents=True)

    if num_workers == 1:
        return [_run_file(job) for job in tqdm(jobs, desc="Processing")]

    return process_map(_run_file, jobs, max_workers=num_workers)
