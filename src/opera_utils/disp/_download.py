import logging
import tempfile
from datetime import datetime
from pathlib import Path

import requests
import xarray as xr
from shapely import from_wkt
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from opera_utils.credentials import get_earthdata_username_password

from ._product import DispProductStack
from ._remote import open_file
from ._search import UrlType, search

logger = logging.getLogger("opera_utils")


def process_file(
    url: str,
    rows: slice | None,
    cols: slice | None,
    output_dir: Path,
    session: requests.Session | None = None,
) -> Path:
    """Download and subset a single NetCDF product.

    Parameters
    ----------
    url : str
        URL of the NetCDF product to download.
    rows : slice | None
        Rows of the product to subset.
    cols : slice | None
        Columns of the product to subset.
    output_dir : Path
        Directory to save the downloaded and subsetted product to.
    session : requests.Session | None
        Authenticated Session to use for downloading the product.

    Returns
    -------
    Path
        Path to the downloaded and subsetted product

    """
    filename = url.split("/")[-1]
    outname = f"{output_dir}/{Path(url).name}"
    outpath = Path(outname)
    if outpath.exists():
        logger.info(f"Skipped (exists): {filename}")
        return outpath

    with tempfile.NamedTemporaryFile(suffix=".nc") as tf:
        temp_path = Path(tf.name)
        if url.startswith("s3://"):
            # For S3 urls, it's fast to just open with xarray on the fsspec object
            with open_file(url) as in_f:
                _extract_subset(in_f, outpath=outpath, rows=rows, cols=cols)
                # out_f.write(in_f.read())
        else:
            # HTTPS seems to run much more slowly
            if session is None:
                session = requests.Session()
                username, password = get_earthdata_username_password()
                session.auth = (username, password)
            response = session.get(url)
            response.raise_for_status()
            with open(temp_path, "wb") as out_f:
                out_f.write(response.content)
            _extract_subset(temp_path, outpath=outpath, rows=rows, cols=cols)

        logger.debug(f"Done: {filename}")
    return outpath


def _extract_subset(
    input_obj, outpath: Path | str, rows: slice | None, cols: slice | None
) -> None:
    X0, X1 = (cols.start, cols.stop) if cols is not None else (None, None)  # type: ignore[union-attr]
    Y0, Y1 = (rows.start, rows.stop) if rows is not None else (None, None)  # type: ignore[union-attr]

    # Open and slice root data
    ds = xr.open_dataset(input_obj, engine="h5netcdf")
    subset = ds.isel(y=slice(Y0, Y1), x=slice(X0, X1))
    subset.to_netcdf(outpath, engine="h5netcdf")

    # Also subset and add /corrections data
    ds_corr = xr.open_dataset(input_obj, engine="h5netcdf", group="corrections")
    corr_subset = ds_corr.isel(y=slice(Y0, Y1), x=slice(X0, X1))
    corr_subset.to_netcdf(outpath, mode="a", engine="h5netcdf", group="corrections")
    # Add the top-level /identification and /metadata too
    for group in "metadata", "identification":
        # Note: we can't use xarray here, due to the np.bytes_ datasets:
        # See https://github.com/pydata/xarray/issues/10389
        import h5py

        with h5py.File(input_obj) as hf, h5py.File(outpath, "a") as dest_hf:
            hf.copy(group, dest_hf, name=group)


def _run_file(
    args: tuple[str, slice | None, slice | None, Path, requests.Session | None],
):
    return process_file(*args)


def run_download(
    frame_id: int | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    wkt: str | None = None,
    url_type: UrlType = UrlType.HTTPS,
    num_workers: int = 8,
    product_version: str | None = "1.0",
    output_dir: Path = Path("./subsets"),
) -> list[Path]:
    """Download and subset DISP-S1 NetCDF products.

    This function downloads and subsets DISP-S1 NetCDF products from the
    Earthdata Search API and saves them to the local file system.

    Optionally, the files can be cropped using the `bbox` or `wkt` parameters.

    Parameters
    ----------
    frame_id : int | None
        Frame ID of the product
    start_datetime : datetime | None
        Start datetime of the product search.
        If None, the search starts at the earliest available product (usually 2016/07).
    end_datetime : datetime | None
        End datetime of the product search.
        If None, the search ends at the latest available product.
    bbox : tuple[float, float, float, float] | None
        Bounding box of the region of interest as
        (west, south, east, north) in degrees longitude/latitude
    wkt : str | None
        Alternative to `bbox`, Well-known text representation of the region of interest
    url_type : UrlType
        Type of URL to use for the product
        Default is HTTPS.
        Note that S3 urls can only be used when running on an in-region EC2 instance.
    num_workers : int
        Number of workers to use for downloading and subsetting
    product_version : str | None
        Product version of the product
    output_dir : Path
        Directory to save the downloaded and subsetted products to.
        Default is "./subsets"

    Returns
    -------
    list[Path]
        List of paths to the downloaded and subsetted products

    """
    if wkt is not None and bbox is not None:
        msg = "Can't provide both `bbox` and `wkt`"
        raise ValueError(msg)
    if wkt is not None:
        poly = from_wkt(wkt)
        bbox = poly.bounds

    if num_workers == 1:
        session = requests.session()
        username, password = get_earthdata_username_password()
        session.auth = (username, password)
    else:
        session = None

    results = search(
        frame_id=frame_id,
        product_version=product_version,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        url_type=url_type,
    )
    n_urls = len(results)
    logger.info(f"Found {n_urls} urls for Frame {frame_id}")

    dps = DispProductStack(results)

    if bbox is not None:
        p = dps.products[0]
        lon_left, lat_bottom, lon_right, lat_top = bbox
        row_start, col_start = p.lonlat_to_rowcol(lon_left, lat_top)
        row_stop, col_stop = p.lonlat_to_rowcol(lon_right, lat_bottom)
        rows = slice(row_start, row_stop)
        cols = slice(col_start, col_stop)
    else:
        rows, cols = None, None
    logger.info(f"Subsetting to Rows: {rows}, cols: {cols}")

    jobs = [
        (filename, rows, cols, output_dir, session)
        for idx, filename in enumerate(dps.filenames)
    ]
    output_dir.mkdir(exist_ok=True, parents=True)

    if num_workers == 1:
        return [_run_file(job) for job in tqdm(jobs)]
    return process_map(_run_file, jobs, max_workers=num_workers)
