from __future__ import annotations

import datetime
import logging
import netrc
import warnings
from enum import Enum
from itertools import groupby
from pathlib import Path
from typing import Literal, Sequence, Union

from packaging.version import parse
from shapely.geometry import box

try:
    import asf_search as asf
    from asf_search.ASFSearchResults import ASFSearchResults
except ImportError:
    warnings.warn("Can't import `asf_search`. Unable to search/download data. ")

from ._types import PathOrStr
from .bursts import normalize_burst_id
from .missing_data import BurstSubsetOption, get_missing_data_options

__all__ = [
    "download_cslcs",
    "search_cslcs",
    "download_cslc_static_layers",
    "search_cslc_static_layers",
    "get_urls",
]

logger = logging.getLogger("opera_utils")


class L2Product(str, Enum):
    """OPERA Level 2 product types available on ASF."""

    CSLC = "CSLC"
    CSLC_STATIC = "CSLC-STATIC"
    RTC = "RTC"
    RTC_STATIC = "RTC-STATIC"


# Type for ASF Search start/end times
DatetimeInput = Union[datetime.datetime, str, None]


def download_cslc_static_layers(
    burst_ids: Sequence[str],
    output_dir: PathOrStr,
    max_jobs: int = 3,
) -> list[Path]:
    """Download the static layers for a sequence of burst IDs.

    Parameters
    ----------
    burst_ids : Sequence[str]
        Sequence of OPERA Burst IDs (e.g. 'T123_012345_IW1')
    output_dir : Path | str
        Location to save output rasters to
    max_jobs : int, optional
        Number of parallel downloads to run, by default 3

    Returns
    -------
    list[Path]
        Locations to saved raster files.
    """
    return _download_for_burst_ids(
        burst_ids=burst_ids,
        output_dir=output_dir,
        max_jobs=max_jobs,
        product=L2Product.CSLC_STATIC,
    )


def search_cslcs(
    start: DatetimeInput | None = None,
    end: DatetimeInput | None = None,
    bounds: Sequence[float] | None = None,
    aoi_polygon: str | None = None,
    track: int | None = None,
    burst_ids: Sequence[str] | None = None,
    max_results: int | None = None,
    product: L2Product = L2Product.CSLC,
    check_missing_data: bool = False,
) -> ASFSearchResults | tuple[ASFSearchResults, list[BurstSubsetOption]]:
    """Search for OPERA CSLC products on ASF.

    Parameters
    ----------
    start : datetime.datetime | str, optional
        Start date of data acquisition. Supports timestamps as well as natural language such as "3 weeks ago"
    end : datetime.datetime | str, optional
        end: End date of data acquisition. Supports timestamps as well as natural language such as "3 weeks ago"
    bounds : Sequence[float], optional
        Bounding box coordinates (min lon, min lat, max lon, max lat)
    aoi_polygon : str, optional
        GeoJSON polygon string, alternative to `bounds`.
    track : int, optional
        Relative orbit number / track / path
    burst_ids : Sequence[str], optional
        Sequence of OPERA Burst IDs (e.g. 'T123_012345_IW1')
    max_results : int, optional
        Maximum number of results to return
    product: L2Product
        Type of OPERA Level 2 product to search for.
        Default is L2Product.CSLC
    check_missing_data : bool, optional
        Whether to remove missing data options from the search results, by default False

    Returns
    -------
    asf_search.ASFSearchResults.ASFSearchResults
        Search results from ASF.
    If `check_missing_data` is True, also returns list[BurstSubsetOption],
        indicating possible spatially-consistent subsets of CSLC bursts.
    """
    logger.info("Searching for OPERA CSLC products")
    # If they passed a bounding box, need a WKT polygon
    if bounds is not None:
        if aoi_polygon is not None:
            raise ValueError("Can't pass both `bounds` and `aoi_polygon`")
        aoi = box(*bounds).wkt
    else:
        aoi = aoi_polygon

    results = asf.search(
        start=start,
        end=end,
        intersectsWith=aoi,
        relativeOrbit=track,
        operaBurstID=list(burst_ids) if burst_ids is not None else None,
        dataset=asf.DATASET.OPERA_S1,
        processingLevel=product.value,
        maxResults=max_results,
    )
    logger.debug(f"Found {len(results)} total results before deduping pgeVersion")
    results = filter_results_by_date_and_version(results)
    logger.info(f"Found {len(results)} results")

    if not check_missing_data:
        return results
    missing_data_options = get_missing_data_options(
        slc_files=[r.properties["url"] for r in results]
    )
    return results, missing_data_options


def download_cslcs(
    burst_ids: Sequence[str],
    output_dir: PathOrStr,
    start: DatetimeInput = None,
    end: DatetimeInput = None,
    max_jobs: int = 3,
) -> list[Path]:
    """Download the static layers for a sequence of burst IDs.

    Parameters
    ----------
    burst_ids : Sequence[str]
        Sequence of OPERA Burst IDs (e.g. 'T123_012345_IW1')
    output_dir : Path | str
        Location to save output rasters to
    start: datetime.datetime | str, optional
        Start date of data acquisition. Supports timestamps as well as natural language such as "3 weeks ago"
    end: datetime.datetime | str, optional
        end: End date of data acquisition. Supports timestamps as well as natural language such as "3 weeks ago"
    max_jobs : int, optional
        Number of parallel downloads to run, by default 3

    Returns
    -------
    list[Path]
        Locations to saved raster files.
    """
    return _download_for_burst_ids(
        burst_ids=burst_ids,
        output_dir=output_dir,
        max_jobs=max_jobs,
        start=start,
        end=end,
        product=L2Product.CSLC,
    )


def search_cslc_static_layers(
    bounds: Sequence[float] | None = None,
    aoi_polygon: str | None = None,
    track: int | None = None,
    burst_ids: Sequence[str] | None = None,
) -> ASFSearchResults:
    """Search for OPERA CSLC Static Layers products on ASF.

    Parameters
    ----------
    bounds : Sequence[float], optional
        Bounding box coordinates (min lon, min lat, max lon, max lat)
    aoi_polygon : str, optional
        GeoJSON polygon string, alternative to `bounds`.
    track : int, optional
        Relative orbit number / track / path
    burst_ids : Sequence[str], optional
        Sequence of OPERA Burst IDs (e.g. 'T123_012345_IW1')

    Returns
    -------
    asf_search.ASFSearchResults.ASFSearchResults
        Search results from ASF.
    """
    return search_cslcs(
        bounds=bounds,
        aoi_polygon=aoi_polygon,
        track=track,
        burst_ids=burst_ids,
        product=L2Product.CSLC_STATIC,
        check_missing_data=False,
    )


def _download_for_burst_ids(
    burst_ids: Sequence[str],
    output_dir: PathOrStr,
    product: L2Product,
    max_jobs: int = 3,
    start: DatetimeInput = None,
    end: DatetimeInput = None,
) -> list[Path]:
    """Download files for one product type fo static layers for a sequence of burst IDs.

    Parameters
    ----------
    burst_ids : Sequence[str]
        Sequence of OPERA Burst IDs (e.g. 'T123_012345_IW1')
    output_dir : Path
        Location to save output rasters to
    product : L2Product
        Type of OPERA product to download.
    max_jobs : int, optional
        Number of parallel downloads to run, by default 3
    start: datetime.datetime | str, optional
        Start date of data acquisition. Supports timestamps as well as natural language such as "3 weeks ago"
    end: datetime.datetime | str, optional
        end: End date of data acquisition. Supports timestamps as well as natural language such as "3 weeks ago"

    Returns
    -------
    list[Path]
        Locations to saved raster files.
    """
    logger.info(
        f"Searching {len(burst_ids)} bursts, {product=} (Dates: {start} to {end})"
    )
    results = asf.search(
        operaBurstID=list(map(normalize_burst_id, burst_ids)),
        processingLevel=product.value,
        start=start,
        end=end,
        dataset=asf.DATASET.OPERA_S1,
    )
    if product == L2Product.CSLC:
        logger.debug(f"Found {len(results)} total results before deduping pgeVersion")
        print(f"Found {len(results)} total results before deduping pgeVersion")
        results = filter_results_by_date_and_version(results)

    msg = f"Found {len(results)} results"
    print(msg)
    if len(results) == 0:
        raise ValueError(msg)
    logger.info(msg)
    session = _get_auth_session()
    urls = get_urls(results)
    asf.download_urls(
        urls=urls, path=str(output_dir), session=session, processes=max_jobs
    )
    return [Path(output_dir) / r.properties["fileName"] for r in results]


def filter_results_by_date_and_version(results: ASFSearchResults) -> ASFSearchResults:
    """Filter ASF search results to retain only one result per unique 'startTime'.

    Function selects the result with the latest 'pgeVersion' if multiple results
    exist for the same 'startTime'.

    Parameters
    ----------
    results : asf_search.ASFSearchResults.ASFSearchResults
        List of ASF search results to filter.

    Returns
    -------
    asf_search.ASFSearchResults.ASFSearchResults
        Filtered list of ASF search results with unique 'startTime',
        each having the latest 'pgeVersion'.
    """
    # First, sort the results primarily by 'startTime' and secondarily by 'pgeVersion' in descending order
    sorted_results = sorted(
        results,
        key=lambda r: (
            r.properties["startTime"],
            r.properties["operaBurstID"],
            r.properties["processingDate"],
            parse(r.properties["pgeVersion"]),
        ),
        reverse=True,
    )

    # It is important to sort by startTime before using groupby,
    # as groupby only works correctly if the input data is sorted by the key
    grouped_by_start_time = groupby(
        sorted_results,
        key=lambda r: (r.properties["startTime"], r.properties["operaBurstID"]),
    )

    # Extract the result with the highest pgeVersion for each group
    filtered_results = [
        max(group, key=lambda r: r.properties["processingDate"])
        for _, group in grouped_by_start_time
    ]

    return ASFSearchResults(filtered_results)


def get_urls(
    results: asf.ASFSearchResults,
    type_: Literal["https", "s3"] = "https",
    file_ext: str = ".h5",
) -> list[str]:
    """Parse the `ASFSearchResults` object for HTTPS or S3 urls."""
    if type_ == "https":
        return [r.properties["url"] for r in results]
    elif type_ == "s3":
        out: list[str] = []
        for r in results:
            if "s3Urls" not in r.properties:
                raise ValueError(f"No S3 URL for {r}")

            for url in r.properties["s3Urls"]:
                if url.endswith(file_ext):
                    out.append(url)
                    break
            else:
                raise ValueError(f"Failed to find HDF5 S3 url for {r}")
        return out

    else:
        raise ValueError(f"type_ must be 'https' or 's3'. Got {type_}")


def _get_auth_session() -> asf.ASFSession:
    host = "urs.earthdata.nasa.gov"

    auth = netrc.netrc().authenticators(host)
    if auth is None:
        raise ValueError(f"No .netrc entry found for {host}")
    username, _, password = auth
    return asf.ASFSession().auth_with_creds(username, password)
