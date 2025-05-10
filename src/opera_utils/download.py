from __future__ import annotations

import datetime
import logging
import netrc
import urllib.parse
import warnings
from collections.abc import Sequence
from enum import Enum
from itertools import groupby
from pathlib import Path
from typing import Literal, Union

from packaging.version import parse
from shapely.geometry import box

try:
    import asf_search as asf
    from asf_search.ASFSearchResults import ASFSearchResults
except ImportError:
    warnings.warn(
        "Can't import `asf_search`. Unable to search/download data. ", stacklevel=2
    )

from ._types import PathOrStr
from .bursts import normalize_burst_id
from .missing_data import BurstSubsetOption, get_missing_data_options

__all__ = [
    "L2Product",
    "RTCStaticLayers",
    "download_cslc_static_layers",
    "download_cslcs",
    "download_rtc_static_layers",
    "download_rtcs",
    "get_urls",
    "search_cslc_static_layers",
    "search_cslcs",
]

logger = logging.getLogger("opera_utils")


class L2Product(str, Enum):
    """OPERA Level 2 product types available on ASF."""

    CSLC = "CSLC"
    CSLC_STATIC = "CSLC-STATIC"
    RTC = "RTC"
    RTC_STATIC = "RTC-STATIC"


class RTCStaticLayers(str, Enum):
    """Available rasters from the RTC-STATIC collection."""

    INCIDENCE_ANGLE = "incidence_angle"
    LOCAL_INCIDENCE_ANGLE = "local_incidence_angle"
    MASK = "mask"
    NUMBER_OF_LOOKS = "number_of_looks"
    RTC_ANF_GAMMA0_TO_BETA0 = "rtc_anf_gamma0_to_beta0"
    RTC_ANF_GAMMA0_TO_SIGMA0 = "rtc_anf_gamma0_to_sigma0"


# Type for ASF Search start/end times
DatetimeInput = Union[datetime.datetime, str, None]


def download_rtc_static_layers(
    burst_ids: Sequence[str],
    output_dir: PathOrStr,
    layers: Sequence[RTCStaticLayers] = (RTCStaticLayers.LOCAL_INCIDENCE_ANGLE,),
    max_jobs: int = 3,
) -> list[Path]:
    """Download the RTC-S1 static layers for a sequence of burst IDs.

    Parameters
    ----------
    burst_ids : Sequence[str]
        Sequence of OPERA Burst IDs (e.g. 'T123_012345_IW1')
    output_dir : Path | str
        Location to save output rasters to
    layers : Sequence[RTCStaticLayers]
        Sequence of static layers rasters to download.
        Choices are contained in `RTCStaticLayers`.
        Default is "local_incidence_angle"
    max_jobs : int, optional
        Number of parallel downloads to run, by default 3

    Returns
    -------
    list[Path]
        Locations to saved raster files.

    """
    selected_layers = [RTCStaticLayers(layer) for layer in layers]

    normalized_burst_ids = list(map(normalize_burst_id, burst_ids))
    results = asf.search(
        operaBurstID=normalized_burst_ids,
        processingLevel=L2Product.RTC_STATIC.value,
        dataset=asf.DATASET.OPERA_S1,
    )

    msg = f"Found {len(results)} results"
    if len(results) == 0:
        raise ValueError(msg)
    logger.info(msg)

    session = _get_auth_session()
    # Note: RTC Static must be slightly different since there's more than one 'url'
    # the "url" and "fileName" ASF search result only give the incidence geotiff
    # There are also these:
    # '.iso.xml'
    # '_local_incidence_angle.tif'
    # '_mask.tif'
    # '_number_of_looks.tif'
    # '_rtc_anf_gamma0_to_beta0.tif'
    # '_rtc_anf_gamma0_to_sigma0.tif'
    #
    # Filter to only the ones we requested
    urls: list[str] = []
    out_paths: list[Path] = []
    for result in results:
        candidate_urls = result.properties["additionalUrls"]
        for u in candidate_urls:
            if any(u.endswith(f"_{layer.value}.tif") for layer in selected_layers):
                urls.append(u)
                out_path = Path(output_dir) / Path(urllib.parse.urlparse(u).path).name
                out_paths.append(out_path)

    if not out_paths:
        msg = f"No urls found for {normalized_burst_ids} and {selected_layers}"
        raise ValueError(msg)

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    asf.download_urls(
        urls=urls, path=str(output_dir), session=session, processes=max_jobs
    )
    return out_paths


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


def search_l2(
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
        Start date of data acquisition.
        Supports timestamps as well as natural language such as "3 weeks ago"
    end : datetime.datetime | str, optional
        end: End date of data acquisition.
        Supports timestamps as well as natural language such as "3 weeks ago"
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
            msg = "Can't pass both `bounds` and `aoi_polygon`"
            raise ValueError(msg)
        aoi = box(*bounds).wkt
    else:
        aoi = aoi_polygon

    opera_burst_ids = (
        list(map(normalize_burst_id, burst_ids)) if burst_ids is not None else None
    )
    results = asf.search(
        start=start,
        end=end,
        intersectsWith=aoi,
        relativeOrbit=track,
        operaBurstID=opera_burst_ids,
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


search_cslcs = search_l2


def download_cslcs(
    burst_ids: Sequence[str],
    output_dir: PathOrStr,
    start: DatetimeInput = None,
    end: DatetimeInput = None,
    max_jobs: int = 3,
) -> list[Path]:
    """Download the matching CSLC-S1 files for a sequence of burst IDs.

    Parameters
    ----------
    burst_ids : Sequence[str]
        Sequence of OPERA Burst IDs (e.g. 'T123_012345_IW1')
    output_dir : Path | str
        Location to save output rasters to
    start: datetime.datetime | str, optional
        Start date of data acquisition.
        Supports timestamps as well as natural language such as "3 weeks ago"
    end: datetime.datetime | str, optional
        end: End date of data acquisition.
        Supports timestamps as well as natural language such as "3 weeks ago"
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


def download_rtcs(
    burst_ids: Sequence[str],
    output_dir: PathOrStr,
    start: DatetimeInput = None,
    end: DatetimeInput = None,
    max_jobs: int = 3,
) -> list[Path]:
    """Download the matching RTC-S1 files for a sequence of burst IDs.

    Parameters
    ----------
    burst_ids : Sequence[str]
        Sequence of OPERA Burst IDs (e.g. 'T123_012345_IW1')
    output_dir : Path | str
        Location to save output rasters to
    start: datetime.datetime | str, optional
        Start date of data acquisition.
        Supports timestamps as well as natural language such as "3 weeks ago"
    end: datetime.datetime | str, optional
        end: End date of data acquisition.
        Supports timestamps as well as natural language such as "3 weeks ago"
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
        product=L2Product.RTC,
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
        Start date of data acquisition.
        Supports timestamps as well as natural language such as "3 weeks ago"
    end: datetime.datetime | str, optional
        end: End date of data acquisition.
        Supports timestamps as well as natural language such as "3 weeks ago"

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
        results = filter_results_by_date_and_version(results)

    msg = f"Found {len(results)} results"
    if len(results) == 0:
        raise ValueError(msg)
    logger.info(msg)
    session = _get_auth_session()
    urls = get_urls(results)
    Path(output_dir).mkdir(exist_ok=True, parents=True)
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
    # First, sort the results primarily by 'startTime' and
    # secondarily by 'pgeVersion' in descending order
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
                msg = f"No S3 URL for {r}"
                raise ValueError(msg)

            for url in r.properties["s3Urls"]:
                if url.endswith(file_ext):
                    out.append(url)
                    break
            else:
                msg = f"Failed to find HDF5 S3 url for {r}"
                raise ValueError(msg)
        return out

    else:
        msg = f"type_ must be 'https' or 's3'. Got {type_}"
        raise ValueError(msg)


def _get_auth_session() -> asf.ASFSession:
    host = "urs.earthdata.nasa.gov"

    auth = netrc.netrc().authenticators(host)
    if auth is None:
        msg = f"No .netrc entry found for {host}"
        raise ValueError(msg)
    username, _, password = auth
    return asf.ASFSession().auth_with_creds(username, password)
