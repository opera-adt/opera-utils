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

try:
    import asf_search as asf
    from asf_search.ASFSearchResults import ASFSearchResults
except ImportError:
    warnings.warn("Can't import `asf_search`. Unable to search/download data. ")

from opera_utils._types import PathOrStr
from opera_utils._utils import LoggingContext

__all__ = [
    "download_cslc_static_layers",
    "download_cslcs",
]

logger = logging.getLogger(__name__)


class _DummyContext:
    """Context manager that does nothing, for use when verbose=False."""

    # return nullcontext(None)
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


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
    verbose: bool = False,
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
    verbose : bool, optional
        Whether to print verbose output, by default False

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
        verbose=verbose,
    )


def download_cslcs(
    burst_ids: Sequence[str],
    output_dir: PathOrStr,
    start: DatetimeInput = None,
    end: DatetimeInput = None,
    max_jobs: int = 3,
    verbose: bool = False,
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
    verbose : bool, optional
        Whether to print verbose output, by default False

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
        verbose=verbose,
    )


def _download_for_burst_ids(
    burst_ids: Sequence[str],
    output_dir: PathOrStr,
    product: L2Product,
    max_jobs: int = 3,
    start: DatetimeInput = None,
    end: DatetimeInput = None,
    verbose: bool = False,
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
    verbose : bool, optional
        Whether to print verbose output, by default False

    Returns
    -------
    list[Path]
        Locations to saved raster files.
    """
    cm = LoggingContext if verbose else _DummyContext
    with cm(logger, level=logging.INFO, handler=logging.StreamHandler()):
        # Make a tuple so it can be hashed
        logger.info(
            f"Searching {len(burst_ids)} for {product} (Dates:{start} to {end})"
        )
        results = asf.search(
            operaBurstID=list(burst_ids),
            processingLevel=product.value,
            start=start,
            end=end,
        )
        logger.debug(f"Found {len(results)} total results before deduping pgeVersion")
        results = filter_results_by_date_and_version(results)
        logger.info(f"Found {len(results)} results")
        session = _get_auth_session()
        urls = _get_urls(results)
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
        key=lambda r: (r.properties["startTime"], parse(r.properties["pgeVersion"])),
        reverse=True,
    )

    # It is important to sort by startTime before using groupby,
    # as groupby only works correctly if the input data is sorted by the key
    grouped_by_start_time = groupby(
        sorted_results, key=lambda r: r.properties["startTime"]
    )

    # Extract the result with the highest pgeVersion for each group
    filtered_results = [
        max(group, key=lambda r: parse(r.properties["pgeVersion"]))
        for _, group in grouped_by_start_time
    ]

    return filtered_results


def _get_urls(
    results: asf.ASFSearchResults,
    type_: Literal["https", "s3"] = "https",
) -> list[str]:
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


def _get_auth_session() -> asf.ASFSession:
    host = "urs.earthdata.nasa.gov"

    auth = netrc.netrc().authenticators(host)
    if auth is None:
        raise ValueError(f"No .netrc entry found for {host}")
    username, _, password = auth
    return asf.ASFSession().auth_with_creds(username, password)
