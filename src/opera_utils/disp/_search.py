"""Search for OPERA DISP granules from CMR.

Examples
--------
$ python -m opera_utils.disp.search --frame-id 11115 --end-datetime 2016-10-01
https://datapool.asf.alaska.edu/DISP/OPERA-S1/OPERA_L3_DISP-S1_IW_F11115_VV_20160810T140735Z_20160903T140736Z_v1.0_20250318T223016Z.nc
https://datapool.asf.alaska.edu/DISP/OPERA-S1/OPERA_L3_DISP-S1_IW_F11115_VV_20160810T140735Z_20160927T140737Z_v1.0_20250318T223016Z.nc
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import requests

from opera_utils.disp._product import DispProduct

__all__ = ["search", "Granule"]

logger = logging.getLogger("opera_utils")


class UrlType(str, Enum):
    """Choices for the orbit direction of a granule."""

    S3 = "s3"
    HTTPS = "https"

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class Granule:
    """Model representing a single granule from CMR.

    Attributes
    ----------
    product : DispProduct
        Object containing known DISP-S1 frame information.
    url : str
        URL (https or s3) containing granule download location.
    size_in_bytes : int, optional
        The size in bytes of the file as listed on CMR.
    """

    product: DispProduct
    url: str
    size_in_bytes: int | None

    @classmethod
    def from_umm(
        cls, umm_data: dict[str, Any], url_type: UrlType = UrlType.HTTPS
    ) -> "Granule":
        """Construct a Granule instance from a raw dictionary.

        Parameters
        ----------
        umm_data : dict[str, Any]
            The raw granule UMM data from the CMR API.
        url_type : UrlType
            Type of url to use from the Product.
            "s3" for S3 URLs (direct access), "https" for HTTPS URLs.

        Returns
        -------
        Granule
            The parsed Granule instance.

        Raises
        ------
        ValueError
            If required temporal extent data is missing.
        """
        url = _get_download_url(umm_data, protocol=url_type)
        product = DispProduct.from_filename(url)
        archive_info = umm_data.get("DataGranule", {}).get(
            "ArchiveAndDistributionInformation", []
        )
        size_in_bytes = archive_info[0].get("SizeInBytes", 0) if archive_info else None
        return cls(
            product=product,
            url=url,
            size_in_bytes=size_in_bytes,
        )

    @property
    def frame_id(self) -> int:
        return self.product.frame_id

    @property
    def reference_datetime(self) -> datetime:
        return self.product.reference_datetime

    @property
    def secondary_datetime(self) -> datetime:
        return self.product.secondary_datetime


def _get_download_url(
    umm_data: dict[str, Any], protocol: UrlType = UrlType.HTTPS
) -> str:
    """Extract a download URL from the product's UMM metadata.

    Parameters
    ----------
    umm_data : dict[str, Any]
        The product's umm metadata dictionary
    protocol : UrlType
        The protocol to use for downloading, either "s3" or "https"

    Returns
    -------
    str
        The download URL

    Raises
    ------
    ValueError
        If no URL with the specified protocol is found or if the protocol is invalid
    """
    if protocol not in ["https", "s3"]:
        raise ValueError(f"Unknown protocol {protocol}; must be https or s3")

    for url in umm_data["RelatedUrls"]:
        if url["Type"].startswith("GET DATA") and url["URL"].startswith(protocol):
            return url["URL"]

    raise ValueError(f"No download URL found for granule {umm_data['GranuleUR']}")


def search(
    frame_id: int | None = None,
    product_version: str | None = "1.0",
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    url_type: UrlType = UrlType.HTTPS,
    use_uat: bool = False,
) -> list[Granule]:
    """Query the CMR for granules matching the given frame ID and product version.

    Parameters
    ----------
    frame_id : int, optional
        The frame ID to search for
    product_version : str, optional
        The product version to search for
    start_datetime : datetime, optional
        The start of the temporal range in UTC.
    end_datetime : datetime, optional
        The end of the temporal range in UTC.
    url_type : UrlType
        The protocol to use for downloading, either "s3" or "https".
    use_uat : bool
        Whether to use the UAT environment instead of main Earthdata endpoint.

    Returns
    -------
    list[Granule]
        list of Granule instances matching the search criteria
    """
    edl_host = "uat.earthdata" if use_uat else "earthdata"
    search_url = f"https://cmr.{edl_host}.nasa.gov/search/granules.umm_json"
    params: dict[str, int | str | list[str]] = {
        "short_name": "OPERA_L3_DISP-S1_V1",
        "page_size": 2000,
    }
    # Optionally narrow search by frame id, product version
    product_filters: list[str] = []
    if product_version:
        product_filters.append(f"float,PRODUCT_VERSION,{product_version}")
    if product_filters:
        params["attribute[]"] = product_filters

    # Optionally narrow search by temporal range
    if start_datetime is not None or end_datetime is not None:
        start_str = start_datetime.isoformat() if start_datetime is not None else ""
        end_str = end_datetime.isoformat() if end_datetime is not None else ""
        params["temporal"] = f"{start_str},{end_str}"

    # If no temporal range is specified, default to all granules
    # Ensure datetime objects are timezone-aware
    if start_datetime is None:
        start_datetime = datetime(2014, 1, 1, tzinfo=timezone.utc)
    else:
        start_datetime = start_datetime.replace(tzinfo=timezone.utc)
    if end_datetime is None:
        end_datetime = datetime(2100, 1, 1, tzinfo=timezone.utc)
    else:
        end_datetime = end_datetime.replace(tzinfo=timezone.utc)

    if frame_id:
        product_filters.append(f"int,FRAME_NUMBER,{frame_id}")
    else:
        warnings.warn("No `frame_id` specified: search may be large", stacklevel=1)

    headers: dict[str, str] = {}
    granules: list[Granule] = []
    while True:
        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        cur_granules = [
            Granule.from_umm(item["umm"], url_type=url_type) for item in data["items"]
        ]
        # CMR filters apply to both the reference and secondary time (as of 2025-03-29)
        # We want to filter just by the secondary time
        granules.extend(
            [
                g
                for g in cur_granules
                if start_datetime <= g.secondary_datetime <= end_datetime
            ]
        )

        if "CMR-Search-After" not in response.headers:
            break

        headers["CMR-Search-After"] = response.headers["CMR-Search-After"]

    # Return sorted list of granules
    return sorted(granules, key=lambda g: (g.frame_id, g.secondary_datetime))
