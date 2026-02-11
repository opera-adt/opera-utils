"""Shared CMR (Common Metadata Repository) search utilities.

Provides helpers for querying Earthdata's CMR API, including pagination
and URL extraction from UMM metadata.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from opera_utils.constants import UrlType

__all__ = [
    "fetch_cmr_pages",
    "get_download_url",
]

logger = logging.getLogger("opera_utils")


def fetch_cmr_pages(
    search_url: str,
    params: dict[str, int | str | list[str]],
) -> list[dict[str, Any]]:
    """Paginate through all CMR search results and return the UMM items.

    Parameters
    ----------
    search_url : str
        The CMR search endpoint URL
        (e.g. "https://cmr.earthdata.nasa.gov/search/granules.umm_json").
    params : dict[str, int | str | list[str]]
        Query parameters for the CMR request (short_name, page_size, etc.).

    Returns
    -------
    list[dict[str, Any]]
        List of raw UMM item dicts from all pages. Each dict has at least
        a ``"umm"`` key containing the granule UMM metadata.

    """
    headers: dict[str, str] = {}
    items: list[dict[str, Any]] = []

    while True:
        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        items.extend(data.get("items", []))

        if "CMR-Search-After" not in response.headers:
            break

        headers["CMR-Search-After"] = response.headers["CMR-Search-After"]

    return items


def get_download_url(
    umm_data: dict[str, Any],
    protocol: UrlType = UrlType.HTTPS,
    filename_suffix: str | None = None,
) -> str:
    """Extract a download URL from a product's UMM metadata.

    Parameters
    ----------
    umm_data : dict[str, Any]
        The product's UMM metadata dictionary.
    protocol : UrlType
        The protocol to use for downloading, either "s3" or "https".
    filename_suffix : str | None, optional
        If given, prefer URLs that end with this suffix (e.g. ".h5").
        Falls back to any matching URL if no suffixed match is found.

    Returns
    -------
    str
        The download URL.

    Raises
    ------
    ValueError
        If no URL with the specified protocol is found or if the protocol
        is invalid.

    """
    if protocol not in ["https", "s3"]:
        msg = f"Unknown protocol {protocol}; must be https or s3"
        raise ValueError(msg)

    proto_str = str(protocol)

    # First pass: try to match protocol + suffix
    if filename_suffix is not None:
        for url in umm_data.get("RelatedUrls", []):
            if (
                url["Type"].startswith("GET DATA")
                and url["URL"].startswith(proto_str)
                and url["URL"].endswith(filename_suffix)
            ):
                return url["URL"]

    # Second pass (or only pass if no suffix): match protocol only
    for url in umm_data.get("RelatedUrls", []):
        if url["Type"].startswith("GET DATA") and url["URL"].startswith(proto_str):
            return url["URL"]

    granule_id = umm_data.get("GranuleUR", "unknown")
    msg = f"No download URL found for granule {granule_id}"
    raise ValueError(msg)
