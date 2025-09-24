# opera_utils/tropo.py
from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import requests

logger = logging.getLogger("opera_utils")

# CMR short name for TROPO v1
TROPO_SHORT_NAME = "OPERA_L4_TROPO-ZENITH_V1"


class UrlType(str, Enum):
    """Preferred data access protocol."""

    HTTPS = "https"
    S3 = "s3"

    def __str__(self) -> str:
        return self.value


def _parse_dt(s: str) -> datetime:
    # CMR returns timestamps like "2016-07-01T00:00:00Z"
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _cmr_search(
    *,
    short_name: str,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    attributes: Iterable[str] | None = None,
    use_uat: bool = False,
) -> list[dict[str, Any]]:
    """Query the CMR for granules matching the product type and date range.

    Parameters
    ----------
    short_name : str
        Name of CMR data set.
    start_datetime : datetime, optional
        The start of the temporal range in UTC.
    end_datetime : datetime, optional
        The end of the temporal range in UTC.
    attributes : Iterable[str], optional
        Filters to use in the CMR query.
    use_uat : bool
        Whether to use the UAT environment instead of main Earthdata endpoint.

    Returns
    -------
    list
        raw UMM granule dicts for a collection.

    """
    edl_host = "uat.earthdata" if use_uat else "earthdata"
    base = f"https://cmr.{edl_host}.nasa.gov/search/granules.umm_json"
    page_size: int = 500

    params: dict[str, Any] = {"short_name": short_name, "page_size": page_size}
    if attributes:
        params["attribute[]"] = list(attributes)

    if start_datetime or end_datetime:
        # Let CMR do initial temporal filtering
        start_str = start_datetime.isoformat() if start_datetime else ""
        end_str = end_datetime.isoformat() if end_datetime else ""
        params["temporal"] = f"{start_str},{end_str}"

    headers: dict[str, str] = {}
    out: list[dict[str, Any]] = []

    while True:
        resp = requests.get(base, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("items", [])
        out.extend([it.get("umm", {}) for it in items])

        # pagination
        sa = resp.headers.get("CMR-Search-After")
        if not sa:
            break
        headers["CMR-Search-After"] = sa

    return out


def _pick_related_url(
    umm: dict[str, Any],
    *,
    kind: str,
    startswith: str | None = None,
    endswith: str | None = None,
) -> str | None:
    for item in umm.get("RelatedUrls", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("Type") != kind:
            continue
        url = item.get("URL")
        if not url:
            continue
        if startswith and not url.startswith(startswith):
            continue
        if endswith and not url.endswith(endswith):
            continue
        return url
    return None
