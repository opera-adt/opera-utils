from __future__ import annotations

import logging
from datetime import datetime, timezone

from opera_utils._cmr import UrlType, _cmr_search

from ._product import TropoProduct

__all__ = ["search_tropo"]

logger = logging.getLogger("opera_utils")

# CMR short name for TROPO v1
TROPO_SHORT_NAME = "OPERA_L4_TROPO-ZENITH_V1"


def search_tropo(
    *,
    product_version: str | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    url_type: UrlType = UrlType.HTTPS,
    use_uat: bool = False,
    print_urls: bool = False,
) -> list[TropoProduct]:
    """Search CMR for OPERA TROPO-ZENITH granules and return parsed products.

    Notes
    -----
    - TROPO granules are global 6-hour fields; we match on their *begin* time falling
      within the requested temporal window (after CMR's coarse temporal prefilter).
    - If no start/end are given, CMR's full record is returned (can be large).

    Examples
    --------
    >>> prods = search_tropo(
    ...     start_datetime=datetime(2016, 7, 1, tzinfo=timezone.utc),
    ...     end_datetime=datetime(2016, 7, 2, tzinfo=timezone.utc),
    ... )
    >>> prods[0].url()  # HTTPS by default
    'https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/.../OPERA_L4_TROPO-ZENITH_....nc'

    """
    attrs: list[str] = []
    if product_version:
        if not product_version.startswith("v"):
            product_version = "v" + product_version
        # here values are strings (e.g., "v1.0"), so use string matching.
        attrs.append(f"string,PRODUCT_VERSION,{product_version}")

    # Fetch raw UMMs
    umms = _cmr_search(
        short_name=TROPO_SHORT_NAME,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        attributes=attrs or None,
        use_uat=use_uat,
    )

    # Parse and do precise temporal filtering by begin time (inclusive)
    prods = [TropoProduct.from_umm(u) for u in umms]
    if start_datetime or end_datetime:
        s = start_datetime or datetime(1900, 1, 1, tzinfo=timezone.utc)
        e = end_datetime or datetime(2100, 1, 1, tzinfo=timezone.utc)
        prods = [p for p in prods if s <= p.begin <= e]

    prods.sort(key=lambda p: p.begin)

    if print_urls:
        for p in prods:
            u = p.url(url_type) or p.url(UrlType.HTTPS) or p.url(UrlType.S3)
            if u:
                print(u)

    return prods
