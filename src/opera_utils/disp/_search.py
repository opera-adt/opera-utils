"""Search for OPERA DISP products from CMR.

Examples
--------
$ python -m opera_utils.disp.search --frame-id 11115 --end-datetime 2016-10-01
https://datapool.asf.alaska.edu/DISP/OPERA-S1/OPERA_L3_DISP-S1_IW_F11115_VV_20160810T140735Z_20160903T140736Z_v1.0_20250318T223016Z.nc
https://datapool.asf.alaska.edu/DISP/OPERA-S1/OPERA_L3_DISP-S1_IW_F11115_VV_20160810T140735Z_20160927T140737Z_v1.0_20250318T223016Z.nc

"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone

from opera_utils._cmr import _cmr_search
from opera_utils.disp._product import DispProduct, UrlType

__all__ = ["search"]

logger = logging.getLogger("opera_utils")


def search(
    frame_id: int | None = None,
    product_version: str | None = "1.0",
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    url_type: UrlType = UrlType.HTTPS,
    use_uat: bool = False,
    print_urls: bool = False,
) -> list[DispProduct]:
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
    print_urls : bool
        If True, prints out the result urls to stdout in addition to returning
        the `DispProduct` objects.
        Default is False.

    Returns
    -------
    list[DispProduct]
        List of products matching the search criteria

    """
    params: dict[str, int | str | list[str]] = {
        "short_name": "OPERA_L3_DISP-S1_V1",
        "page_size": 500,
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
    results = _cmr_search(
        short_name="OPERA_L3_DISP-S1_V1",
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        attributes=product_filters,
        use_uat=use_uat,
    )
    products = [DispProduct.from_umm(r, url_type=url_type) for r in results]

    products = [
        p for p in products if start_datetime <= p.secondary_datetime <= end_datetime
    ]
    # Return sorted list of products
    products = sorted(products, key=lambda g: (g.frame_id, g.secondary_datetime))
    if print_urls:
        for p in products:
            print(p.filename)
    return products
