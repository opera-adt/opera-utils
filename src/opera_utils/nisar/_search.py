"""Search for NISAR GSLC products from CMR.

Examples
--------
$ python -m opera_utils.nisar._search --track-frame 004_076_A_022

"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone

from opera_utils._cmr import fetch_cmr_pages

from ._product import GslcProduct, UrlType

__all__ = ["search"]

logger = logging.getLogger("opera_utils")

# CMR short name for NISAR GSLC products
# This may need to be updated once official products are released
NISAR_GSLC_SHORT_NAME = "NISAR_L2_GSLC_BETA_V1"


def search(
    bbox: tuple[float, float, float, float] | None = None,
    track_frame: str | None = None,
    track_frame_number: int | None = None,
    orbit_direction: str | None = None,
    cycle_number: int | None = None,
    relative_orbit_number: int | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    url_type: UrlType = UrlType.HTTPS,
    short_name: str = NISAR_GSLC_SHORT_NAME,
    provider: str = "ASF",
    print_urls: bool = False,
) -> list[GslcProduct]:
    """Query CMR for NISAR GSLC granules matching the given criteria.

    Parameters
    ----------
    bbox : tuple[float, float, float, float], optional
        Bounding box as (west, south, east, north) in degrees lon/lat.
        CMR will return products that intersect this region.
    track_frame : str, optional
        The track/frame identifier to search for, in format "CCC_RRR_D_TTT"
        where CCC=cycle, RRR=relative orbit, D=direction (A/D), TTT=track frame.
        Example: "004_076_A_022". Note: cycle changes between acquisitions,
        so this is typically only useful for finding a specific granule.
    track_frame_number : int, optional
        The track frame number (e.g., 8). This stays constant for repeat passes.
    orbit_direction : str, optional
        Orbit direction: "A" for ascending, "D" for descending.
    cycle_number : int, optional
        The cycle number to search for.
    relative_orbit_number : int, optional
        The relative orbit number to search for.
    start_datetime : datetime, optional
        The start of the temporal range in UTC.
    end_datetime : datetime, optional
        The end of the temporal range in UTC.
    url_type : UrlType
        The protocol to use for downloading, either "s3" or "https".
    short_name : str
        The CMR collection short name.
        Default is "NISAR_L2_GSLC_BETA_V1" (beta products).
    provider : str
        The CMR data provider. Default is "ASF".
    print_urls : bool
        If True, prints out the result urls to stdout in addition to returning
        the `GslcProduct` objects.
        Default is False.

    Returns
    -------
    list[GslcProduct]
        List of products matching the search criteria.

    Examples
    --------
    Search by bounding box (most ergonomic for finding a time series):

    >>> products = search(bbox=(40.62, 13.56, 40.72, 13.64))

    Search by orbit parameters:

    >>> products = search(relative_orbit_number=172, track_frame_number=8,
    ...                   orbit_direction="A")

    """
    search_url = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
    params: dict[str, int | str | list[str]] = {
        "short_name": short_name,
        "provider": provider,
        "page_size": 500,
    }

    # Add bounding box for spatial filtering (CMR format: west,south,east,north)
    if bbox is not None:
        west, south, east, north = bbox
        params["bounding_box"] = f"{west},{south},{east},{north}"

    # Optionally narrow search by temporal range
    if start_datetime is not None or end_datetime is not None:
        start_str = start_datetime.isoformat() if start_datetime is not None else ""
        end_str = end_datetime.isoformat() if end_datetime is not None else ""
        params["temporal"] = f"{start_str},{end_str}"

    # If no temporal range is specified, set wide defaults for filtering
    if start_datetime is None:
        start_datetime = datetime(2024, 1, 1, tzinfo=timezone.utc)
    else:
        start_datetime = start_datetime.replace(tzinfo=timezone.utc)
    if end_datetime is None:
        end_datetime = datetime(2100, 1, 1, tzinfo=timezone.utc)
    else:
        end_datetime = end_datetime.replace(tzinfo=timezone.utc)

    # Add attribute filters for track/frame if provided
    # CMR attribute names: TRACK_NUMBER (=relative orbit), FRAME_NUMBER (=track frame),
    # ASCENDING_DESCENDING (=orbit direction as "ASCENDING" or "DESCENDING")
    attribute_filters: list[str] = []
    if relative_orbit_number is not None:
        attribute_filters.append(f"int,TRACK_NUMBER,{relative_orbit_number}")
    if track_frame_number is not None:
        attribute_filters.append(f"int,FRAME_NUMBER,{track_frame_number}")
    if orbit_direction is not None:
        dir_str = "ASCENDING" if orbit_direction.upper() == "A" else "DESCENDING"
        attribute_filters.append(f"string,ASCENDING_DESCENDING,{dir_str}")
    if attribute_filters:
        params["attribute[]"] = attribute_filters

    # Warn if no spatial or track constraints specified
    no_constraints = (
        bbox is None
        and track_frame is None
        and cycle_number is None
        and relative_orbit_number is None
    )
    if no_constraints:
        warnings.warn(
            "No bbox, track_frame, cycle_number, or relative_orbit_number specified: "
            "search may be large",
            stacklevel=1,
        )

    items = fetch_cmr_pages(search_url, params)

    products: list[GslcProduct] = []
    for item in items:
        try:
            product = GslcProduct.from_umm(item["umm"], url_type=url_type)
            # Filter by track_frame if specified (exact match including cycle)
            if track_frame is not None and product.track_frame_id != track_frame:
                continue
            # Filter by track_frame_number (stays constant for repeat passes)
            if (
                track_frame_number is not None
                and product.track_frame_number != track_frame_number
            ):
                continue
            # Filter by orbit direction
            if (
                orbit_direction is not None
                and str(product.orbit_direction) != orbit_direction.upper()
            ):
                continue
            # Filter by datetime
            if start_datetime <= product.start_datetime <= end_datetime:
                products.append(product)
        except (ValueError, KeyError) as e:
            logger.debug(f"Skipping granule due to parse error: {e}")
            continue

    # Return sorted list of products
    products = sorted(products, key=lambda g: (g.track_frame_id, g.start_datetime))

    if print_urls:
        for p in products:
            print(p.filename)

    return products


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search for NISAR GSLC products")
    parser.add_argument(
        "--track-frame", type=str, help="Track/frame ID (e.g. 004_076_A_022)"
    )
    parser.add_argument("--cycle-number", type=int, help="Cycle number")
    parser.add_argument(
        "--relative-orbit-number", type=int, help="Relative orbit number"
    )
    parser.add_argument(
        "--start-datetime", type=str, help="Start datetime (ISO format)"
    )
    parser.add_argument("--end-datetime", type=str, help="End datetime (ISO format)")

    args = parser.parse_args()

    start_dt = (
        datetime.fromisoformat(args.start_datetime) if args.start_datetime else None
    )
    end_dt = datetime.fromisoformat(args.end_datetime) if args.end_datetime else None

    results = search(
        track_frame=args.track_frame,
        cycle_number=args.cycle_number,
        relative_orbit_number=args.relative_orbit_number,
        start_datetime=start_dt,
        end_datetime=end_dt,
        print_urls=True,
    )
    print(f"Found {len(results)} products")
