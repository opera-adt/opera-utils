import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import requests

from opera_utils.disp._product import DispProduct

logger = logging.getLogger("opera_utils")


class OrbitPass(str, Enum):
    """Choices for the orbit direction of a granule."""

    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"


@dataclass
class Granule:
    """Model representing a single granule from CMR.

    Attributes
    ----------
    product : DispProduct
        The product information.
    url : str
        URL (https or s3) containing granule download location.
    frame_id : int
        The frame number of the granule.
    orbit_pass : OrbitPass
        The orbit direction ("ASCENDING" or "DESCENDING").
    reference_datetime : datetime
        The beginning date/time of the granule.
    secondary_datetime : datetime
        The ending date/time of the granule.
    """

    product: DispProduct
    url: str
    orbit_pass: OrbitPass

    @classmethod
    def from_umm(
        cls,
        umm_data: dict[str, Any],
        url_type: Literal["s3", "https"] = "https",
    ) -> "Granule":
        """Construct a Granule instance from a raw dictionary.

        Parameters
        ----------
        umm_data : dict[str, Any]
            The raw granule UMM data from the CMR API.
        url_type : Literal["s3", "https"]
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
        from ._product import DispProduct

        url = get_download_url(umm_data, protocol=url_type)
        product = DispProduct.from_filename(url)
        additional_attributes = umm_data.get("AdditionalAttributes", [])
        orbit_pass = OrbitPass(
            _get_attr(additional_attributes, "ASCENDING_DESCENDING").upper()
        )
        return cls(
            product=product,
            url=url,
            orbit_pass=orbit_pass,
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


def get_download_url(
    umm_data: dict[str, Any],
    protocol: Literal["s3", "https"] = "https",
) -> str:
    """Extract a download URL from the product's UMM metadata.

    Parameters
    ----------
    umm_data : dict[str, Any]
        The product's umm metadata dictionary
    protocol : Literal["s3", "https"]
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


def _get_attr(attrs: list[dict[str, Any]], name: str) -> str:
    """Get the first attribute value for a given name."""
    for attr in attrs:
        if attr.get("Name") == name:
            values = attr.get("Values", [])
            if values:
                return values[0]
    raise ValueError(f"Missing attribute {name}")


def get_products(
    frame_id: int | None = None,
    product_version: str | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    use_uat: bool = False,
) -> list[Granule]:
    """Query the CMR for granules matching the given frame ID and product version.

    Parameters
    ----------
    frame_id : int
        The frame ID to search for
    product_version : str
        The product version to search for
    start_datetime : datetime, optional
        The start of the temporal range in UTC.
    end_datetime : datetime, optional
        The end of the temporal range in UTC.
    use_uat : bool
        Whether to use the UAT environment.
        Default is False (uses main Earthdata environment)

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
    if frame_id:
        product_filters.append(f"int,FRAME_NUMBER,{frame_id}")
    if product_version:
        product_filters.append(f"float,PRODUCT_VERSION,{product_version}")
    if product_filters:
        params["attribute[]"] = product_filters

    # Optionally narrow search by temporal range
    if start_datetime is not None or end_datetime is not None:
        start_str = start_datetime.isoformat() if start_datetime is not None else ""
        end_str = end_datetime.isoformat() if end_datetime is not None else ""
        params["temporal"] = f"{start_str},{end_str}"

    headers: dict[str, str] = {}
    granules: list[Granule] = []

    while True:
        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        granules.extend([Granule.from_umm(item["umm"]) for item in data["items"]])

        if "CMR-Search-After" not in response.headers:
            break

        headers["CMR-Search-After"] = response.headers["CMR-Search-After"]

    return granules


if __name__ == "__main__":
    import tyro

    for granule in sorted(tyro.cli(get_products), key=lambda g: (g.frame_id,)):
        print(granule.url)
