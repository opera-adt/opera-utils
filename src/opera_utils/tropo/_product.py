from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from opera_utils._cmr import UrlType, _pick_related_url

__all__ = ["TropoProduct"]

logger = logging.getLogger("opera_utils")

# CMR short name for TROPO v1
TROPO_SHORT_NAME = "OPERA_L4_TROPO-ZENITH_V1"


def _parse_dt(s: str) -> datetime:
    # CMR returns Zulu timestamps, e.g., "2016-07-01T00:00:00Z"
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class TropoProduct:
    """Parsed OPERA L4 TROPO-ZENITH granule metadata (one 6-hour global field)."""

    granule_ur: str
    short_name: str
    version: str
    begin: datetime
    end: datetime
    product_type: str | None
    product_type_desc: str | None
    product_version: str | None

    # URLs (if present)
    https_url: str | None
    s3_url: str | None
    browse_png_url: str | None
    iso_xml_url: str | None

    # DataGranule bits
    size_in_bytes: int | None
    md5: str | None

    raw_umm: dict[str, Any]  # keep for anything not modeled above

    @property
    def mid_datetime(self) -> datetime:
        return self.begin + (self.end - self.begin) / 2

    @property
    def cadence_hours(self) -> float:
        return (self.end - self.begin).total_seconds() / 3600.0

    def url(self, preferred: UrlType = UrlType.HTTPS) -> str | None:
        if preferred is UrlType.S3 and self.s3_url:
            return self.s3_url
        return self.https_url

    @property
    def filename(self) -> str | None:
        u = self.url(UrlType.HTTPS) or self.url(UrlType.S3)
        return None if u is None else u.split("/")[-1]

    @staticmethod
    def from_umm(umm: dict[str, Any]) -> TropoProduct:
        # Temporal
        rng = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
        begin = _parse_dt(rng["BeginningDateTime"])
        end = _parse_dt(rng["EndingDateTime"])

        granule_ur: str = umm["GranuleUR"]
        short_name = umm.get("CollectionReference", {}).get("ShortName", "")
        version = umm.get("CollectionReference", {}).get("Version", "")

        # Additional attributes
        aa = umm.get("AdditionalAttributes", []) or []
        aa_map = {item.get("Name"): (item.get("Values") or [None])[0] for item in aa}
        product_type = aa_map.get("PRODUCT_TYPE")
        product_type_desc = aa_map.get("PRODUCT_TYPE_DESC")
        product_version = aa_map.get("PRODUCT_VERSION")

        # URLs
        https_url = _pick_related_url(umm, kind="GET DATA", startswith="https")
        s3_url = _pick_related_url(umm, kind="GET DATA", startswith="s3")
        browse_png_url = _pick_related_url(
            umm, kind="GET RELATED VISUALIZATION", endswith=".png"
        )
        iso_xml_url = _pick_related_url(
            umm, kind="EXTENDED METADATA", endswith=".iso.xml"
        )

        # DataGranule / size / md5
        archive_info = (
            umm.get("DataGranule", {}).get("ArchiveAndDistributionInformation", [])
            or []
        )
        size_in_bytes = None
        md5 = None
        if archive_info:
            # First entry is typically the .nc
            primary = archive_info[0]
            size_in_bytes = primary.get("SizeInBytes")
            md5 = (primary.get("Checksum") or {}).get("Value")

        return TropoProduct(
            granule_ur=granule_ur,
            short_name=short_name,
            version=version,
            begin=begin,
            end=end,
            product_type=product_type,
            product_type_desc=product_type_desc,
            product_version=product_version,
            https_url=https_url,
            s3_url=s3_url,
            browse_png_url=browse_png_url,
            iso_xml_url=iso_xml_url,
            size_in_bytes=size_in_bytes,
            md5=md5,
            raw_umm=umm,
        )
