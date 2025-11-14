from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from opera_utils._cmr import UrlType, _pick_related_url

__all__ = ["TropoProduct"]

logger = logging.getLogger("opera_utils")

TROPO_SHORT_NAME = "OPERA_L4_TROPO-ZENITH_V1"


@dataclass(frozen=True)
class TropoProduct:
    """Parsed OPERA L4 TROPO-ZENITH granule metadata (one 6-hour global field)."""

    granule_ur: str
    start: datetime
    end: datetime
    product_version: str | None

    # URLs (if present)
    https_url: str | None
    s3_url: str | None
    browse_png_url: str | None

    size_in_bytes: int | None

    @property
    def mid_datetime(self) -> datetime:
        return self.start + (self.end - self.start) / 2

    @property
    def cadence_hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0

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
        start = datetime.fromisoformat(rng["BeginningDateTime"])
        end = datetime.fromisoformat(rng["EndingDateTime"])

        granule_ur: str = umm["GranuleUR"]

        # Additional attributes
        aa = umm.get("AdditionalAttributes", []) or []
        aa_map = {item.get("Name"): (item.get("Values") or [None])[0] for item in aa}
        product_version = aa_map.get("PRODUCT_VERSION")

        # URLs
        https_url = _pick_related_url(umm, kind="GET DATA", startswith="https")
        s3_url = _pick_related_url(
            umm, kind="GET DATA VIA DIRECT ACCESS", startswith="s3"
        )
        browse_png_url = _pick_related_url(
            umm, kind="GET RELATED VISUALIZATION", endswith=".png"
        )

        # DataGranule / size / md5
        archive_info = (
            umm.get("DataGranule", {}).get("ArchiveAndDistributionInformation", [])
            or []
        )
        size_in_bytes = None
        if archive_info:
            # First entry is typically the .nc
            primary = archive_info[0]
            size_in_bytes = primary.get("SizeInBytes")

        return TropoProduct(
            granule_ur=granule_ur,
            start=start,
            end=end,
            product_version=product_version,
            https_url=https_url,
            s3_url=s3_url,
            browse_png_url=browse_png_url,
            size_in_bytes=size_in_bytes,
        )
