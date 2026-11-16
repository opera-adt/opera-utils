"""Unit tests for DISP-S1-STATIC search parsing (no network)."""

from opera_utils.constants import UrlType
from opera_utils.disp._static import _parse_granule

ITEM = {
    "umm": {
        "GranuleUR": "OPERA_L3_DISP-S1-STATIC_F08882_20140403_S1A_v1.0",
        "AdditionalAttributes": [
            {"Name": "ASCENDING_DESCENDING", "Values": ["ASCENDING"]},
            {"Name": "FRAME_NUMBER", "Values": ["8882"]},
        ],
        "RelatedUrls": [
            {"URL": f"{p}OPERA_L3_DISP-S1-STATIC_F08882_20140403_S1A_v1.0_{layer}.tif"}
            for p in (
                "https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/x/",
                "s3://asf-cumulus-prod-opera-products/x/",
            )
            for layer in ("dem", "layover_shadow_mask", "line_of_sight_enu")
        ] + [{"URL": "https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/x/browse.png"}],
    }
}


def test_parse_granule_https_and_s3():
    g = _parse_granule(ITEM, UrlType.HTTPS)
    assert g.frame_id == 8882
    assert g.direction == "ascending"
    assert set(g.urls) == {"dem", "layover_shadow_mask", "line_of_sight_enu"}
    assert all(u.startswith("https://") for u in g.urls.values())
    s3_urls = _parse_granule(ITEM, UrlType.S3).urls.values()
    assert all(u.startswith("s3://") for u in s3_urls)
