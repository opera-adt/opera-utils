"""Tests for opera_utils.nisar._search and _gunw_search modules."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from opera_utils.constants import UrlType
from opera_utils.nisar._gunw_search import search_gunw
from opera_utils.nisar._product import GslcProduct, GunwProduct
from opera_utils.nisar._search import search

# Two valid NISAR GSLC filenames (different cycles, same track/frame)
FILE_1 = "NISAR_L2_PR_GSLC_004_076_A_022_2005_QPDH_A_20251103T110514_20251103T110549_X05007_N_F_J_001.h5"
FILE_2 = "NISAR_L2_PR_GSLC_005_076_A_022_2005_QPDH_A_20251115T110514_20251115T110549_X05008_N_F_J_001.h5"
# Different orbit direction
FILE_DESC = "NISAR_L2_PR_GSLC_004_076_D_022_2005_QPDH_A_20251103T110514_20251103T110549_X05007_N_F_J_001.h5"


def _make_umm_item(filename: str, protocol: str = "https") -> dict:
    """Build a minimal CMR UMM item dict for the given filename."""
    if protocol == "https":
        url = f"https://datapool.asf.alaska.edu/GSLC/NISAR/{filename}"
    else:
        url = f"s3://nisar-gslc-bucket/{filename}"
    return {
        "umm": {
            "RelatedUrls": [
                {"URL": url, "Type": "GET DATA"},
            ],
            "DataGranule": {
                "ArchiveAndDistributionInformation": [{"SizeInBytes": 500_000_000}]
            },
        }
    }


@pytest.fixture
def cmr_response_json():
    """A single-page CMR response with two ascending products."""
    return {
        "items": [
            _make_umm_item(FILE_1),
            _make_umm_item(FILE_2),
        ]
    }


class MockResponse:
    def __init__(self, json_data, status_code=200, headers=None):
        self.json_data = json_data
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code != 200:
            msg = f"Status code {self.status_code}"
            raise ValueError(msg)


class TestSearch:
    def test_basic_search(self, cmr_response_json):
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(cmr_response_json)
            products = search(
                bbox=(40.0, 13.0, 41.0, 14.0),
                relative_orbit_number=76,
            )

        assert len(products) == 2
        assert all(isinstance(p, GslcProduct) for p in products)

        # Check correct URL was called
        args, kwargs = mock_get.call_args
        assert args[0] == "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
        assert kwargs["params"]["short_name"] == "NISAR_L2_GSLC_BETA_V1"
        assert kwargs["params"]["page_size"] == 500
        assert kwargs["params"]["bounding_box"] == "40.0,13.0,41.0,14.0"

    def test_results_sorted_by_track_frame_and_date(self, cmr_response_json):
        """Results should be sorted by (track_frame_id, start_datetime)."""
        # Reverse the items so they arrive out of order
        cmr_response_json["items"].reverse()
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(cmr_response_json)
            products = search(bbox=(40.0, 13.0, 41.0, 14.0))

        assert products[0].start_datetime < products[1].start_datetime

    def test_search_with_url_type_s3(self):
        response = {
            "items": [
                _make_umm_item(FILE_1, protocol="s3"),
            ]
        }
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(response)
            products = search(bbox=(40.0, 13.0, 41.0, 14.0), url_type=UrlType.S3)

        assert len(products) == 1
        assert str(products[0].filename).startswith("s3://")

    def test_filter_by_track_frame(self, cmr_response_json):
        """track_frame matches all products for a geographic footprint (across cycles)."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(cmr_response_json)
            products = search(
                bbox=(40.0, 13.0, 41.0, 14.0),
                track_frame="076_A_022",
            )

        # Both FILE_1 (cycle 4) and FILE_2 (cycle 5) share the same track_frame_id
        assert len(products) == 2

    def test_filter_by_orbit_direction(self):
        """Descending products filtered out when orbit_direction='A'."""
        response = {
            "items": [
                _make_umm_item(FILE_1),
                _make_umm_item(FILE_DESC),
            ]
        }
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(response)
            products = search(bbox=(40.0, 13.0, 41.0, 14.0), orbit_direction="A")

        assert len(products) == 1
        assert str(products[0].orbit_direction) == "A"

    def test_filter_by_temporal_range(self, cmr_response_json):
        """Only products within the datetime range are returned."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(cmr_response_json)
            products = search(
                bbox=(40.0, 13.0, 41.0, 14.0),
                start_datetime=datetime(2025, 11, 1, tzinfo=timezone.utc),
                end_datetime=datetime(2025, 11, 10, tzinfo=timezone.utc),
            )

        # Only FILE_1 (Nov 3) is in range; FILE_2 (Nov 15) is out
        assert len(products) == 1
        assert products[0].cycle_number == 4

    def test_pagination(self):
        """CMR-Search-After header triggers additional pages."""
        page1_response = MockResponse(
            {"items": [_make_umm_item(FILE_1)]},
            headers={"CMR-Search-After": "cursor_token"},
        )
        page2_response = MockResponse(
            {"items": [_make_umm_item(FILE_2)]},
        )
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.side_effect = [page1_response, page2_response]
            products = search(bbox=(40.0, 13.0, 41.0, 14.0))

        assert mock_get.call_count == 2
        assert len(products) == 2
        # Second call should include the CMR-Search-After header
        _, second_kwargs = mock_get.call_args_list[1]
        assert second_kwargs["headers"]["CMR-Search-After"] == "cursor_token"

    def test_no_constraints_warning(self):
        """Warns when no spatial or orbit constraints are specified."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse({"items": []})
            with pytest.warns(UserWarning, match="search may be large"):
                search()

    def test_empty_results(self):
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse({"items": []})
            products = search(bbox=(0.0, 0.0, 1.0, 1.0))

        assert products == []

    def test_attribute_filters_in_params(self):
        """CMR attribute filters are built for orbit parameters."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse({"items": []})
            search(
                relative_orbit_number=76,
                track_frame_number=22,
                orbit_direction="A",
            )

        _, kwargs = mock_get.call_args
        attrs = kwargs["params"]["attribute[]"]
        assert "int,TRACK_NUMBER,76" in attrs
        assert "int,FRAME_NUMBER,22" in attrs
        assert "string,ASCENDING_DESCENDING,ASCENDING" in attrs

    def test_track_frame_pushes_cmr_filters(self):
        """track_frame string is parsed into CMR attribute filters."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse({"items": []})
            search(track_frame="076_A_022")

        _, kwargs = mock_get.call_args
        attrs = kwargs["params"]["attribute[]"]
        assert "int,TRACK_NUMBER,76" in attrs
        assert "int,FRAME_NUMBER,22" in attrs
        assert "string,ASCENDING_DESCENDING,ASCENDING" in attrs

    def test_temporal_param_in_request(self):
        """Start/end datetime is sent to CMR as temporal parameter."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse({"items": []})
            search(
                bbox=(0, 0, 1, 1),
                start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
                end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
            )

        _, kwargs = mock_get.call_args
        assert "temporal" in kwargs["params"]

    def test_skips_unparseable_granules(self):
        """Granules that fail to parse are silently skipped."""
        response = {
            "items": [
                {
                    "umm": {
                        "RelatedUrls": [
                            {"URL": "https://bad_name.h5", "Type": "GET DATA"}
                        ]
                    }
                },
                _make_umm_item(FILE_1),
            ]
        }
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(response)
            products = search(bbox=(40.0, 13.0, 41.0, 14.0))

        assert len(products) == 1


# ---------------------------------------------------------------------------
# NISAR GUNW search tests
# ---------------------------------------------------------------------------

# Two valid NISAR GUNW filenames — same track/frame, different secondary cycles
# Format: NISAR_L2_PR_GUNW_<ref_cycle>_<rel_orbit>_<dir>_<frame>_<sec_cycle>_
#         <scene>_<pol>_<ref_start>_<ref_end>_<sec_start>_<sec_end>_<crid>_<f1>_<f2>_<f3>_<counter>.h5
GUNW_1 = (
    "NISAR_L2_PR_GUNW_004_076_A_022_005_4000_SH_"
    "20251103T110514_20251103T110549_"
    "20251115T110514_20251115T110549_"
    "X05010_N_P_J_001.h5"
)
GUNW_2 = (
    "NISAR_L2_PR_GUNW_004_076_A_022_006_4000_SH_"
    "20251103T110514_20251103T110549_"
    "20251127T110514_20251127T110549_"
    "X05011_N_P_J_001.h5"
)
# Same track/frame but descending direction
GUNW_DESC = (
    "NISAR_L2_PR_GUNW_004_076_D_022_005_4000_SH_"
    "20251103T110514_20251103T110549_"
    "20251115T110514_20251115T110549_"
    "X05010_N_P_J_001.h5"
)
# Different reference cycle (ref=003 instead of 004)
GUNW_OTHER_REF = (
    "NISAR_L2_PR_GUNW_003_076_A_022_005_4000_SH_"
    "20251103T110514_20251103T110549_"
    "20251115T110514_20251115T110549_"
    "X05010_N_P_J_001.h5"
)


def _make_gunw_umm_item(filename: str, protocol: str = "https") -> dict:
    """Build a minimal CMR UMM item dict for the given GUNW filename."""
    if protocol == "https":
        url = f"https://datapool.asf.alaska.edu/GUNW/NISAR/{filename}"
    else:
        url = f"s3://nisar-gunw-bucket/{filename}"
    return {
        "umm": {
            "RelatedUrls": [
                {"URL": url, "Type": "GET DATA"},
            ],
            "DataGranule": {
                "ArchiveAndDistributionInformation": [{"SizeInBytes": 1_000_000_000}]
            },
        }
    }


@pytest.fixture
def gunw_cmr_response_json():
    """A single-page CMR response with two ascending GUNW products."""
    return {
        "items": [
            _make_gunw_umm_item(GUNW_1),
            _make_gunw_umm_item(GUNW_2),
        ]
    }


class TestSearchGunw:
    def test_basic_search(self, gunw_cmr_response_json):
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(gunw_cmr_response_json)
            products = search_gunw(
                bbox=(40.0, 13.0, 41.0, 14.0),
                relative_orbit_number=76,
            )

        assert len(products) == 2
        assert all(isinstance(p, GunwProduct) for p in products)

        args, kwargs = mock_get.call_args
        assert args[0] == "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
        assert kwargs["params"]["short_name"] == "NISAR_L2_GUNW_BETA_V1"
        assert kwargs["params"]["page_size"] == 500
        assert kwargs["params"]["bounding_box"] == "40.0,13.0,41.0,14.0"

    def test_results_sorted_by_track_frame_and_secondary_date(
        self, gunw_cmr_response_json
    ):
        """Results should be sorted by (track_frame_id, secondary_datetime)."""
        gunw_cmr_response_json["items"].reverse()
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(gunw_cmr_response_json)
            products = search_gunw(bbox=(40.0, 13.0, 41.0, 14.0))

        assert products[0].secondary_datetime < products[1].secondary_datetime

    def test_search_with_url_type_s3(self):
        response = {"items": [_make_gunw_umm_item(GUNW_1, protocol="s3")]}
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(response)
            products = search_gunw(bbox=(40.0, 13.0, 41.0, 14.0), url_type=UrlType.S3)

        assert len(products) == 1
        assert str(products[0].filename).startswith("s3://")

    def test_filter_by_track_frame(self, gunw_cmr_response_json):
        """track_frame matches all products for a geographic footprint."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(gunw_cmr_response_json)
            products = search_gunw(
                bbox=(40.0, 13.0, 41.0, 14.0),
                track_frame="076_A_022",
            )

        assert len(products) == 2
        assert all(p.track_frame_id == "076_A_022" for p in products)

    def test_filter_by_orbit_direction(self):
        """Descending products are filtered out when orbit_direction='A'."""
        response = {
            "items": [
                _make_gunw_umm_item(GUNW_1),
                _make_gunw_umm_item(GUNW_DESC),
            ]
        }
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(response)
            products = search_gunw(bbox=(40.0, 13.0, 41.0, 14.0), orbit_direction="A")

        assert len(products) == 1
        assert str(products[0].orbit_direction) == "A"

    def test_filter_by_temporal_range(self, gunw_cmr_response_json):
        """Only products whose secondary date falls within the range are returned."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(gunw_cmr_response_json)
            products = search_gunw(
                bbox=(40.0, 13.0, 41.0, 14.0),
                start_datetime=datetime(2025, 11, 1, tzinfo=timezone.utc),
                end_datetime=datetime(2025, 11, 20, tzinfo=timezone.utc),
            )

        # GUNW_1 secondary=Nov 15 is in range; GUNW_2 secondary=Nov 27 is out
        assert len(products) == 1
        assert products[0].secondary_cycle_number == 5

    def test_filter_by_reference_cycle_number(self):
        """Only products with matching reference_cycle_number are returned."""
        response = {
            "items": [
                _make_gunw_umm_item(GUNW_1),  # ref_cycle=4
                _make_gunw_umm_item(GUNW_OTHER_REF),  # ref_cycle=3
            ]
        }
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(response)
            products = search_gunw(
                bbox=(40.0, 13.0, 41.0, 14.0),
                reference_cycle_number=4,
            )

        assert len(products) == 1
        assert products[0].reference_cycle_number == 4

    def test_filter_by_secondary_cycle_number(self, gunw_cmr_response_json):
        """Only products with matching secondary_cycle_number are returned."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(gunw_cmr_response_json)
            products = search_gunw(
                bbox=(40.0, 13.0, 41.0, 14.0),
                secondary_cycle_number=5,
            )

        assert len(products) == 1
        assert products[0].secondary_cycle_number == 5

    def test_pagination(self):
        """CMR-Search-After header triggers additional pages."""
        page1 = MockResponse(
            {"items": [_make_gunw_umm_item(GUNW_1)]},
            headers={"CMR-Search-After": "cursor_token"},
        )
        page2 = MockResponse({"items": [_make_gunw_umm_item(GUNW_2)]})
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.side_effect = [page1, page2]
            products = search_gunw(bbox=(40.0, 13.0, 41.0, 14.0))

        assert mock_get.call_count == 2
        assert len(products) == 2
        _, second_kwargs = mock_get.call_args_list[1]
        assert second_kwargs["headers"]["CMR-Search-After"] == "cursor_token"

    def test_no_constraints_warning(self):
        """Warns when no spatial or orbit constraints are specified."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse({"items": []})
            with pytest.warns(UserWarning, match="search may be large"):
                search_gunw()

    def test_empty_results(self):
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse({"items": []})
            products = search_gunw(bbox=(0.0, 0.0, 1.0, 1.0))

        assert products == []

    def test_attribute_filters_in_params(self):
        """CMR attribute filters are built for orbit parameters."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse({"items": []})
            search_gunw(
                relative_orbit_number=76,
                track_frame_number=22,
                orbit_direction="A",
            )

        _, kwargs = mock_get.call_args
        attrs = kwargs["params"]["attribute[]"]
        assert "int,TRACK_NUMBER,76" in attrs
        assert "int,FRAME_NUMBER,22" in attrs
        assert "string,ASCENDING_DESCENDING,ASCENDING" in attrs

    def test_track_frame_pushes_cmr_filters(self):
        """track_frame string is parsed into CMR attribute filters."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse({"items": []})
            search_gunw(track_frame="076_A_022")

        _, kwargs = mock_get.call_args
        attrs = kwargs["params"]["attribute[]"]
        assert "int,TRACK_NUMBER,76" in attrs
        assert "int,FRAME_NUMBER,22" in attrs
        assert "string,ASCENDING_DESCENDING,ASCENDING" in attrs

    def test_descending_direction_attribute_filter(self):
        """Descending orbit direction maps to DESCENDING in attribute filter."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse({"items": []})
            search_gunw(relative_orbit_number=76, orbit_direction="D")

        _, kwargs = mock_get.call_args
        attrs = kwargs["params"]["attribute[]"]
        assert "string,ASCENDING_DESCENDING,DESCENDING" in attrs

    def test_temporal_param_in_request(self):
        """Start/end datetime is sent to CMR as temporal parameter."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse({"items": []})
            search_gunw(
                bbox=(0, 0, 1, 1),
                start_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
                end_datetime=datetime(2025, 12, 31, tzinfo=timezone.utc),
            )

        _, kwargs = mock_get.call_args
        assert "temporal" in kwargs["params"]

    def test_skips_unparseable_granules(self):
        """Granules that fail to parse are silently skipped."""
        response = {
            "items": [
                {
                    "umm": {
                        "RelatedUrls": [
                            {"URL": "https://bad_name.h5", "Type": "GET DATA"}
                        ]
                    }
                },
                _make_gunw_umm_item(GUNW_1),
            ]
        }
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(response)
            products = search_gunw(bbox=(40.0, 13.0, 41.0, 14.0))

        assert len(products) == 1

    def test_gunw_product_pair_id(self, gunw_cmr_response_json):
        """GunwProduct.pair_id gives YYYYMMDD_YYYYMMDD (reference_secondary)."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(gunw_cmr_response_json)
            products = search_gunw(bbox=(40.0, 13.0, 41.0, 14.0))

        assert products[0].pair_id == "20251103_20251115"
        assert products[1].pair_id == "20251103_20251127"

    def test_size_in_bytes_populated(self, gunw_cmr_response_json):
        """size_in_bytes is set from ArchiveAndDistributionInformation."""
        with patch("opera_utils._cmr.requests.get") as mock_get:
            mock_get.return_value = MockResponse(gunw_cmr_response_json)
            products = search_gunw(bbox=(40.0, 13.0, 41.0, 14.0))

        assert all(p.size_in_bytes == 1_000_000_000 for p in products)
