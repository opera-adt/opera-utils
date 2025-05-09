from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from opera_utils.disp._product import DispProduct, UrlType
from opera_utils.disp._search import search


# Sample CMR response JSON that would be returned by CMR search
@pytest.fixture
def cmr_response_json():
    return {
        "items": [
            {
                "meta": {"concept-id": "C1234-OPERA"},
                "umm": {
                    "RelatedUrls": [
                        {
                            "URL": (
                                "https://datapool.asf.alaska.edu/DISP/OPERA-S1/OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v1.0_20250318T222753Z.nc"
                            ),
                            "Type": "GET DATA",
                        },
                        {
                            "URL": (
                                "s3://opera-staging/OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v1.0_20250318T222753Z.nc"
                            ),
                            "Type": "GET DATA VIA DIRECT ACCESS",
                        },
                    ],
                    "AdditionalAttributes": [
                        {"Name": "FRAME_NUMBER", "Values": ["11116"]},
                        {"Name": "PRODUCT_VERSION", "Values": ["1.0"]},
                    ],
                    "TemporalExtent": {
                        "RangeDateTime": {
                            "BeginningDateTime": "2016-07-05T14:07:55.000Z",
                            "EndingDateTime": "2016-07-29T14:07:56.000Z",
                        }
                    },
                },
            },
            {
                "meta": {"concept-id": "C1235-OPERA"},
                "umm": {
                    "RelatedUrls": [
                        {
                            "URL": (
                                "https://datapool.asf.alaska.edu/DISP/OPERA-S1/OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160810T140756Z_v1.0_20250318T222753Z.nc"
                            ),
                            "Type": "GET DATA",
                        },
                        {
                            "URL": (
                                "s3://opera-staging/OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160810T140756Z_v1.0_20250318T222753Z.nc"
                            ),
                            "Type": "GET DATA VIA DIRECT ACCESS",
                        },
                    ],
                    "AdditionalAttributes": [
                        {"Name": "FRAME_NUMBER", "Values": ["11116"]},
                        {"Name": "PRODUCT_VERSION", "Values": ["1.0"]},
                    ],
                    "TemporalExtent": {
                        "RangeDateTime": {
                            "BeginningDateTime": "2016-07-05T14:07:55.000Z",
                            "EndingDateTime": "2016-08-10T14:07:56.000Z",
                        }
                    },
                },
            },
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


def test_search_with_mock_response(cmr_response_json):
    with patch("requests.get") as mock_get:
        # Set up the mock response
        mock_get.return_value = MockResponse(cmr_response_json)

        # Call the search function
        products = search(frame_id=11116)

        # Verify correct URL and parameters were used
        args, kwargs = mock_get.call_args
        assert args[0] == "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
        assert kwargs["params"]["short_name"] == "OPERA_L3_DISP-S1_V1"
        assert kwargs["params"]["page_size"] == 2000
        assert "int,FRAME_NUMBER,11116" in kwargs["params"]["attribute[]"]
        assert "float,PRODUCT_VERSION,1.0" in kwargs["params"]["attribute[]"]

        # Check results are correct
        assert len(products) == 2
        assert all(isinstance(p, DispProduct) for p in products)
        assert products[0].frame_id == 11116
        assert products[0].version == "1.0"
        assert products[0].reference_datetime == datetime(
            2016, 7, 5, 14, 7, 55, tzinfo=timezone.utc
        )
        assert products[0].secondary_datetime == datetime(
            2016, 7, 29, 14, 7, 56, tzinfo=timezone.utc
        )

        # Check sorting by secondary date
        assert products[0].secondary_datetime < products[1].secondary_datetime


def test_search_with_url_type(cmr_response_json):
    with patch("requests.get") as mock_get:
        mock_get.return_value = MockResponse(cmr_response_json)

        # Test with HTTPS URL type
        products = search(frame_id=11116, url_type=UrlType.HTTPS)
        assert products[0].filename.startswith("https://")

        # Test with S3 URL type
        products = search(frame_id=11116, url_type=UrlType.S3)
        assert products[0].filename.startswith("s3://")


def test_search_with_uat(cmr_response_json):
    with patch("requests.get") as mock_get:
        mock_get.return_value = MockResponse(cmr_response_json)

        # Call search with use_uat=True
        _products = search(frame_id=11116, use_uat=True)

        # Check correct URL was used
        args, _ = mock_get.call_args
        assert args[0] == "https://cmr.uat.earthdata.nasa.gov/search/granules.umm_json"
