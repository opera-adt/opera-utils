from datetime import datetime

import asf_search as asf
import pytest

from opera_utils.download import L2Product, filter_results_by_date_and_version


# XXX remove this ones release for https://github.com/kevin1024/vcrpy/issues/888 is out
@pytest.fixture(autouse=True)
def patch_VCRHTTPResponse_version_string():
    from vcr.stubs import VCRHTTPResponse

    if not hasattr(VCRHTTPResponse, "version_string"):
        VCRHTTPResponse.version_string = None


@pytest.mark.filterwarnings(
    "ignore:Parsing dates involving a day of month without a year"
    " specified.*:DeprecationWarning"
)
@pytest.mark.vcr
def test_download_filter():
    burst_ids = ["t087_185683_iw2", "t087_185682_iw2"]
    start, end = datetime(2022, 10, 1), datetime(2023, 3, 29)
    product = L2Product.CSLC
    results = asf.search(
        operaBurstID=list(burst_ids),
        processingLevel=product.value,
        start=start,
        end=end,
    )
    assert len(results) == 60
    assert len(filter_results_by_date_and_version(results)) == 30
