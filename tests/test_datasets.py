import pytest

from opera_utils import datasets


# XXX remove this ones release for https://github.com/kevin1024/vcrpy/issues/888 is out
@pytest.fixture(autouse=True)
def patch_VCRHTTPResponse_version_string():
    from vcr.stubs import VCRHTTPResponse

    if not hasattr(VCRHTTPResponse, "version_string"):
        VCRHTTPResponse.version_string = None


@pytest.fixture(scope="module")
def vcr_config():
    return {
        # Replace the Authorization request header with "DUMMY" in cassettes
        "filter_headers": [("Authorization", "DUMMY")]
    }


@pytest.mark.vcr
def test_registry():
    # Check that the registry is up to date
    p = datasets.POOCH
    assert len(p.registry_files) == 4
    assert (
        p.base_url == "https://github.com/opera-adt/burst_db/releases/download/v0.9.0/"
    )
    for f in p.registry_files:
        assert p.is_available(f)
