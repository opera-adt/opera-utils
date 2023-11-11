import pytest

from opera_utils import datasets


@pytest.mark.vcr
def test_registry():
    # Check that the registry is up to date
    p = datasets.POOCH
    assert len(p.registry_files) == 4
    assert (
        p.base_url == "https://github.com/opera-adt/burst_db/releases/download/v0.3.1/"
    )
    for f in p.registry_files:
        assert p.is_available(f)
