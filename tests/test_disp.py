from datetime import datetime
from pathlib import Path

import pytest

from opera_utils.disp._product import DispProduct


@pytest.fixture
def urls():
    return [
        "https://datapool-test.asf.alaska.edu/DISP/OPERA-S1/OPERA_L3_DISP-S1_IW_F08622_VV_20160716T225042Z_20160926T225046Z_v1.1_20250215T011840Z.nc",
        "https://datapool-test.asf.alaska.edu/DISP/OPERA-S1/OPERA_L3_DISP-S1_IW_F08622_VV_20160926T225046Z_20161113T225046Z_v1.1_20250215T011840Z.nc",
        "https://datapool-test.asf.alaska.edu/DISP/OPERA-S1/OPERA_L3_DISP-S1_IW_F08622_VV_20160926T225046Z_20161020T225046Z_v1.1_20250215T011840Z.nc",
    ]


@pytest.fixture
def names(urls):
    return [Path(url).name for url in urls]


def test_disp_product(names):
    products = [DispProduct.from_filename(name) for name in names]
    assert all(p.frame_id == 8622 for p in products)
    assert all(p.version == "1.1" for p in products)
    assert all(
        p.generation_datetime == datetime(2025, 2, 15, 1, 18, 40) for p in products
    )

    assert products[0].reference_datetime == datetime(2016, 7, 16, 22, 50, 42)
    assert products[0].secondary_datetime == datetime(2016, 9, 26, 22, 50, 46)

    assert products[2].reference_datetime == datetime(2016, 9, 26, 22, 50, 46)
    assert products[2].secondary_datetime == datetime(2016, 10, 20, 22, 50, 46)
