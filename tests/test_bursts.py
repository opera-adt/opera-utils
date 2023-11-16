from __future__ import annotations

import random
from pathlib import Path

import pytest

from opera_utils import filter_by_burst_id, get_burst_id, group_by_burst
from opera_utils._helpers import flatten


def test_get_burst_id():
    assert (
        get_burst_id("t087_185678_iw2/20180210/t087_185678_iw2_20180210.h5")
        == "t087_185678_iw2"
    )
    # Check the official naming convention
    fn = "OPERA_L2_CSLC-S1_T087-185678-IW2_20180210T232711Z_20230101T100506Z_S1A_VV_v1.0.h5"  # noqa
    assert get_burst_id(fn) == "t087_185678_iw2"


def test_group_by_burst():
    expected = {
        "t087_185678_iw2": [
            Path("t087_185678_iw2/20180210/t087_185678_iw2_20180210.h5"),
            Path("t087_185678_iw2/20180318/t087_185678_iw2_20180318.h5"),
            Path("t087_185678_iw2/20180423/t087_185678_iw2_20180423.h5"),
        ],
        "t087_185678_iw3": [
            Path("t087_185678_iw3/20180210/t087_185678_iw3_20180210.h5"),
            Path("t087_185678_iw3/20180318/t087_185678_iw3_20180318.h5"),
            Path("t087_185678_iw3/20180517/t087_185678_iw3_20180517.h5"),
        ],
        "t087_185679_iw1": [
            Path("t087_185679_iw1/20180210/t087_185679_iw1_20180210.h5"),
            Path("t087_185679_iw1/20180318/t087_185679_iw1_20180318.h5"),
        ],
    }
    in_files = list(flatten(expected.values()))

    assert group_by_burst(in_files) == expected

    # Any order should work
    random.shuffle(in_files)
    # but the order of the lists of each key may be different
    for burst, file_list in group_by_burst(in_files).items():
        assert sorted(file_list) == sorted(expected[burst])


def test_group_by_burst_product_version():
    # Should also match this:
    # OPERA_L2_CSLC-S1_T078-165495-IW3_20190906T232711Z_20230101T100506Z_S1A_VV_v1.0.h5
    base = "OPERA_L2_CSLC-S1_"
    ending = "20230101T100506Z_S1A_VV_v1.0.h5"
    expected = {
        "t087_185678_iw2": [
            Path(f"{base}_T087-185678-IW2_20180210T232711Z_{ending}"),
            Path(f"{base}_T087-185678-IW2_20180318T232711Z_{ending}"),
            Path(f"{base}_T087-185678-IW2_20180423T232711Z_{ending}"),
        ],
        "t087_185678_iw3": [
            Path(f"{base}_T087-185678-IW3_20180210T232711Z_{ending}"),
            Path(f"{base}_T087-185678-IW3_20180318T232711Z_{ending}"),
            Path(f"{base}_T087-185678-IW3_20180517T232711Z_{ending}"),
        ],
        "t087_185679_iw1": [
            Path(f"{base}_T087-185679-IW1_20180210T232711Z_{ending}"),
            Path(f"{base}_T087-185679-IW1_20180318T232711Z_{ending}"),
        ],
    }
    in_files = list(flatten(expected.values()))

    assert group_by_burst(in_files) == expected


def test_group_by_burst_non_opera():
    with pytest.raises(ValueError, match="Could not parse burst id"):
        group_by_burst(["20200101.slc", "20200202.slc"])
        # A combination should still error
        group_by_burst(
            [
                "20200101.slc",
                Path("t087_185679_iw1/20180210/t087_185679_iw1_20180210_VV.h5"),
            ]
        )


def test_filter_by_burst_id():
    burst_id = "t087_185678_iw2"
    in_files = [
        Path("t087_185678_iw1/20180210/t087_185678_iw1_20180210.h5"),
        Path("t087_185678_iw2/20180318/t087_185678_iw2_20180318.h5"),
        Path("t087_185678_iw3/20180423/t087_185678_iw3_20180423.h5"),
    ]
    expected = [in_files[1]]
    assert filter_by_burst_id(in_files, burst_id) == expected

    # Multiple burst ids
    burst_ids = ["t087_185678_iw2", "t087_185678_iw3"]
    expected = [in_files[1], in_files[2]]
    assert filter_by_burst_id(in_files, burst_ids) == expected

    # Any order should work
    expected = [in_files[1]]
    random.shuffle(in_files)
    assert filter_by_burst_id(in_files, burst_id) == expected
