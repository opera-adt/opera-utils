from __future__ import annotations

import datetime
import itertools
import zipfile
from pathlib import Path

import numpy as np
import pytest

from opera_utils import missing_data


@pytest.fixture
def burst_ids():
    return ["t042_088913_iw1", "t042_088913_iw2", "t042_088913_iw3"]


@pytest.fixture
def dates():
    # Space every 12 dates
    num_dates = 5
    start = datetime.datetime(2020, 1, 1)
    return [start + datetime.timedelta(days=12 * i) for i in range(num_dates)]


@pytest.fixture
def B():
    return np.array(
        [
            [0, 1, 1, 1, 1],  # first burst ID
            [0, 1, 1, 1, 1],  # second burst ID
            [1, 0, 0, 1, 1],  # third burst ID
        ]
    ).astype(bool)


@pytest.fixture
def burst_id_date_tuples(burst_ids, dates, B):
    all_tuples = list(itertools.product(burst_ids, dates))
    # now remove the ones that are False in B
    tuples = [t for t, b in zip(all_tuples, B.flatten()) if b]
    assert len(tuples) == B.sum()
    return tuples


@pytest.fixture
def filenames(burst_id_date_tuples):
    return [
        f"{burst_id}_{date.strftime('%Y%m%d')}.h5"
        for burst_id, date in burst_id_date_tuples
    ]


def test_get_burst_id_to_dates(filenames, burst_id_date_tuples):
    # Check that it works with a list tuples
    # Check that it works with a list of filenames
    from collections import defaultdict

    expected = defaultdict(list)
    for burst_id, date in burst_id_date_tuples:
        expected[burst_id].append(date)

    out1 = missing_data.get_burst_id_to_dates(burst_id_date_tuples=burst_id_date_tuples)
    assert out1 == expected

    out2 = missing_data.get_burst_id_to_dates(slc_files=filenames)
    assert out2 == expected

    # check order of inputs does matter
    out3 = missing_data.get_burst_id_to_dates(
        burst_id_date_tuples=burst_id_date_tuples[::-1]
    )
    with pytest.raises(AssertionError):
        assert out3 == expected


def test_get_burst_id_date_incidence(B, filenames):
    B_out = missing_data.get_burst_id_date_incidence(
        burst_id_to_dates=missing_data.get_burst_id_to_dates(slc_files=filenames)
    )
    np.testing.assert_array_equal(B_out, B)


# What we want to check are 3 possible options
# 1. Use the first/second burst ID, discard date 1
#   Total number of bursts: (4 dates * 2 IDs) = 8
# 2. Use all three burst IDs, discard dates 1, 2, 3
#   Total number of bursts: (2 dates * 3 IDs) = 6
# 3. Use the third burst ID, discard dates 2, 3
#   Total number of bursts: (3 dates * 1 ID) = 3
def test_get_missing_data_options(filenames):
    mdos = missing_data.get_missing_data_options(slc_files=filenames)
    assert isinstance(mdos[0], missing_data.BurstSubsetOption)

    assert len(mdos) == 3
    mdo = mdos[0]
    assert mdo.num_dates == 4
    assert len(mdo.burst_ids) == 2
    assert mdo.num_burst_ids == 2
    assert mdo.total_num_bursts == 8
    assert mdo.num_candidate_bursts == len(filenames)

    mdo = mdos[1]
    assert mdo.num_dates == 2
    assert len(mdo.burst_ids) == 3
    assert mdo.num_burst_ids == 3
    assert mdo.total_num_bursts == 6
    assert mdo.num_candidate_bursts == len(filenames)

    mdo = mdos[2]
    assert mdo.num_dates == 3
    assert len(mdo.burst_ids) == 1
    assert mdo.num_burst_ids == 1
    assert mdo.total_num_bursts == 3
    assert mdo.num_candidate_bursts == len(filenames)


@pytest.fixture
def idaho_slc_list() -> list[str]:
    p = Path(__file__).parent / "data" / "idaho_slc_file_list.txt.zip"

    # unzip the file and return the list of strings
    with zipfile.ZipFile(p) as z:
        with z.open(z.namelist()[0]) as f:
            return f.read().decode().splitlines()


def test_get_missing_data_options_real(idaho_slc_list):
    burst_subset_options = missing_data.get_missing_data_options(idaho_slc_list)

    full_burst_id_list = (
        "t071_151161_iw1",
        "t071_151161_iw2",
        "t071_151161_iw3",
        "t071_151162_iw1",
        "t071_151162_iw2",
        "t071_151162_iw3",
        "t071_151163_iw1",
        "t071_151163_iw2",
        "t071_151163_iw3",
        "t071_151164_iw1",
        "t071_151164_iw2",
        "t071_151164_iw3",
        "t071_151165_iw1",
        "t071_151165_iw2",
        "t071_151165_iw3",
        "t071_151166_iw1",
        "t071_151166_iw2",
        "t071_151166_iw3",
        "t071_151167_iw1",
        "t071_151167_iw2",
        "t071_151167_iw3",
        "t071_151168_iw1",
        "t071_151168_iw2",
        "t071_151168_iw3",
        "t071_151169_iw1",
        "t071_151169_iw2",
        "t071_151169_iw3",
    )
    # The correct options should be
    expected_1 = full_burst_id_list[3:]
    expected_2 = full_burst_id_list[-3:]
    expected_3 = full_burst_id_list

    expected_id_lists = [expected_1, expected_2, expected_3]
    expected_num_dates = [173, 245, 11]
    expected_total_num_bursts = [4152, 735, 297]
    for i, option in enumerate(burst_subset_options):
        assert option.burst_ids == expected_id_lists[i]
        assert option.num_burst_ids == len(expected_id_lists[i])
        assert option.num_dates == expected_num_dates[i]
        assert option.total_num_bursts == expected_total_num_bursts[i]
