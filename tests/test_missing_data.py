import datetime
import itertools
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
    start = datetime.date(2020, 1, 1)
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
    missing_data.get_burst_id_date_incidence


# What we want to check are 3 possible options
def test_get_missing_data_options(B, filenames):
    missing_data.get_missing_data_options


def test_get_dates():
    assert missing_data.get_dates("20200303_20210101.int") == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]

    assert missing_data.get_dates("20200303.slc")[0] == datetime.date(2020, 3, 3)
    assert missing_data.get_dates(Path("20200303.slc"))[0] == datetime.date(2020, 3, 3)
    # Check that it's the filename, not the path
    assert missing_data.get_dates(Path("/usr/19990101/asdf20200303.tif"))[
        0
    ] == datetime.date(2020, 3, 3)
    assert missing_data.get_dates("/usr/19990101/asdf20200303.tif")[0] == datetime.date(
        2020, 3, 3
    )

    assert missing_data.get_dates("/usr/19990101/20200303_20210101.int") == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]

    assert missing_data.get_dates("/usr/19990101/notadate.tif") == []


def test_get_dates_with_format():
    # try other date formats
    fmt = "%Y-%m-%d"
    assert missing_data.get_dates("2020-03-03_2021-01-01.int", fmt) == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]


def test_get_dates_with_gdal_string():
    # Checks that is can parse 'NETCDF:"/path/to/file.nc":variable'
    assert missing_data.get_dates(
        'NETCDF:"/usr/19990101/20200303_20210101.nc":variable'
    ) == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]
    assert missing_data.get_dates(
        'NETCDF:"/usr/19990101/20200303_20210101.nc":"//variable/2"'
    ) == [
        datetime.date(2020, 3, 3),
        datetime.date(2021, 1, 1),
    ]
    # Check the derived dataset name too
    assert missing_data.get_dates(
        'DERIVED_SUBDATASET:AMPLITUDE:"/usr/19990101/20200303_20210101.int"'
    ) == [datetime.date(2020, 3, 3), datetime.date(2021, 1, 1)]
