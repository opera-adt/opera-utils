import datetime
from pathlib import Path

import numpy as np
import pytest

from opera_utils._cslc import (
    get_cslc_orbit,
    get_lonlat_grid,
    get_orbit_arrays,
    get_radar_wavelength,
    get_xy_coords,
    get_zero_doppler_time,
    make_nodata_mask,
    parse_filename,
)

TEST_FILE = (
    Path(__file__).parent
    / "data"
    / "OPERA_L2_CSLC-S1_T087-185683-IW2_20221228T161651Z_20240504T181714Z_S1A_VV_v1.1.h5"
)


pytestmark = pytest.mark.filterwarnings(
    # h5py: `product` is deprecated as of NumPy 1.25.0, and will be removed in NumPy 2.0.
    # Please use `prod` instead.:DeprecationWarning:h5py/_hl/dataset.py
    "ignore:.*product.*:DeprecationWarning",
    "ignore:.*invalid value encountered.*:RuntimeWarning",
)


def test_file_regex():
    filename = (
        "OPERA_L2_CSLC-S1_T113-241377-IW2_20240716T105712Z_20240717T073255Z_S1A_VV_v1.1"
    )
    result = parse_filename(filename)
    expected = {
        "project": "OPERA",
        "level": "L2",
        "product_type": "CSLC-S1",
        "burst_id": "t113_241377_iw2",
        "start_datetime": datetime.datetime(
            2024, 7, 16, 10, 57, 12, tzinfo=datetime.timezone.utc
        ),
        "generation_datetime": datetime.datetime(
            2024, 7, 17, 7, 32, 55, tzinfo=datetime.timezone.utc
        ),
        "sensor": "S1A",
        "polarization": "VV",
        "product_version": "1.1",
    }
    assert result == expected

    filename = (
        "OPERA_L2_CSLC-S1_T056-123456-IW1_20230101T000000Z_20230102T235959Z_S1B_HH_v1.0"
    )
    expected = {
        "project": "OPERA",
        "level": "L2",
        "product_type": "CSLC-S1",
        "burst_id": "t056_123456_iw1",
        "start_datetime": datetime.datetime(
            2023, 1, 1, 0, 0, tzinfo=datetime.timezone.utc
        ),
        "generation_datetime": datetime.datetime(
            2023, 1, 2, 23, 59, 59, tzinfo=datetime.timezone.utc
        ),
        "sensor": "S1B",
        "polarization": "HH",
        "product_version": "1.0",
    }
    result = parse_filename(filename)
    assert result == expected


def test_compass_regex():
    filename = "t042_123456_iw2_20240102.h5"
    result = parse_filename(filename)
    expected = {
        "burst_id": "t042_123456_iw2",
        "start_datetime": datetime.datetime(
            2024, 1, 2, 0, 0, 0, tzinfo=datetime.timezone.utc
        ),
    }
    assert result == expected


def test_compressed_file_regex():
    filename = "OPERA_L2_COMPRESSED-CSLC-S1_T124-264305-IW1_20171209T000000Z_20170916T000000Z_20171209T000000Z_20250521T121130Z_VV_v1.0.h5"
    result = parse_filename(filename)
    expected = {
        "project": "OPERA",
        "level": "L2",
        "is_compressed": "COMPRESSED-",
        "product_type": "CSLC-S1",
        "burst_id": "t124_264305_iw1",
        "start_datetime": datetime.datetime(
            2017, 12, 9, 0, 0, tzinfo=datetime.timezone.utc
        ),
        "ministack_start_datetime": "20170916T000000Z",
        "ministack_stop_datetime": "20171209T000000Z",
        "generation_datetime": datetime.datetime(
            2025, 5, 21, 12, 11, 30, tzinfo=datetime.timezone.utc
        ),
        "polarization": "VV",
        "product_version": "1.0",
    }
    assert result == expected


def test_get_radar_wavelength():
    wvl = get_radar_wavelength(TEST_FILE)
    assert wvl == 0.05546576


def test_get_zero_doppler_time():
    ztd = get_zero_doppler_time(TEST_FILE)
    assert ztd == datetime.datetime(2022, 12, 28, 16, 16, 51, 13157)


def test_get_orbit_arrays():
    orbit_arrays = get_orbit_arrays(TEST_FILE)
    t = np.array(
        [
            172740.986843,
            172750.986843,
            172760.986843,
            172770.986843,
            172780.986843,
            172790.986843,
            172800.986843,
            172810.986843,
            172820.986843,
            172830.986843,
            172840.986843,
            172850.986843,
            172860.986843,
        ]
    )
    pos = np.array(
        [
            [-5694796.751419, -3249947.584436, 2652888.878987],
            [-5726813.662507, -3249421.07908, 2583994.542928],
            [-5758187.256021, -3248483.923562, 2514808.265927],
            [-5788913.413303, -3247137.16311, 2445337.863755],
            [-5818988.090509, -3245381.895235, 2375591.184622],
            [-5848407.319157, -3243219.269512, 2305576.10826],
            [-5877167.206674, -3240650.487339, 2235300.545014],
            [-5905263.936927, -3237676.801694, 2164772.434919],
            [-5932693.770745, -3234299.51689, 2093999.746773],
            [-5959453.04643, -3230519.988302, 2022990.477211],
            [-5985538.180255, -3226339.622115, 1951752.649764],
            [-6010945.666951, -3221759.875032, 1880294.313926],
            [-6035672.080179, -3216782.253999, 1808623.544221],
        ]
    )
    vel = np.array(
        [
            [-3233.717848, 32.084506, -6874.576863],
            [-3169.594503, 73.200035, -6904.160643],
            [-3105.05557, 114.213663, -6932.964498],
            [-3040.108502, 155.120151, -6960.985138],
            [-2974.760808, 195.914281, -6988.219365],
            [-2909.020054, 236.590857, -7014.664073],
            [-2842.893855, 277.144711, -7040.316246],
            [-2776.389883, 317.570695, -7065.172962],
            [-2709.51586, 357.863691, -7089.231391],
            [-2642.279557, 398.018603, -7112.488797],
            [-2574.688796, 438.030367, -7134.942537],
            [-2506.751447, 477.893945, -7156.59006],
            [-2438.475426, 517.604327, -7177.42891],
        ]
    )
    ref_epoch = datetime.datetime(2022, 12, 26, 16, 16, 51, 13157)
    np.testing.assert_equal(orbit_arrays[0], t)
    np.testing.assert_equal(orbit_arrays[1], pos)
    np.testing.assert_equal(orbit_arrays[2], vel)
    assert orbit_arrays[3] == ref_epoch


# Skip if isce3 not installed
try:
    import isce3
except ImportError:
    isce3 = None


@pytest.mark.skipif(isce3 is None, reason="isce3 not installed")
def test_get_cslc_orbit():
    orbit = get_cslc_orbit(TEST_FILE)
    assert isinstance(orbit, isce3.core.Orbit)

    assert orbit.size == 13
    expected = datetime.datetime(2022, 12, 28, 16, 15, 52, 0)
    # Ignore the nanosecond:
    assert orbit.start_datetime.isoformat()[:-3] == expected.isoformat(
        timespec="microseconds"
    )
    assert orbit.mid_time == 60.0
    assert orbit.spacing == 10

    orbit_arrays = get_orbit_arrays(TEST_FILE)
    # orbit.time is a "linspace"
    np.testing.assert_array_equal(orbit.position, orbit_arrays[1])
    np.testing.assert_array_equal(orbit.velocity, orbit_arrays[2])


def test_get_lonlat_grid():
    lons, lats = get_lonlat_grid(TEST_FILE)
    assert lons.shape == (50, 207)
    assert lons.dtype == np.float64
    assert lons[0, 0] == -156.2666837419528
    assert lats.shape == (50, 207)
    assert lats.dtype == np.float64
    assert lats[0, 0] == 19.616218372288117
    assert lons[0, 0] != lons[-1, 0]


def test_get_xy_coords():
    x, y, epsg = get_xy_coords(TEST_FILE)

    assert x.shape == (207,)
    assert x.dtype == np.float64
    assert x[0] == 157322.5
    assert x[-1] == 260322.5

    assert y.shape == (50,)
    assert y.dtype == np.float64
    assert y[0] == 2172295.0
    assert y[-1] == 2123295.0

    assert epsg == 32605


def test_make_nodata_mask(tmp_path):
    out_file = tmp_path / "mask.tif"
    make_nodata_mask([TEST_FILE], out_file=out_file, buffer_pixels=100)
    from osgeo import gdal

    ds = gdal.Open(str(out_file))
    bnd = ds.GetRasterBand(1)
    assert bnd.DataType == gdal.GDT_Byte
    data = bnd.ReadAsArray()
    assert data.sum() > 0
