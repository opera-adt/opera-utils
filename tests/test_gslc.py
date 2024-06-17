import datetime
from pathlib import Path

import numpy as np
import pytest

from opera_utils._gslc import (
    get_cslc_orbit,
    get_lonlat_grid,
    get_orbit_arrays,
    get_radar_wavelength,
    get_xy_coords,
    get_zero_doppler_time,
)

TEST_FILE = (
    Path(__file__).parent
    / "data"
    / "OPERA_L2_CSLC-S1_T042-088905-IW1_20231009T140757Z_20231010T204936Z_S1A_VV_v1.0.h5"
)


pytestmark = pytest.mark.filterwarnings(
    # h5py: `product` is deprecated as of NumPy 1.25.0, and will be removed in NumPy 2.0.
    # Please use `prod` instead.:DeprecationWarning:h5py/_hl/dataset.py
    "ignore:.*product.*:DeprecationWarning",
)


def test_get_radar_wavelength():
    wvl = get_radar_wavelength(TEST_FILE)
    assert wvl == 0.05546576


def test_get_zero_doppler_time():
    ztd = get_zero_doppler_time(TEST_FILE)
    assert ztd == datetime.datetime(2023, 10, 9, 14, 7, 57, 208273)


def test_get_orbit_arrays():
    orbit_arrays = get_orbit_arrays(TEST_FILE)
    t = np.array(
        [
            172744.69566,
            172754.69566,
            172764.69566,
            172774.69566,
            172784.69566,
            172794.69566,
            172804.69566,
            172814.69566,
            172824.69566,
            172834.69566,
            172844.69566,
            172854.69566,
        ]
    )
    pos = np.array(
        [
            [-2136712.656972, -4807353.216281, 4723105.32593],
            [-2173278.123467, -4844728.748413, 4668055.885545],
            [-2209654.076842, -4881508.083748, 4612479.371246],
            [-2245835.560861, -4917687.36548, 4556382.044784],
            [-2281817.635428, -4953262.81132, 4499770.227249],
            [-2317595.377268, -4988230.71392, 4442650.298363],
            [-2353163.880601, -5022587.441289, 4385028.695791],
            [-2388518.257822, -5056329.4372, 4326911.91443],
            [-2423653.640169, -5089453.221581, 4268306.505655],
            [-2458565.178419, -5121955.390932, 4209219.076557],
            [-2493248.043545, -5153832.618673, 4149656.289196],
            [-2527697.427368, -5185081.655488, 4089624.859836],
        ]
    )
    vel = np.array(
        [
            [-3665.856712, -3767.232653, -5478.383128],
            [-3647.153734, -3707.808288, -5531.401613],
            [-3627.954346, -3647.994528, -5583.796915],
            [-3608.260129, -3587.798804, -5635.563062],
            [-3588.072733, -3527.228592, -5686.694153],
            [-3567.393873, -3466.291407, -5737.18436],
            [-3546.225334, -3404.994808, -5787.027928],
            [-3524.568967, -3343.346389, -5836.219171],
            [-3502.42669, -3281.353786, -5884.752482],
            [-3479.800488, -3219.02467, -5932.622324],
            [-3456.692412, -3156.366752, -5979.823238],
            [-3433.104583, -3093.387775, -6026.34984],
        ]
    )
    ref_epoch = datetime.datetime(2023, 10, 7, 14, 7, 57, 208273)
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

    assert orbit.size == 12
    expected = datetime.datetime(2023, 10, 9, 14, 7, 1, 903933)
    # Ignore the nanosecond:
    assert orbit.start_datetime.isoformat()[:-3] == expected.isoformat()
    assert orbit.mid_time == 55.0
    assert orbit.spacing == 10

    orbit_arrays = get_orbit_arrays(TEST_FILE)
    # orbit.time is a "linspace"
    np.testing.assert_array_equal(orbit.position, orbit_arrays[1])
    np.testing.assert_array_equal(orbit.velocity, orbit_arrays[2])


def test_get_lonlat_grid():
    lons, lats = get_lonlat_grid(TEST_FILE)
    assert lons.shape == (46, 199)
    assert lons.dtype == np.float64
    assert lons[0, 0] == -120.2641415906683
    assert lats.shape == (46, 199)
    assert lats.dtype == np.float64
    assert lats[0, 0] == 39.69554281541014
    assert np.all(np.diff(lons) > 0)
    # Lats are in descending order
    assert np.all(np.diff(lats) < 0)
    # It's *not* exactly a square regular grid in lat/lon, dur to warping
    assert lons[0, 0] != lons[-1, 0]


def test_get_xy_coords():
    x, y, epsg = get_xy_coords(TEST_FILE)

    assert x.shape == (199,)
    assert x.dtype == np.float64
    assert x[0] == 734582.5
    assert x[-1] == 833582.5

    assert y.shape == (46,)
    assert y.dtype == np.float64
    assert y[0] == 4397545.0
    assert y[-1] == 4352545.0

    assert epsg == 32610
