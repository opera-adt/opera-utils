import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore::UserWarning:h5py is running against HDF5.*"
)
