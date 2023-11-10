import h5py
import numpy as np
import pytest

from opera_utils.h5explorer import HDF5Explorer


@pytest.fixture
def h5_file(tmp_path):
    fname = tmp_path / "test_file.h5"
    with h5py.File(fname, "w") as hf:
        hf["/data/name1"] = np.random.rand(10, 10)
        hf["/data/name2"] = np.random.rand(10, 10)
        hf["metadata/dset1"] = 2

    return fname


def test_hdf5_explorer(h5_file):
    h = HDF5Explorer(h5_file)
    assert h.data.name1.shape == (10, 10)
    assert h.data.name2.shape == (10, 10)
    assert h.metadata.dset1 == 2
    assert dir(h) == ["data", "metadata"]
    assert dir(h.data) == ["name1", "name2"]
    assert dir(h.metadata) == ["dset1"]
