"""Tests for the remote module in disp package, rewritten to use pytest monkeypatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from opera_utils.credentials import AWSCredentials
from opera_utils.disp._product import DispProduct
from opera_utils.disp._remote import get_https_fs, get_s3_fs, open_h5


# @pytest.mark.vcr
@pytest.mark.skip(
    reason="fsspec fetching entire file; `size_policy` needs investigation"
)
def test_open_h5_disp_product():
    url = "https://datapool-test.asf.alaska.edu/DISP/OPERA-S1/OPERA_L3_DISP-S1_IW_F11115_VV_20160810T140735Z_20160903T140736Z_v1.1_20250215T012240Z.nc"
    dp = DispProduct(url)
    with open_h5(url, asf_endpoint="OPERA_UAT", rdcc_nbytes=1024) as hf:
        assert "displacement" in hf
        assert "phase_similarity" in hf
        assert hf["displacement"].shape == dp.shape
        # Fetch real displacement value at one pixel
        d = hf["displacement"][4000, 4000]
        assert d == np.float32(0.047729492)


def test_get_https_fs(monkeypatch):
    """Test get_https_fs function."""

    # Mock get_earthdata_username_password
    def mock_get_creds(u=None, p=None, host=None):
        return ("test_user", "test_pass")

    monkeypatch.setattr(
        "opera_utils.disp._remote.get_earthdata_username_password", mock_get_creds
    )

    # Mock fsspec.filesystem
    mock_fs = MagicMock()

    def mock_fsspec_filesystem(protocol, **kwargs):
        return mock_fs

    monkeypatch.setattr("fsspec.filesystem", mock_fsspec_filesystem)

    # Call the function
    result = get_https_fs("custom_user", "custom_pass", "custom.host")

    assert result == mock_fs


def test_get_s3_fs_from_env(monkeypatch):
    """Test get_s3_fs function with credentials from environment."""
    # Mock AWSCredentials.from_env
    mock_creds = MagicMock()
    mock_creds.access_key_id = "env_id"
    mock_creds.secret_access_key = "env_secret"
    mock_creds.session_token = "env_token"

    def mock_from_env():
        return mock_creds

    monkeypatch.setattr(AWSCredentials, "from_env", mock_from_env)

    # Mock AWSCredentials.from_asf so we can check it does NOT get called
    mock_from_asf = MagicMock()
    monkeypatch.setattr(AWSCredentials, "from_asf", mock_from_asf)

    # Mock s3fs.S3FileSystem
    mock_s3fs_instance = MagicMock()

    def mock_s3fs_init(*args, **kwargs):
        return mock_s3fs_instance

    monkeypatch.setattr("s3fs.S3FileSystem", mock_s3fs_init)

    # Call the function
    result = get_s3_fs()
    assert result == mock_s3fs_instance


def test_get_s3_fs_from_asf(monkeypatch):
    """Test get_s3_fs function with credentials from ASF (when env fails)."""

    # Make from_env raise KeyError to simulate missing env vars
    def mock_from_env():
        msg = "No AWS credentials in environment"
        raise KeyError(msg)

    monkeypatch.setattr(AWSCredentials, "from_env", mock_from_env)

    # Mock AWSCredentials.from_asf
    mock_creds = MagicMock()
    mock_creds.access_key_id = "asf_id"
    mock_creds.secret_access_key = "asf_secret"
    mock_creds.session_token = "asf_token"

    def mock_from_asf(endpoint):
        return mock_creds

    monkeypatch.setattr(AWSCredentials, "from_asf", mock_from_asf)

    # Mock s3fs.S3FileSystem
    mock_s3fs_instance = MagicMock()

    def mock_s3fs_init(*args, **kwargs):
        return mock_s3fs_instance

    monkeypatch.setattr("s3fs.S3FileSystem", mock_s3fs_init)

    # Call the function
    result = get_s3_fs("opera-uat")

    # Verify
    assert result == mock_s3fs_instance


def test_open_h5_https(monkeypatch):
    """Test open_h5 with an HTTPS URL."""
    # Mock get_https_fs
    mock_fs = MagicMock()
    mock_byte_stream = MagicMock()
    mock_fs.open.return_value = mock_byte_stream

    def mock_get_https_fs(earthdata_username=None, earthdata_password=None, host=None):
        return mock_fs

    monkeypatch.setattr("opera_utils.disp._remote.get_https_fs", mock_get_https_fs)

    # Mock h5py.File
    mock_file = MagicMock()

    def mock_h5py_file(bytestream, mode="r", fs_page_size=None, rdcc_nbytes=None):
        return mock_file

    monkeypatch.setattr("h5py.File", mock_h5py_file)

    url = "https://datapool-test.asf.alaska.edu/DISP/OPERA-S1/OPERA_L3_DISP-S1_IW_F11115_VV_20160810T140735Z_20160903T140736Z_v1.1_20250215T012240Z.nc"
    result = open_h5(
        url,
        page_size=8192,
        rdcc_nbytes=2000000,
        earthdata_username="test_user",
        earthdata_password="test_pass",
    )

    assert result == mock_file


def test_open_h5_s3(monkeypatch):
    """Test open_h5 with an S3 URL."""
    # Mock get_s3_fs
    mock_fs = MagicMock()
    mock_byte_stream = MagicMock()
    mock_fs.open.return_value = mock_byte_stream

    def mock_get_s3_fs(asf_endpoint="opera"):
        return mock_fs

    monkeypatch.setattr("opera_utils.disp._remote.get_s3_fs", mock_get_s3_fs)

    # Mock h5py.File
    mock_file = MagicMock()

    def mock_h5py_file(bytestream, mode="r", **kwargs):
        return mock_file

    monkeypatch.setattr("h5py.File", mock_h5py_file)

    # Call the function
    result = open_h5(
        "s3://bucket/file.h5",
        asf_endpoint="OPERA_UAT",
    )
    assert result == mock_file


def test_open_h5_local(monkeypatch):
    """Test open_h5 with a local file path."""
    # Mock filesystem
    mock_fs = MagicMock()
    mock_byte_stream = MagicMock()
    mock_fs.open.return_value = mock_byte_stream

    def mock_fsspec_filesystem(protocol):
        return mock_fs

    monkeypatch.setattr("fsspec.filesystem", mock_fsspec_filesystem)

    # Mock h5py.File
    mock_file = MagicMock()

    def mock_h5py_file(bytestream, mode="r", **kwargs):
        return mock_file

    monkeypatch.setattr("h5py.File", mock_h5py_file)

    # We also need to mock Path.resolve() -> path.as_uri()
    file_path = Path("/path/to/local/file.h5")

    # The object that will replace the return value of .resolve()
    mock_resolved_path = MagicMock()
    mock_resolved_path.as_uri.return_value = "file:///path/to/local/file.h5"

    def mock_resolve(*args, **kwargs):
        return mock_resolved_path

    monkeypatch.setattr(Path, "resolve", mock_resolve)

    # Call the function
    result = open_h5(file_path)
    assert result == mock_file

    # Try with a str, not path
    result = open_h5("/path/to/local/file.h5")
    assert result == mock_file


def test_open_h5_invalid_url():
    """Test open_h5 with an invalid URL scheme."""
    with pytest.raises(ValueError, match="Unrecognized scheme"):
        open_h5("ftp://example.com/file.h5")
