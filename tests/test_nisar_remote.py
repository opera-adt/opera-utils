"""Tests for opera_utils.nisar._remote module."""

from unittest.mock import MagicMock

import pytest

from opera_utils.nisar._remote import (
    _get_url_str,
    get_https_fs,
    get_s3_fs,
    open_file,
    open_h5,
)


class TestGetUrlStr:
    """Tests for _get_url_str function."""

    def test_string_passthrough(self):
        url = "https://example.com/data/file.h5"
        assert _get_url_str(url) == url

    def test_s3_string_passthrough(self):
        url = "s3://bucket/data/file.h5"
        assert _get_url_str(url) == url

    def test_path_to_uri(self, tmp_path):
        test_file = tmp_path / "test.h5"
        test_file.touch()
        result = _get_url_str(test_file)
        assert result.startswith("file://")
        assert str(test_file) in result


class TestGetHttpsFs:
    """Tests for get_https_fs function."""

    def test_creates_authenticated_filesystem(self, monkeypatch):
        def mock_get_creds(username, password, host):
            return ("test_user", "test_pass")

        monkeypatch.setattr(
            "opera_utils.nisar._remote.get_earthdata_username_password",
            mock_get_creds,
        )

        mock_fs = MagicMock()
        monkeypatch.setattr(
            "opera_utils.nisar._remote.fsspec.filesystem",
            lambda *a, **kw: mock_fs,  # noqa: ARG005
        )

        fs = get_https_fs()
        assert fs is mock_fs


class TestGetS3Fs:
    """Tests for get_s3_fs function."""

    def test_from_env_credentials(self, monkeypatch):
        mock_creds = MagicMock()
        mock_creds.access_key_id = "AKIATEST"
        mock_creds.secret_access_key = "secret"
        mock_creds.session_token = "token"

        monkeypatch.setattr(
            "opera_utils.nisar._remote.AWSCredentials.from_env",
            lambda: mock_creds,
        )

        mock_s3fs = MagicMock()
        monkeypatch.setattr("opera_utils.nisar._remote.s3fs.S3FileSystem", mock_s3fs)

        get_s3_fs()

        mock_s3fs.assert_called_once_with(
            key="AKIATEST", secret="secret", token="token"
        )


class TestOpenFile:
    """Tests for open_file function."""

    def test_local_file(self, tmp_path, monkeypatch):
        test_file = tmp_path / "test.h5"
        test_file.write_bytes(b"test content")

        mock_fs = MagicMock()
        mock_open = MagicMock()
        mock_fs.open.return_value = mock_open

        monkeypatch.setattr(
            "opera_utils.nisar._remote.fsspec.filesystem",
            lambda protocol: mock_fs,  # noqa: ARG005
        )

        result = open_file(test_file)

        mock_fs.open.assert_called_once()
        assert result is mock_open

    def test_https_url(self, monkeypatch):
        mock_fs = MagicMock()
        mock_open = MagicMock()
        mock_fs.open.return_value = mock_open

        monkeypatch.setattr(
            "opera_utils.nisar._remote.get_https_fs",
            lambda username, password: mock_fs,  # noqa: ARG005
        )

        result = open_file("https://example.com/data/file.h5")

        mock_fs.open.assert_called_once()
        assert result is mock_open

    def test_s3_url(self, monkeypatch):
        mock_fs = MagicMock()
        mock_open = MagicMock()
        mock_fs.open.return_value = mock_open

        monkeypatch.setattr("opera_utils.nisar._remote.get_s3_fs", lambda: mock_fs)

        result = open_file("s3://bucket/data/file.h5")

        mock_fs.open.assert_called_once()
        assert result is mock_open

    def test_invalid_scheme(self):
        with pytest.raises(ValueError, match="Unrecognized scheme"):
            open_file("ftp://example.com/data/file.h5")


class TestOpenH5:
    """Tests for open_h5 function."""

    def test_opens_h5_file(self, monkeypatch):
        mock_byte_stream = MagicMock()
        mock_h5_file = MagicMock()

        monkeypatch.setattr(
            "opera_utils.nisar._remote.open_file",
            lambda *a, **kw: mock_byte_stream,  # noqa: ARG005
        )
        monkeypatch.setattr(
            "opera_utils.nisar._remote.h5py.File",
            lambda *a, **kw: mock_h5_file,  # noqa: ARG005
        )

        result = open_h5("https://example.com/data/file.h5")

        assert result is mock_h5_file

    def test_custom_page_size(self, monkeypatch):
        mock_byte_stream = MagicMock()
        h5_calls = []

        def mock_h5_file(stream, mode, **kwargs):
            h5_calls.append(kwargs)
            return MagicMock()

        monkeypatch.setattr(
            "opera_utils.nisar._remote.open_file",
            lambda *a, **kw: mock_byte_stream,  # noqa: ARG005
        )
        monkeypatch.setattr("opera_utils.nisar._remote.h5py.File", mock_h5_file)

        open_h5("https://example.com/data/file.h5", page_size=8 * 1024 * 1024)

        assert h5_calls[0]["fs_page_size"] == 8 * 1024 * 1024
