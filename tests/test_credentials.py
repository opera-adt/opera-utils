"""Tests for the credentials module using pytest monkeypatch."""

from __future__ import annotations

import netrc
import sys
from unittest.mock import MagicMock

import pytest
import requests

from opera_utils.credentials import (
    ASFCredentialEndpoints,
    AWSCredentials,
    EarthdataLoginError,
    get_earthdata_username_password,
    get_temporary_aws_credentials,
)


def test_aws_credentials_dataclass():
    """Test that AWSCredentials dataclass has expected methods."""
    creds = AWSCredentials(
        access_key_id="test_id",
        secret_access_key="test_secret",
        session_token="test_token",
    )
    # Test to_env method
    env_vars = creds.to_env()
    assert env_vars["AWS_ACCESS_KEY_ID"] == "test_id"
    assert env_vars["AWS_SECRET_ACCESS_KEY"] == "test_secret"
    assert env_vars["AWS_SESSION_TOKEN"] == "test_token"

    # Test to_h5py_kwargs method
    h5py_kwargs = creds.to_h5py_kwargs()
    assert h5py_kwargs["secret_id"] == "test_id"
    assert h5py_kwargs["secret_key"] == "test_secret"
    assert h5py_kwargs["session_token"] == "test_token"


def test_aws_credentials_from_asf(monkeypatch):
    """Test AWSCredentials.from_asf method using monkeypatch."""

    def mock_get_temp_creds(*args, **kwargs):
        return {
            "accessKeyId": "test_id",
            "secretAccessKey": "test_secret",
            "sessionToken": "test_token",
        }

    monkeypatch.setattr(
        "opera_utils.credentials.get_temporary_aws_credentials", mock_get_temp_creds
    )

    creds = AWSCredentials.from_asf(ASFCredentialEndpoints.OPERA)
    assert creds.access_key_id == "test_id"
    assert creds.secret_access_key == "test_secret"
    assert creds.session_token == "test_token"


def test_aws_credentials_from_env(monkeypatch):
    """Test AWSCredentials.from_env method using monkeypatch."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env_id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env_secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "env_token")

    creds = AWSCredentials.from_env()
    assert creds.access_key_id == "env_id"
    assert creds.secret_access_key == "env_secret"
    assert creds.session_token == "env_token"


def test_aws_credentials_from_env_missing_vars(monkeypatch):
    """Test AWSCredentials.from_env raises KeyError when env vars missing."""
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    with pytest.raises(KeyError):
        AWSCredentials.from_env()


@pytest.mark.vcr
def test_get_temporary_aws_credentials():
    """Test get_temporary_aws_credentials with authentication."""
    result = get_temporary_aws_credentials()

    expected = {
        "accessKeyId": "FAKEACCESS",
        "secretAccessKey": "FAKESECRET",
        "sessionToken": "FAKESESSION",
        "expiration": "2025-04-08 14:20:17+00:00",
    }
    assert result == expected


@pytest.mark.vcr
def test_get_temporary_aws_credentials_different_endpoint():
    """Test get_temporary_aws_credentials with authentication."""
    if sys.version_info < (3, 10):
        # Problem:
        # ('Received response with content-encoding: gzip, but failed to decode it.',
        #     error('Error -3 while decompressing data: incorrect header check'))
        pytest.skip("Skipping fake decoding on Python 3.9")
    result = get_temporary_aws_credentials(endpoint=ASFCredentialEndpoints.OPERA_UAT)
    expected = {
        "accessKeyId": "FAKEACCESS",
        "secretAccessKey": "FAKESECRET",
        "sessionToken": "FAKESESSION",
        "expiration": "2025-04-08 14:22:33+00:00",
    }

    assert result == expected


def test_get_temporary_aws_credentials_error(monkeypatch):
    """Test get_temporary_aws_credentials with HTTP error."""
    mock_response = MagicMock()
    mock_response.status_code = 200  # so we skip the auth check
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Test error"
    )

    def mock_requests_get(url, *args, **kwargs):
        return mock_response

    monkeypatch.setattr(requests, "get", mock_requests_get)

    with pytest.raises(requests.exceptions.HTTPError, match="Test error"):
        get_temporary_aws_credentials.__wrapped__(
            ASFCredentialEndpoints.OPERA, None, None
        )


def test_get_earthdata_username_password_direct():
    """Test get_earthdata_username_password with direct parameters."""
    username, password = get_earthdata_username_password("direct_user", "direct_pass")
    assert username == "direct_user"
    assert password == "direct_pass"


def test_get_earthdata_username_password_from_env(monkeypatch):
    """Test get_earthdata_username_password from environment variables."""

    # Clear out netrc usage
    def mock_netrc():
        raise FileNotFoundError()

    monkeypatch.setattr(netrc, "netrc", mock_netrc)
    monkeypatch.setenv("EARTHDATA_USERNAME", "env_user")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "env_pass")

    username, password = get_earthdata_username_password()
    assert username == "env_user"
    assert password == "env_pass"


def test_get_earthdata_username_password_from_netrc(monkeypatch):
    """Test get_earthdata_username_password from .netrc file."""
    mock_netrc_instance = MagicMock()
    mock_netrc_instance.authenticators.return_value = ("netrc_user", None, "netrc_pass")

    def mock_netrc():
        return mock_netrc_instance

    monkeypatch.setattr(netrc, "netrc", mock_netrc)
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)

    username, password = get_earthdata_username_password()
    assert username == "netrc_user"
    assert password == "netrc_pass"
    mock_netrc_instance.authenticators.assert_called_once_with("urs.earthdata.nasa.gov")


def test_get_earthdata_username_password_no_credentials(monkeypatch):
    """Test get_earthdata_username_password with no credentials available."""
    # Mock netrc to return None
    mock_netrc_instance = MagicMock()
    mock_netrc_instance.authenticators.return_value = None

    def mock_netrc():
        return mock_netrc_instance

    monkeypatch.setattr(netrc, "netrc", mock_netrc)
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)

    with pytest.raises(EarthdataLoginError, match="No credentials found"):
        get_earthdata_username_password()


def test_get_earthdata_username_password_netrc_error(monkeypatch):
    """Test get_earthdata_username_password with netrc error."""

    # netrc raises exception
    def mock_netrc_error():
        msg = "No .netrc file"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(netrc, "netrc", mock_netrc_error)

    # Provide backup direct credentials
    username, password = get_earthdata_username_password("backup_user", "backup_pass")
    assert username == "backup_user"
    assert password == "backup_pass"

    # No direct credentials + netrc error + no env => fail
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)

    with pytest.raises(EarthdataLoginError, match="No credentials found"):
        get_earthdata_username_password()


def test_get_earthdata_username_password_invalid_host():
    """Test get_earthdata_username_password with invalid host."""
    with pytest.raises(ValueError, match="Invalid host"):
        get_earthdata_username_password(host="invalid.host.com")
