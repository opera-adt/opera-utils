"""Tests for the credentials module."""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from opera_utils.credentials import (
    ASFCredentialEndpoints,
    AWSCredentials,
    EarthdataLoginFailure,
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


@patch("opera_utils.credentials.get_temporary_aws_credentials")
def test_aws_credentials_from_asf(mock_get_temp_creds):
    """Test AWSCredentials.from_asf method."""
    # Mock the response from get_temporary_aws_credentials
    mock_get_temp_creds.return_value = {
        "accessKeyId": "test_id",
        "secretAccessKey": "test_secret",
        "sessionToken": "test_token",
    }

    # Call the method
    creds = AWSCredentials.from_asf(ASFCredentialEndpoints.OPERA)

    # Verify the response
    assert creds.access_key_id == "test_id"
    assert creds.secret_access_key == "test_secret"
    assert creds.session_token == "test_token"

    # Verify the mock was called with expected args
    mock_get_temp_creds.assert_called_once_with(ASFCredentialEndpoints.OPERA)


@patch.dict(os.environ, {
    "AWS_ACCESS_KEY_ID": "env_id",
    "AWS_SECRET_ACCESS_KEY": "env_secret",
    "AWS_SESSION_TOKEN": "env_token",
})
def test_aws_credentials_from_env():
    """Test AWSCredentials.from_env method."""
    creds = AWSCredentials.from_env()
    assert creds.access_key_id == "env_id"
    assert creds.secret_access_key == "env_secret"
    assert creds.session_token == "env_token"


def test_aws_credentials_from_env_missing_vars():
    """Test AWSCredentials.from_env raises KeyError when env vars missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(KeyError):
            AWSCredentials.from_env()


# Need to patch the cache decorator to avoid interference between tests
@patch("opera_utils.credentials.get_temporary_aws_credentials", wraps=get_temporary_aws_credentials)
@patch("requests.get")
def test_get_temporary_aws_credentials_direct(mock_requests_get, _):
    """Test get_temporary_aws_credentials with direct response (no auth needed)."""
    # Mock response for direct success (no auth needed)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "accessKeyId": "direct_id",
        "secretAccessKey": "direct_secret",
        "sessionToken": "direct_token",
    }
    mock_requests_get.return_value = mock_response

    # Call the function with a string endpoint to test the conversion
    result = get_temporary_aws_credentials("OPERA", None, None)

    # Verify the result
    assert result == mock_response.json.return_value
    mock_requests_get.assert_called_once_with(ASFCredentialEndpoints.OPERA.value)


# Override the cache functionality to avoid test interference
@patch("opera_utils.credentials.get_temporary_aws_credentials.__wrapped__", wraps=get_temporary_aws_credentials.__wrapped__)  
@patch("opera_utils.credentials.get_earthdata_username_password")
@patch("requests.get")
def test_get_temporary_aws_credentials_with_auth(mock_requests_get, mock_get_creds, _):
    """Test get_temporary_aws_credentials with authentication."""
    # First response needs auth
    auth_needed_response = MagicMock()
    auth_needed_response.status_code = 401
    auth_needed_response.url = "https://urs.earthdata.nasa.gov/oauth/authorize?client_id=test"

    # Second response (after auth) succeeds
    auth_success_response = MagicMock()
    auth_success_response.status_code = 200
    auth_success_response.json.return_value = {
        "accessKeyId": "auth_id",
        "secretAccessKey": "auth_secret",
        "sessionToken": "auth_token",
    }

    # Set up mock to return different responses
    mock_requests_get.side_effect = [auth_needed_response, auth_success_response]

    # Mock the credentials
    mock_get_creds.return_value = ("test_user", "test_pass")

    # Call the function's wrapped version to bypass the cache
    result = get_temporary_aws_credentials.__wrapped__(ASFCredentialEndpoints.OPERA, None, None)

    # Verify the result
    assert result == auth_success_response.json.return_value
    assert mock_requests_get.call_count == 2
    mock_requests_get.assert_any_call(ASFCredentialEndpoints.OPERA.value)
    mock_requests_get.assert_any_call(auth_needed_response.url, auth=("test_user", "test_pass"))
    mock_get_creds.assert_called_once_with(None, None, host="urs.earthdata.nasa.gov")


@patch("opera_utils.credentials.get_temporary_aws_credentials.__wrapped__", wraps=get_temporary_aws_credentials.__wrapped__)
@patch("requests.get")
def test_get_temporary_aws_credentials_error(mock_requests_get, _):
    """Test get_temporary_aws_credentials with HTTP error."""
    # Mock response to raise exception
    mock_response = MagicMock()
    mock_response.status_code = 200  # to avoid auth path
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Test error")
    mock_requests_get.return_value = mock_response

    # Call the function and verify it raises the expected exception
    with pytest.raises(requests.exceptions.HTTPError, match="Test error"):
        get_temporary_aws_credentials.__wrapped__(ASFCredentialEndpoints.OPERA, None, None)


def test_get_earthdata_username_password_direct():
    """Test get_earthdata_username_password with direct parameters."""
    username, password = get_earthdata_username_password("direct_user", "direct_pass")
    assert username == "direct_user"
    assert password == "direct_pass"


# To avoid test interference from environment variables
@patch.dict(os.environ, {"EARTHDATA_USERNAME": "env_user", "EARTHDATA_PASSWORD": "env_pass"}, clear=True)
def test_get_earthdata_username_password_from_env():
    """Test get_earthdata_username_password from environment variables."""
    # Mock netrc to ensure we don't use it
    with patch("netrc.netrc") as mock_netrc:
        mock_netrc.side_effect = FileNotFoundError()
        
        username, password = get_earthdata_username_password()
        assert username == "env_user"
        assert password == "env_pass"


@patch("netrc.netrc")
def test_get_earthdata_username_password_from_netrc(mock_netrc):
    """Test get_earthdata_username_password from .netrc file."""
    # Mock the netrc authenticators
    mock_netrc_instance = MagicMock()
    mock_netrc_instance.authenticators.return_value = ("netrc_user", None, "netrc_pass")
    mock_netrc.return_value = mock_netrc_instance

    # Clean environment to ensure we're testing netrc only
    with patch.dict(os.environ, {}, clear=True):
        username, password = get_earthdata_username_password()
        assert username == "netrc_user"
        assert password == "netrc_pass"
        mock_netrc_instance.authenticators.assert_called_once_with("urs.earthdata.nasa.gov")


@patch("netrc.netrc")
def test_get_earthdata_username_password_no_credentials(mock_netrc):
    """Test get_earthdata_username_password with no credentials available."""
    # Mock netrc to return None
    mock_netrc_instance = MagicMock()
    mock_netrc_instance.authenticators.return_value = None
    mock_netrc.return_value = mock_netrc_instance

    # Clean environment
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="No credentials found"):
            get_earthdata_username_password()


@patch("netrc.netrc")
def test_get_earthdata_username_password_netrc_error(mock_netrc):
    """Test get_earthdata_username_password with netrc error."""
    # Mock netrc to raise exception
    mock_netrc.side_effect = FileNotFoundError("No .netrc file")

    # Clean environment but provide direct credentials
    with patch.dict(os.environ, {}, clear=True):
        username, password = get_earthdata_username_password("backup_user", "backup_pass")
        assert username == "backup_user"
        assert password == "backup_pass"

    # Clean environment with no direct credentials
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="No credentials found"):
            get_earthdata_username_password()


def test_get_earthdata_username_password_invalid_host():
    """Test get_earthdata_username_password with invalid host."""
    with pytest.raises(ValueError, match="Invalid host"):
        get_earthdata_username_password(host="invalid.host.com")