from __future__ import annotations

import netrc
import os
from dataclasses import dataclass
from enum import Enum
from functools import cache

import requests
from typing_extensions import Self

__all__ = [
    "AWSCredentials",
    "get_earthdata_username_password",
    "get_temporary_aws_credentials",
]


class EarthdataLoginError(Exception):
    """Exception raised when Earthdata Login credentials are not found."""


class ASFCredentialEndpoints(Enum):
    """Enumeration of ASF temporary credentials endpoints.

    See READMEs of the endpoints for more information, e.g.
    https://cumulus.asf.alaska.edu/s3credentialsREADME
    """

    OPERA = "https://cumulus.asf.alaska.edu/s3credentials"
    OPERA_UAT = "https://cumulus-test.asf.alaska.edu/s3credentials"
    SENTINEL1 = "https://sentinel1.asf.alaska.edu/s3credentials"


ENDPOINT_TO_HOST = {
    ASFCredentialEndpoints.OPERA: "urs.earthdata.nasa.gov",
    ASFCredentialEndpoints.OPERA_UAT: "uat.urs.earthdata.nasa.gov",
    ASFCredentialEndpoints.SENTINEL1: "urs.earthdata.nasa.gov",
}


@dataclass
class AWSCredentials:
    """AWS credentials for direct S3 access."""

    access_key_id: str
    secret_access_key: str
    session_token: str | None

    def to_env(self) -> dict[str, str]:
        """Return the environment variable format of values: `AWS_`.

        Settable using os.environ.
        """
        creds = {
            "AWS_ACCESS_KEY_ID": self.access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.secret_access_key,
        }
        if self.session_token is not None:
            creds["AWS_SESSION_TOKEN"] = self.session_token
        return creds

    def to_h5py_kwargs(self) -> dict[str, str]:
        creds = {
            "secret_id": self.access_key_id,
            "secret_key": self.secret_access_key,
        }
        if self.session_token is not None:
            creds["session_token"] = self.session_token
        return creds

    @classmethod
    def from_asf(
        cls, endpoint: ASFCredentialEndpoints = ASFCredentialEndpoints.OPERA
    ) -> Self:
        """Get temporary AWS S3 access credentials.

        Requests new credentials if credentials are expired, or gets from the cache.

        Assumes Earthdata Login credentials are available via a .netrc file,
        or via EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables.
        """
        data = get_temporary_aws_credentials(endpoint)
        return cls(
            access_key_id=data["accessKeyId"],
            secret_access_key=data["secretAccessKey"],
            session_token=data["sessionToken"],
        )

    @classmethod
    def from_env(cls) -> Self:
        """Get AWS credentials from 'AWS_*' environment variables.

        Required environment variables:
            AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, (and optionally AWS_SESSION_TOKEN)

        Raises
        ------
        KeyError
            If any of the required environment variables are not set.

        """
        return cls(
            access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            session_token=os.environ.get("AWS_SESSION_TOKEN"),
        )

    @classmethod
    def from_boto(cls) -> Self:
        """Get AWS credentials for the current session using botocore.

        botocore must be installed.

        Raises
        ------
        ImportError
            If botocore is not installed.

        """
        import botocore.session

        session = botocore.session.get_session()
        credentials = session.get_credentials()
        if credentials is None:
            msg = "No credentials found in boto3 session."
            raise ValueError(msg)
        frozen_credentials = credentials.get_frozen_credentials()
        return cls(
            access_key_id=frozen_credentials.access_key,
            secret_access_key=frozen_credentials.secret_key,
            session_token=frozen_credentials.token,
        )


@cache
def get_temporary_aws_credentials(
    endpoint: ASFCredentialEndpoints = ASFCredentialEndpoints.OPERA,
    earthdata_username: str | None = None,
    earthdata_password: str | None = None,
) -> dict[str, str]:
    """Get temporary AWS S3 access credentials.

    Requests new credentials if credentials are expired, or gets from the cache.
    Earthdata arguments are used to request new S3 credentials from ASF.

    Parameters
    ----------
    endpoint : ASFCredentialEndpoints
        The endpoint to request credentials from.
        Default is OPERA.
    earthdata_username : str | None
        Earthdata Login username.
    earthdata_password : str | None
        Earthdata Login password.

    Returns
    -------
    dict:
        JSON response from s3credentials URL.

    Raises
    ------
    EarthdataLoginFailure
        If the Earthdata Login credentials are not found.
    requests.exceptions.HTTPError
        If the request to the endpoint fails.

    """
    resp = requests.get(endpoint.value)
    if resp.status_code == 401 and "nasa.gov/oauth/authorize?" in resp.url:
        username, password = get_earthdata_username_password(
            earthdata_username, earthdata_password, host=ENDPOINT_TO_HOST[endpoint]
        )
        auth = (username, password)
        resp = requests.get(resp.url, auth=auth)
    resp.raise_for_status()
    return resp.json()


def get_earthdata_username_password(
    earthdata_username: str | None = None,
    earthdata_password: str | None = None,
    host: str = "urs.earthdata.nasa.gov",
) -> tuple[str, str]:
    """Get Earthdata Login credentials.

    Parameters
    ----------
    earthdata_username : str | None
        Earthdata Login username.
    earthdata_password : str | None
        Earthdata Login password.
    host : str
        The host for which to authenticate using netrc.
        Default is "urs.earthdata.nasa.gov".

    Returns
    -------
    tuple[str, str]
        A tuple containing the username and password.

    Raises
    ------
    ValueError
        If no credentials are found.

    """
    if host not in set(ENDPOINT_TO_HOST.values()):
        msg = f"Invalid host: {host}. Choices: {ENDPOINT_TO_HOST.values()}"
        raise ValueError(msg)

    # Case 1: Use provided credentials if both are specified
    if earthdata_username and earthdata_password:
        return earthdata_username, earthdata_password

    # Case 2: Try to get credentials from .netrc file
    try:
        auth = netrc.netrc().authenticators(host)
        if auth and auth[0] and auth[2]:
            return auth[0], auth[2]
    except (FileNotFoundError, netrc.NetrcParseError):
        pass

    # Case 3: Try to get credentials from environment variables
    username = os.environ.get("EARTHDATA_USERNAME", "")
    password = os.environ.get("EARTHDATA_PASSWORD", "")
    if username and password:
        return username, password

    # No valid credentials found
    msg = (
        "No credentials found: neither valid parameters provided, .netrc file has a"
        f" '{host}' entry, nor environment variables set."
    )
    raise EarthdataLoginError(msg)
