import netrc
import os
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Self

import requests

__all__ = [
    "AWSCredentials",
    "get_earthdata_username_password",
    "get_temporary_aws_credentials",
]


class EarthdataLoginFailure(Exception):
    """Exception raised when Earthdata Login credentials are not found."""

    pass


class ASFCredentialEndpoints(Enum):
    """Enumeration of ASF temporary credentials endpoints.

    See READMEs of the endpoints for more information, e.g.
    https://cumulus.asf.alaska.edu/s3credentialsREADME
    """

    OPERA = "https://cumulus.asf.alaska.edu/s3credentials"
    OPERA_UAT = "https://cumulus-test.asf.alaska.edu/s3credentials"
    SENTINEL1 = "https://sentinel1.asf.alaska.edu/s3credentials"


@dataclass
class AWSCredentials:
    """AWS credentials for direct S3 access."""

    access_key_id: str
    secret_access_key: str
    session_token: str

    def to_env(self) -> dict[str, str]:
        """Return the environment variable format of values: `AWS_`.

        Settable using os.environ.
        """
        return {
            "AWS_ACCESS_KEY_ID": self.access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.secret_access_key,
            "AWS_SESSION_TOKEN": self.session_token,
        }

    def to_h5py_kwargs(self) -> dict[str, str]:
        return {
            "secret_id": self.access_key_id,
            "secret_key": self.secret_access_key,
            "session_token": self.session_token,
        }

    @classmethod
    def from_asf(
        cls, endpoint: str | ASFCredentialEndpoints = ASFCredentialEndpoints.OPERA
    ) -> Self:
        """Get temporary AWS S3 access credentials.

        Requests new credentials if credentials are expired, or gets from the cache.

        Assumes Earthdata Login credentials are available via a .netrc file,
        or via EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables.
        """
        data = get_temporary_aws_credentials(ASFCredentialEndpoints(endpoint))
        return cls(
            access_key_id=data["accessKeyId"],
            secret_access_key=data["secretAccessKey"],
            session_token=data["sessionToken"],
        )

    @classmethod
    def from_env(cls) -> Self:
        """Get AWS credentials from 'AWS_*' environment variables.

        Required environment variables:
            AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN

        Raises
        ------
        KeyError
            If any of the required environment variables are not set.
        """
        return cls(
            access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            session_token=os.environ["AWS_SESSION_TOKEN"],
        )


@cache
def get_temporary_aws_credentials(
    endpoint: ASFCredentialEndpoints = ASFCredentialEndpoints.OPERA,
    earthdata_username: str | None = None,
    earthdata_password: str | None = None,
    host: str = "urs.earthdata.nasa.gov",
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
    host : str
        The host for which to authenticate using netrc.
        Default is "urs.earthdata.nasa.gov".

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
    username, password = get_earthdata_username_password(
        earthdata_username, earthdata_password, host
    )
    auth = (username, password)
    resp = requests.get(endpoint.value, auth=auth)
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
    raise ValueError(
        "No credentials found: neither valid parameters provided, .netrc file has a"
        f" '{host}' entry, nor environment variables set."
    )
