from __future__ import annotations

from os import fsdecode
from pathlib import Path
from typing import Any

import aiohttp
import fsspec
import h5py
import s3fs

from opera_utils.credentials import (
    ENDPOINT_TO_HOST,
    ASFCredentialEndpoints,
    AWSCredentials,
    get_earthdata_username_password,
)
from opera_utils.disp._product import DispProduct

__all__ = ["open_h5"]


def open_h5(
    url: str | Path | DispProduct,
    page_size: int = 4 * 1024 * 1024,
    rdcc_nbytes: int = 1024 * 1024 * 1000,
    earthdata_username: str | None = None,
    earthdata_password: str | None = None,
    asf_endpoint: str | ASFCredentialEndpoints = "OPERA",
    fsspec_kwargs: dict[str, Any] = {"cache_type": "first"},
) -> h5py.File:
    """Open a remote (or local) HDF5 file.

    Can handle both HTTPS access (your Earthdata login credentials) and
    S3 URLs (for direct S3 access via temporary AWS credentials).

    Note that direct S3 access is only available for Earthdata granules
    if you are running on an in-region EC2 instance.

    For HTTPS access, the Earthdata Login credentials are used to authenticate
    the request. There are 3 methods of providing the login credentials:
    1. Directly passed as arguments
    2. From the ~.netrc file
    3. From environment variables EARTHDATA_USERNAME and EARTHDATA_PASSWORD.

    For S3 access, AWS credentials required.
    They will be retrieved from the environment variables
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_SESSION_TOKEN, or
    temporary AWS credentials can be requested from ASF using the
    Earthdata Login credentials.

    Parameters
    ----------
    url : str
        The URL of the HDF5 file to be accessed. Can be a local file path, an
        HTTPS URL, or an S3 URL.
    page_size : int, optional
        The page size to use for the HDF5 file.
        Default is 4MB.
        Must be a power of 2 and larger than the page size used to create the
        HDF5 file (which is 4 MB for OPERA DISP-S1)
    rdcc_nbytes : int
        The number of bytes to use for the read cache.
        Default is 1GB
    earthdata_username : str | None, optional
        Earthdata Login username, if environment variables are not set.
    earthdata_password : str | None, optional
        Earthdata Login password, if environment variables are not set.
    asf_endpoint : str
        (For S3 access) The ASF endpoint to use for temporary AWS credentials.
        Choices are "OPERA", "OPERA_UAT", "SENTINEL1"
        Default is "OPERA"
    fsspec_kwargs : dict[str, Any], optional
        Additional keyword arguments to pass to fsspec.
        Default is `{"cache_type": "first"}`.


    Returns
    -------
    h5py.File
        An opened HDF5 file.

    Raises
    ------
    ValueError
        If the .netrc file does not contain authentication information for the
        specified host.

    """
    url_str = fsdecode(url.resolve().as_uri() if isinstance(url, Path) else url)

    if isinstance(asf_endpoint, str):
        assert asf_endpoint.upper() in {"OPERA", "OPERA_UAT", "SENTINEL1"}
        asf_endpoint = ASFCredentialEndpoints[asf_endpoint.upper()]

    if url_str.startswith("http"):
        fs = get_https_fs(
            earthdata_username, earthdata_password, host=ENDPOINT_TO_HOST[asf_endpoint]
        )
    elif url_str.startswith("s3://"):
        fs = get_s3_fs(asf_endpoint=asf_endpoint)
    elif url_str.startswith("file://"):
        fs = fsspec.filesystem("file")
    else:
        raise ValueError(f"Unrecognized scheme for {url_str}")

    # h5py arguments used to set the "cloud-friendly" parameters
    cloud_kwargs = dict(fs_page_size=page_size, rdcc_nbytes=rdcc_nbytes)
    # Create the Open File-like object from fsspec
    byte_stream = fs.open(path=url_str, mode="rb", **fsspec_kwargs)
    return h5py.File(byte_stream, mode="r", **cloud_kwargs)


def get_https_fs(
    earthdata_username: str | None = None,
    earthdata_password: str | None = None,
    host="urs.earthdata.nasa.gov",
) -> fsspec.AbstractFileSystem:
    """Create an fsspec filesystem object authenticated using netrc for HTTP access.

    Parameters
    ----------
    earthdata_username : str | None
        Earthdata Login username, if environment variables are not set.
    earthdata_password : str | None
        Earthdata Login password, if environment variables are not set.
    host : str, optional
        The host for which to authenticate using netrc.
        Default is "urs.earthdata.nasa.gov".

    Returns
    -------
    fsspec.AbstractFileSystem
        An authenticated fsspec filesystem object for HTTP access.

    Raises
    ------
    ValueError
        If the .netrc file does not contain authentication information for the
        specified host.

    """
    username, password = get_earthdata_username_password(
        earthdata_username, earthdata_password, host=host
    )

    fs = fsspec.filesystem(
        "https", client_kwargs={"auth": aiohttp.BasicAuth(username, password)}
    )
    return fs


def get_s3_fs(
    asf_endpoint: ASFCredentialEndpoints = ASFCredentialEndpoints.OPERA,
) -> s3fs.S3FileSystem:
    """Create an fsspec filesystem object authenticated using temporary AWS credentials.

    Parameters
    ----------
    asf_endpoint : ASFCredentialEndpoints, optional
        The ASF endpoint to use for temporary AWS credentials, by default "OPERA"

    Returns
    -------
    s3fs.S3FileSystem
        An authenticated fsspec filesystem object for S3 access.
    """
    try:
        creds = AWSCredentials.from_env()
    except KeyError:
        creds = AWSCredentials.from_asf(endpoint=asf_endpoint)
    s3_fs = s3fs.S3FileSystem(
        key=creds.access_key_id,
        secret=creds.secret_access_key,
        token=creds.session_token,
    )
    return s3_fs
