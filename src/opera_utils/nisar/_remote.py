"""Remote file access for NISAR GSLC products.

This module re-exports the shared remote access utilities from
:mod:`opera_utils._remote`. NISAR GSLC products use the same Earthdata
authentication infrastructure as DISP-S1 products.
"""

from __future__ import annotations

from opera_utils._remote import (
    get_https_fs,
    get_s3_fs,
    get_url_str,
    open_file,
    open_h5,
)

__all__ = ["get_https_fs", "get_s3_fs", "open_file", "open_h5"]

# Re-export (keep backward-compatible private name)
_get_url_str = get_url_str
