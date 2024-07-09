from __future__ import annotations

import pooch

__all__ = [
    "fetch_frame_geometries_simple",
    "fetch_burst_id_geometries_simple",
    "fetch_burst_to_frame_mapping_file",
    "fetch_frame_to_burst_mapping_file",
]

# See: https://github.com/opera-adt/burst_db/tree/main/src/burst_db/data
# BASE_URL = "https://github.com/opera-adt/burst_db/raw/v{version}/src/burst_db/data"
# BASE_URL = "https://github.com/opera-adt/burst_db/raw/v0.3.0/src/burst_db/data"
BASE_URL = "https://github.com/opera-adt/burst_db/releases/download/v{version}/"

# $ ls *json.zip | xargs -n1 shasum -a 256
# df795cdad6e3f85c29aafd2ca4bbc60a7f011319eb428f0c036d49361a3cb205  burst-id-geometries-simple-0.5.0.geojson.zip
# 120560ea7af47c492477e2803c4a42ff3c7cb0dfe0d867d7a81ff0ef138ea05b  frame-geometries-simple-0.5.0.geojson.zip
# 2909b1065f41f753c203be8a567c2e6770a047f1c863f04db8efd2d13e353854  opera-s1-disp-0.5.0-burst-to-frame.json.zip
# 14d62b32a76a24937c8c73dc88ce60a0ea02c4458d102c9ba1b1191b147e045f  opera-s1-disp-0.5.0-frame-to-burst.json.zip

BURST_DB_VERSION = "0.5.0"

POOCH = pooch.create(
    # Folder where the data will be stored. For a sensible default, use the
    # default cache folder for your OS.
    path=pooch.os_cache("opera_utils"),
    # Base URL of the remote data store. Will call .format on this string
    # to insert the version (see below).
    base_url=BASE_URL,
    # Pooches are versioned so that you can use multiple versions of a
    # package simultaneously. Use PEP440 compliant version number. The
    # version will be appended to the path.
    version=BURST_DB_VERSION,
    # If a version as a "+XX.XXXXX" suffix, we'll assume that this is a dev
    # version and replace the version with this string.
    version_dev="main",
    # An environment variable that overwrites the path.
    env="OPERA_UTILS_DATA_DIR",
    # The cache file registry. A dictionary with all files managed by this
    # pooch. Keys are the file names (relative to *base_url*) and values
    # are their respective SHA256 hashes. Files will be downloaded
    # automatically when needed.
    registry={
        f"burst-id-geometries-simple-{BURST_DB_VERSION}.geojson.zip": "df795cdad6e3f85c29aafd2ca4bbc60a7f011319eb428f0c036d49361a3cb205",
        f"frame-geometries-simple-{BURST_DB_VERSION}.geojson.zip": "120560ea7af47c492477e2803c4a42ff3c7cb0dfe0d867d7a81ff0ef138ea05b",
        f"opera-s1-disp-{BURST_DB_VERSION}-burst-to-frame.json.zip": "2909b1065f41f753c203be8a567c2e6770a047f1c863f04db8efd2d13e353854",
        f"opera-s1-disp-{BURST_DB_VERSION}-frame-to-burst.json.zip": "14d62b32a76a24937c8c73dc88ce60a0ea02c4458d102c9ba1b1191b147e045f",
    },
)


def fetch_frame_geometries_simple() -> str:
    """Get the simplified frame geometries for the burst database."""
    return POOCH.fetch(f"frame-geometries-simple-{BURST_DB_VERSION}.geojson.zip")


def fetch_burst_id_geometries_simple() -> str:
    """Get the simplified burst ID geometries for the burst database."""
    return POOCH.fetch(f"burst-id-geometries-simple-{BURST_DB_VERSION}.geojson.zip")


def fetch_burst_to_frame_mapping_file() -> str:
    """Get the burst-to-frame mapping for the burst database."""
    return POOCH.fetch(f"opera-s1-disp-{BURST_DB_VERSION}-burst-to-frame.json.zip")


def fetch_frame_to_burst_mapping_file() -> str:
    """Get the frame-to-burst mapping for the burst database."""
    return POOCH.fetch(f"opera-s1-disp-{BURST_DB_VERSION}-frame-to-burst.json.zip")
