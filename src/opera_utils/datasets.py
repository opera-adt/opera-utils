from __future__ import annotations

import pooch

__all__ = [
    "fetch_frame_geometries_simple",
    "fetch_burst_id_geometries_simple",
    "fetch_burst_to_frame_mapping_file",
    "fetch_frame_to_burst_mapping_file",
]

BASE_URL = "https://github.com/opera-adt/burst_db/releases/download/v{version}/"

# $ ls *json.zip | xargs -n1 shasum -a 256
# 0b3119030c7fde89be4afaa713455ea021ea1f1bd7563b02aabeecafd9b4c0f0  burst-id-geometries-simple-0.7.0.geojson.zip
# e6d578cc1be77d0b8c1f400d52f425bd4dc29d01b3a0057cf5f85ec5e5519d76  frame-geometries-simple-0.7.0.geojson.zip
# 0611d8807c4a6a9b0dfe5f0906e47f455b924d19a8a42fcfcab7f0f09f717a75  opera-s1-disp-0.7.0-burst-to-frame.json.zip
# 22ac2854fd17ed56e90eb71ab27d03f7157e013b6bd982ca512a5212cc136f62  opera-s1-disp-0.7.0-frame-to-burst.json.zip

BURST_DB_VERSION = "0.7.0"

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
        f"burst-id-geometries-simple-{BURST_DB_VERSION}.geojson.zip": "0b3119030c7fde89be4afaa713455ea021ea1f1bd7563b02aabeecafd9b4c0f0",
        f"frame-geometries-simple-{BURST_DB_VERSION}.geojson.zip": "e6d578cc1be77d0b8c1f400d52f425bd4dc29d01b3a0057cf5f85ec5e5519d76",
        f"opera-s1-disp-{BURST_DB_VERSION}-burst-to-frame.json.zip": "0611d8807c4a6a9b0dfe5f0906e47f455b924d19a8a42fcfcab7f0f09f717a75",
        f"opera-s1-disp-{BURST_DB_VERSION}-frame-to-burst.json.zip": "22ac2854fd17ed56e90eb71ab27d03f7157e013b6bd982ca512a5212cc136f62",
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
