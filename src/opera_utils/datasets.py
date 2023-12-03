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
# 8ee9cae079b9adb24e223b9ff9c81c66506a2a1a72a456220133a9f7f5d4d93b  burst_id_geometries_simple.geojson.zip
# 86657e4e578cfced18a66984758fff9a1bf94e8591a288be0d1ad391399f2e59  frame_geometries_simple.geojson.zip
# 436cce345378dc31e81ed661497bab2e744217a5d63c0bb92817dc837786cd22  opera-s1-disp-burst-to-frame.json.zip
# 8b7ed8c8d90ef3d3348bc226958a26a2cb8ab302a6466762aa971b8f7333517f  opera-s1-disp-frame-to-burst.json.zip

BURST_DB_VERSION = "0.3.1"

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
        f"frame-geometries-simple-{BURST_DB_VERSION}.geojson.zip": "963f63577221a3baa20f3a2101c7a01eefb0cc853f6f111708a5bb35bebfc0ed",
        f"burst-id-geometries-simple-{BURST_DB_VERSION}.geojson.zip": "e75cc27809448d7ace2164879626fb0b5616b16981a6b2d6d234e3b17cb615fa",
        f"opera-s1-disp-burst-to-frame-{BURST_DB_VERSION}.json.zip": "436cce345378dc31e81ed661497bab2e744217a5d63c0bb92817dc837786cd22",
        f"opera-s1-disp-frame-to-burst-{BURST_DB_VERSION}.json.zip": "a48382afcb89f0ff681982b0fc24476ec9c6c1b8a67ae1a26cf380a450ffadc0",
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
    return POOCH.fetch(f"opera-s1-disp-burst-to-frame-{BURST_DB_VERSION}.json.zip")


def fetch_frame_to_burst_mapping_file() -> str:
    """Get the frame-to-burst mapping for the burst database."""
    return POOCH.fetch(f"opera-s1-disp-frame-to-burst-{BURST_DB_VERSION}.json.zip")
