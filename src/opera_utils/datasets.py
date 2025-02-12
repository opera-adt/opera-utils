from __future__ import annotations

import pooch

__all__ = [
    "fetch_frame_geometries_simple",
    "fetch_burst_id_geometries_simple",
    "fetch_burst_to_frame_mapping_file",
    "fetch_frame_to_burst_mapping_file",
]

BASE_URL = "https://github.com/opera-adt/burst_db/releases/download/v{version}/"

BURST_DB_VERSION = "0.8.0"

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
    # $ cd test_080/ && ls *json.zip | xargs -n1 shasum -a 256
    # 123f5c415bbec322145b831d6dbe37fa7725fc5aaba6f2f6cf9b4eb73ddbc96a  burst-id-geometries-simple-0.8.0.geojson.zip
    # 9a5f9bb1184527a143898fce846074b460810bde26d20f3d356b24b7faddfeff  frame-geometries-simple-0.8.0.geojson.zip
    # 4282f65fb8a08196d65633accce8ff5115b6623e941f8c76d973e8233af988e5  opera-s1-disp-0.8.0-burst-to-frame.json.zip
    # d457e8eb081af4d38f817ee2b34808ecbf01e28cc0de7f435d52999e427d631e  opera-s1-disp-0.8.0-frame-to-burst.json.zip
    registry={
        f"burst-id-geometries-simple-{BURST_DB_VERSION}.geojson.zip": "123f5c415bbec322145b831d6dbe37fa7725fc5aaba6f2f6cf9b4eb73ddbc96a",
        f"frame-geometries-simple-{BURST_DB_VERSION}.geojson.zip": "9a5f9bb1184527a143898fce846074b460810bde26d20f3d356b24b7faddfeff",
        f"opera-s1-disp-{BURST_DB_VERSION}-burst-to-frame.json.zip": "4282f65fb8a08196d65633accce8ff5115b6623e941f8c76d973e8233af988e5",
        f"opera-s1-disp-{BURST_DB_VERSION}-frame-to-burst.json.zip": "d457e8eb081af4d38f817ee2b34808ecbf01e28cc0de7f435d52999e427d631e",
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
