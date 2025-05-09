from __future__ import annotations

import pooch

__all__ = [
    "fetch_burst_id_geometries_simple",
    "fetch_burst_to_frame_mapping_file",
    "fetch_frame_geometries_simple",
    "fetch_frame_to_burst_mapping_file",
]

BASE_URL = "https://github.com/opera-adt/burst_db/releases/download/v{version}/"

BURST_DB_VERSION = "0.9.0"

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
    # $ cd test_090/ && ls *json.zip | xargs -n1 shasum -a 256
    # 4d9a046029e0dbe03a0746f5e7df645d4746e134676751085f55a341eb23f466  burst-id-geometries-simple-0.9.0.geojson.zip
    # 115182c420a0446f3f015c16e623fe9535337679796e44a65cf6154392f66eb2  frame-geometries-simple-0.9.0.geojson.zip
    # 93c458a6324970366d65e3639554d05e9ed46947dee2d32e2728e098336b8c9a  opera-s1-disp-0.9.0-burst-to-frame.json.zip
    # 0a0662d47f10e49dc20f1809407916b87565d1ed33c988ba86a3d6547bb4a28f  opera-s1-disp-0.9.0-frame-to-burst.json.zip
    registry={
        f"burst-id-geometries-simple-{BURST_DB_VERSION}.geojson.zip": (
            "4d9a046029e0dbe03a0746f5e7df645d4746e134676751085f55a341eb23f466"
        ),
        f"frame-geometries-simple-{BURST_DB_VERSION}.geojson.zip": (
            "115182c420a0446f3f015c16e623fe9535337679796e44a65cf6154392f66eb2"
        ),
        f"opera-s1-disp-{BURST_DB_VERSION}-burst-to-frame.json.zip": (
            "93c458a6324970366d65e3639554d05e9ed46947dee2d32e2728e098336b8c9a"
        ),
        f"opera-s1-disp-{BURST_DB_VERSION}-frame-to-burst.json.zip": (
            "0a0662d47f10e49dc20f1809407916b87565d1ed33c988ba86a3d6547bb4a28f"
        ),
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
