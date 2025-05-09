"""Tests for the burst_frame_db module."""

import json
import zipfile
from unittest import mock

import pytest

import opera_utils
from opera_utils import burst_frame_db
from opera_utils._types import Bbox

# Sample mock data for tests
MOCK_FRAME_MAPPING = {
    "data": {
        "1": {
            "epsg": 32631,
            "is_land": False,
            "is_north_america": False,
            "xmin": 500160,
            "ymin": 78240,
            "xmax": 789960,
            "ymax": 322740,
            "burst_ids": [
                "t001_000001_iw1",
                "t001_000001_iw2",
                "t001_000001_iw3",
                "t001_000002_iw1",
                "t001_000002_iw2",
                "t001_000002_iw3",
                "t001_000003_iw1",
                "t001_000003_iw2",
                "t001_000003_iw3",
                "t001_000004_iw1",
                "t001_000004_iw2",
                "t001_000004_iw3",
                "t001_000005_iw1",
                "t001_000005_iw2",
                "t001_000005_iw3",
                "t001_000006_iw1",
                "t001_000006_iw2",
                "t001_000006_iw3",
                "t001_000007_iw1",
                "t001_000007_iw2",
                "t001_000007_iw3",
                "t001_000008_iw1",
                "t001_000008_iw2",
                "t001_000008_iw3",
                "t001_000009_iw1",
                "t001_000009_iw2",
                "t001_000009_iw3",
            ],
        },
        "2": {
            "epsg": 32631,
            "is_land": False,
            "is_north_america": False,
            "xmin": 469200,
            "ymin": 225720,
            "xmax": 758790,
            "ymax": 469860,
            "burst_ids": [
                "t001_000009_iw1",
                "t001_000009_iw2",
                "t001_000009_iw3",
                "t001_000010_iw1",
                "t001_000010_iw2",
                "t001_000010_iw3",
                "t001_000011_iw1",
                "t001_000011_iw2",
                "t001_000011_iw3",
                "t001_000012_iw1",
                "t001_000012_iw2",
                "t001_000012_iw3",
                "t001_000013_iw1",
                "t001_000013_iw2",
                "t001_000013_iw3",
                "t001_000014_iw1",
                "t001_000014_iw2",
                "t001_000014_iw3",
                "t001_000015_iw1",
                "t001_000015_iw2",
                "t001_000015_iw3",
                "t001_000016_iw1",
                "t001_000016_iw2",
                "t001_000016_iw3",
                "t001_000017_iw1",
                "t001_000017_iw2",
                "t001_000017_iw3",
            ],
        },
    }
}

MOCK_BURST_MAPPING = {
    "data": {
        "t001_000008_iw3": {"frame_ids": [1]},
        "t001_000009_iw1": {"frame_ids": [1, 2]},
        "t001_000009_iw2": {"frame_ids": [1, 2]},
    }
}

MOCK_FRAME_GEOJSON = {
    "type": "FeatureCollection",
    "name": "SELECT",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": [
        {
            "type": "Feature",
            "id": 1,
            "properties": {
                "is_land": 0,
                "is_north_america": False,
                "orbit_pass": "ASCENDING",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [3.046658, 2.282384],
                        [5.240285, 2.87221],
                        [5.560554, 1.337647],
                        [3.367581, 0.753216],
                        [3.046658, 2.282384],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "id": 2,
            "properties": {
                "is_land": 0,
                "is_north_america": False,
                "orbit_pass": "ASCENDING",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [2.767839, 3.616908],
                        [4.965271, 4.202952],
                        [5.282563, 2.668107],
                        [3.0872, 2.087587],
                        [2.767839, 3.616908],
                    ]
                ],
            },
        },
    ],
}

MOCK_BURST_GEOJSON = {
    "type": "FeatureCollection",
    "name": "SELECT",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": [
        {
            "type": "Feature",
            "id": 1,
            "properties": {
                "burst_id_jpl": "t001_000001_iw1",
                "is_land": 0,
                "is_north_america": False,
                "orbit_pass": "ASCENDING",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [3.367581, 0.753216, 0.0],
                        [4.129596, 0.911666, 0.0],
                        [4.087013, 1.115115, 0.0],
                        [3.326852, 0.947997, 0.0],
                        [3.367581, 0.753216, 0.0],
                    ]
                ],
            },
        },
        {
            "type": "Feature",
            "id": 2,
            "properties": {
                "burst_id_jpl": "t001_000001_iw2",
                "is_land": 0,
                "is_north_america": False,
                "orbit_pass": "ASCENDING",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [4.091781, 0.974022, 0.0],
                        [4.890103, 1.140293, 0.0],
                        [4.849909, 1.332363, 0.0],
                        [4.05327, 1.158069, 0.0],
                        [4.091781, 0.974022, 0.0],
                    ]
                ],
            },
        },
    ],
}


@pytest.fixture
def mock_dataset_files(tmp_path):
    """Create mock dataset files for testing."""
    frame_map_path = tmp_path / "frame_to_burst.json.zip"
    burst_map_path = tmp_path / "burst_to_frame.json.zip"
    frame_geo_path = tmp_path / "frame_geometries.json.zip"
    burst_geo_path = tmp_path / "burst_geometries.json.zip"

    # Create the zip files with mock data
    with zipfile.ZipFile(frame_map_path, "w") as zf:
        zf.writestr("frame_to_burst.json", json.dumps(MOCK_FRAME_MAPPING))

    with zipfile.ZipFile(burst_map_path, "w") as zf:
        zf.writestr("burst_to_frame.json", json.dumps(MOCK_BURST_MAPPING))

    with zipfile.ZipFile(frame_geo_path, "w") as zf:
        zf.writestr("frame_geometries.json", json.dumps(MOCK_FRAME_GEOJSON))

    with zipfile.ZipFile(burst_geo_path, "w") as zf:
        zf.writestr("burst_geometries.json", json.dumps(MOCK_BURST_GEOJSON))

    return {
        "frame_map": frame_map_path,
        "burst_map": burst_map_path,
        "frame_geo": frame_geo_path,
        "burst_geo": burst_geo_path,
    }


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("test.json", {"key": "value"}),
        ("test.json.zip", {"key": "value"}),
    ],
)
def test_read_zipped_json(tmp_path, filename, expected):
    """Test reading from a zipped or unzipped JSON file."""
    if filename.endswith(".zip"):
        path = tmp_path / filename
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(filename.replace(".zip", ""), json.dumps(expected))
    else:
        path = tmp_path / filename
        with open(path, "w") as f:
            json.dump(expected, f)

    result = burst_frame_db.read_zipped_json(path)
    assert result == expected


def test_get_frame_to_burst_mapping(mock_dataset_files):
    """Test getting frame to burst mapping."""
    with mock.patch(
        "opera_utils.datasets.fetch_frame_to_burst_mapping_file",
        return_value=mock_dataset_files["frame_map"],
    ):
        result = burst_frame_db.get_frame_to_burst_mapping(1)

        assert result["epsg"] == 32631
        assert "t001_000001_iw1" in result["burst_ids"]
        assert "t001_000001_iw2" in result["burst_ids"]
        assert "t001_000001_iw3" in result["burst_ids"]


def test_get_burst_to_frame_mapping(mock_dataset_files):
    """Test getting burst to frame mapping."""
    with mock.patch(
        "opera_utils.datasets.fetch_burst_to_frame_mapping_file",
        return_value=mock_dataset_files["burst_map"],
    ):
        result = burst_frame_db.get_burst_to_frame_mapping("t001_000008_iw3")

        assert result["frame_ids"] == [1]


def test_get_frame_bbox(mock_dataset_files):
    """Test getting frame bounding box."""
    with mock.patch(
        "opera_utils.datasets.fetch_frame_to_burst_mapping_file",
        return_value=mock_dataset_files["frame_map"],
    ):
        epsg, bbox = burst_frame_db.get_frame_bbox(1)

        assert epsg == 32631
        assert bbox.left == 500160.0
        assert bbox.bottom == 78240.0
        assert bbox.right == 789960.0
        assert bbox.top == 322740.0


def test_get_burst_ids_for_frame(mock_dataset_files):
    """Test getting burst IDs for a frame."""
    with mock.patch(
        "opera_utils.datasets.fetch_frame_to_burst_mapping_file",
        return_value=mock_dataset_files["frame_map"],
    ):
        burst_ids = burst_frame_db.get_burst_ids_for_frame(1)

        assert "t001_000001_iw1" in burst_ids
        assert "t001_000001_iw2" in burst_ids
        assert "t001_000008_iw3" in burst_ids
        assert len(burst_ids) == 27


def test_get_frame_ids_for_burst(mock_dataset_files):
    """Test getting frame IDs for a burst."""
    with mock.patch(
        "opera_utils.datasets.fetch_burst_to_frame_mapping_file",
        return_value=mock_dataset_files["burst_map"],
    ):
        frame_ids = burst_frame_db.get_frame_ids_for_burst("t001_000009_iw1")

        assert 1 in frame_ids
        assert 2 in frame_ids
        assert len(frame_ids) == 2


def test_get_frame_geojson_dict(mock_dataset_files):
    """Test getting frame GeoJSON as dict."""
    with mock.patch(
        "opera_utils.datasets.fetch_frame_geometries_simple",
        return_value=mock_dataset_files["frame_geo"],
    ):
        result = burst_frame_db.get_frame_geojson(as_geodataframe=False)

        assert result["type"] == "FeatureCollection"
        assert len(result["features"]) == 2

        # Test filtering
        result = burst_frame_db.get_frame_geojson([1], as_geodataframe=False)
        assert len(result["features"]) == 1
        assert result["features"][0]["id"] == 1


def test_get_burst_geojson_dict(mock_dataset_files):
    """Test getting burst GeoJSON as dict."""
    with mock.patch(
        "opera_utils.datasets.fetch_burst_id_geometries_simple",
        return_value=mock_dataset_files["burst_geo"],
    ):
        result = burst_frame_db.get_burst_id_geojson(as_geodataframe=False)

        assert result["type"] == "FeatureCollection"
        assert len(result["features"]) == 2

        # Test filtering
        result = burst_frame_db.get_burst_id_geojson(
            ["t001_000001_iw1"], as_geodataframe=False
        )
        assert len(result["features"]) == 1
        assert result["features"][0]["properties"]["burst_id_jpl"] == "t001_000001_iw1"


def test_get_frame_orbit_pass(mock_dataset_files):
    """Test getting frame to burst mapping."""
    with mock.patch(
        "opera_utils.datasets.fetch_burst_id_geometries_simple",
        return_value=mock_dataset_files["burst_geo"],
    ):
        result = burst_frame_db.get_frame_orbit_pass(1)
        assert result == [burst_frame_db.OrbitPass("ASCENDING")]

        result = burst_frame_db.get_frame_orbit_pass([1, 2])
        assert result == [
            burst_frame_db.OrbitPass("ASCENDING"),
            burst_frame_db.OrbitPass("ASCENDING"),
        ]


# Tests that require geopandas
@pytest.mark.skipif(
    not opera_utils.burst_frame_db._has_geopandas,
    reason="geopandas and pyogrio are required for these tests",
)
class TestGeopandasIntegration:
    """Tests for functions that require geopandas."""

    def test_get_frame_geodataframe(self, mock_dataset_files):
        """Test getting frame geometries as GeoDataFrame."""
        with (
            mock.patch(
                "opera_utils.datasets.fetch_frame_geometries_simple",
                return_value=mock_dataset_files["frame_geo"],
            ),
            mock.patch("pyogrio.read_dataframe") as mock_read,
        ):
            burst_frame_db.get_frame_geodataframe()
            mock_read.assert_called_once()

    def test_get_burst_geodataframe(self, mock_dataset_files):
        """Test getting burst geometries as GeoDataFrame."""
        with (
            mock.patch(
                "opera_utils.datasets.fetch_burst_id_geometries_simple",
                return_value=mock_dataset_files["burst_geo"],
            ),
            mock.patch("pyogrio.read_dataframe") as mock_read,
        ):
            burst_frame_db.get_burst_geodataframe()
            mock_read.assert_called_once()

    def test_get_intersecting_frames(self):
        """Test getting intersecting frames."""
        with mock.patch(
            "opera_utils.burst_frame_db.get_frame_geodataframe"
        ) as mock_get_frame:
            import geopandas as gpd
            from shapely.geometry import box

            # Create mock GeoDataFrame
            mock_gdf = gpd.GeoDataFrame(
                {"frame_id": [8622, 8623]},
                geometry=[box(100, 200, 300, 400), box(200, 300, 400, 500)],
            )
            mock_gdf.index = [8622, 8623]
            mock_gdf.index.name = "frame_id"

            # Set up mock to return the fake GeoDataFrame
            mock_get_frame.return_value = mock_gdf

            result = burst_frame_db.get_intersecting_frames(
                Bbox(150, 250, 350, 450),
            )
            assert isinstance(result, dict)
            assert "features" in result
