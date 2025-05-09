import json
import sys

import pytest

from opera_utils.cli import cli_app

FRAME_ID = 11115


def test_help(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["opera-utils", "--help"])

        with pytest.raises(SystemExit):
            cli_app()
        s = capsys.readouterr()
        assert s.out.startswith("usage: opera-utils")


def test_frame_bbox_help(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["opera-utils", "disp-s1-frame-bbox", "--help"])

        with pytest.raises(SystemExit):
            cli_app()
        s = capsys.readouterr()
        assert "Print the DISP-S1 EPSG/bounding box for FRAME_ID" in s.out


def test_frame_bbox_basic(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["opera-utils", "disp-s1-frame-bbox", str(FRAME_ID)])

        cli_app()
        s = capsys.readouterr()
        try:
            data = json.loads(s.out)
            assert "epsg" in data
            assert "bbox" in data
            assert isinstance(data["bbox"], list)
            assert len(data["bbox"]) == 4
        except json.JSONDecodeError:
            pytest.fail(f"Output was not valid JSON: {s.out}")


def test_frame_bbox_latlon(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    with monkeypatch.context() as m:
        m.setattr(
            sys,
            "argv",
            ["opera-utils", "disp-s1-frame-bbox", str(FRAME_ID), "--latlon"],
        )

        cli_app()
        s = capsys.readouterr()
        data = json.loads(s.out)
        assert "epsg" in data
        assert "bbox" in data
        assert isinstance(data["bbox"], list)
        assert len(data["bbox"]) == 4


def test_frame_bbox_bounds_only(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    with monkeypatch.context() as m:
        m.setattr(
            sys,
            "argv",
            ["opera-utils", "disp-s1-frame-bbox", str(FRAME_ID), "--bounds-only"],
        )

        cli_app()
        s = capsys.readouterr()
        # Expecting just a list like [1, 2, 3, 4]
        data = json.loads(s.out)
        assert isinstance(data, list)
        assert len(data) == 4
        assert isinstance(data[0], (int, float))  # Check type of elements


def test_frame_bbox_missing_arg(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    with monkeypatch.context() as m:
        # Missing frame_id argument
        m.setattr(sys, "argv", ["opera-utils", "disp-s1-frame-bbox"])

        with pytest.raises(SystemExit):
            cli_app()
        s = capsys.readouterr()
        assert "arguments are required: INT" in s.err


def test_intersects_help(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["opera-utils", "disp-s1-intersects", "--help"])

        with pytest.raises(SystemExit):
            cli_app()
        s = capsys.readouterr()
        assert (
            "Get the DISP-S1 frames that intersect with the given bounding box" in s.out
        )
        assert "--bbox" in s.out
        assert "--point" in s.out
        assert "--ids-only" in s.out


def test_intersects_bbox(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    with monkeypatch.context() as m:
        # Use plausible coordinates
        bbox = ["--bbox", "-114", "31", "-113", "32"]
        m.setattr(sys, "argv", ["opera-utils", "disp-s1-intersects", *bbox])

        cli_app()
        s = capsys.readouterr()
        json.loads(s.out)


def test_intersects_point(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    with monkeypatch.context() as m:
        point = ["--point", "-114.1", "31.5"]
        m.setattr(sys, "argv", ["opera-utils", "disp-s1-intersects", *point])

        cli_app()
        s = capsys.readouterr()
        result = json.loads(s.out)
        # check whole first result
        assert result["features"][0] == {
            "id": "24725",
            "type": "Feature",
            "properties": {
                "is_land": 1,
                "is_north_america": True,
                "orbit_pass": "ASCENDING",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-114.442868, 32.723514],
                        [-111.812723, 33.244115],
                        [-111.51108, 31.711421],
                        [-114.098757, 31.198782],
                        [-114.442868, 32.723514],
                    ]
                ],
            },
        }
        # check ids
        assert result["features"][0]["id"] == "24725"
        assert result["features"][1]["id"] == "26692"
        assert result["features"][2]["id"] == "44324"


def test_intersects_ids_only(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    with monkeypatch.context() as m:
        bbox = ["--bbox", "-114.1", "31.5", "-113.1", "32.5", "--ids-only"]
        m.setattr(sys, "argv", ["opera-utils", "disp-s1-intersects", *bbox])

        cli_app()
        s = capsys.readouterr()
        result_ids = list(map(int, s.out.strip().split("\n")))
        assert result_ids == [7091, 7092, 24724, 24725, 26691, 26692, 44324, 44325]


def test_intersects_missing_arg(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    with monkeypatch.context() as m:
        # Neither bbox nor point provided
        m.setattr(sys, "argv", ["opera-utils", "disp-s1-intersects"])

        # Now that we're using keyword-only args with *, our function raises ValueError directly
        with pytest.raises(ValueError, match="Either bbox or point must be provided"):
            cli_app()


def test_intersects_point_geojson(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """Verify output for a specific point intersection (GeoJSON)."""
    with monkeypatch.context() as m:
        point = ["--point", "-114", "31"]  # Longitude, Latitude
        expected_ids = {
            "24724",
            "26692",
            "44324",
        }  # Use a set for order-independent comparison

        m.setattr(sys, "argv", ["opera-utils", "disp-s1-intersects", *point])

        cli_app()
        s = capsys.readouterr()

        try:
            data = json.loads(s.out)
        except json.JSONDecodeError:
            pytest.fail(f"Output was not valid JSON: {s.out}")

        # Check top-level GeoJSON structure
        assert isinstance(data, dict), "Output should be a JSON object"
        assert (
            data.get("type") == "FeatureCollection"
        ), "JSON 'type' should be FeatureCollection"
        assert "features" in data, "JSON should have a 'features' key"
        assert isinstance(data["features"], list), "'features' should be a list"

        # Check the number of features found
        assert len(data["features"]) == len(
            expected_ids
        ), f"Expected {len(expected_ids)} features, found {len(data['features'])}"

        # Extract IDs and compare with expected IDs
        actual_ids = set()
        for feature in data["features"]:
            assert "id" in feature, "Each feature must have an 'id'"
            # Check basic feature structure
            assert feature.get("type") == "Feature"
            assert "properties" in feature
            assert "geometry" in feature
            assert isinstance(feature["geometry"], dict)
            assert (
                feature["geometry"].get("type") == "Polygon"
            )  # Based on example output

            actual_ids.add(
                str(feature["id"])
            )  # Ensure ID is treated as string if needed

        assert (
            actual_ids == expected_ids
        ), f"Expected IDs {expected_ids}, but found {actual_ids}"


def test_intersects_point_ids_only(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """Verify output for a specific point intersection (--ids-only)."""
    with monkeypatch.context() as m:
        point = [
            "--point",
            "-114",
            "31",
            "--ids-only",
        ]  # Longitude, Latitude as strings for CLI
        expected_ids = {
            "24724",
            "26692",
            "44324",
        }  # Use a set for order-independent comparison

        m.setattr(sys, "argv", ["opera-utils", "disp-s1-intersects", *point])

        cli_app()
        s = capsys.readouterr()

        # Output should be newline-separated IDs
        # Remove leading/trailing whitespace and split into lines
        # Filter out empty lines just in case
        actual_ids = {line for line in s.out.strip().splitlines() if line}

        assert (
            actual_ids == expected_ids
        ), f"Expected IDs {expected_ids}, but found {actual_ids}"


def test_missing_data_options_help(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    with monkeypatch.context() as m:
        m.setattr(
            sys, "argv", ["opera-utils", "disp-s1-missing-data-options", "--help"]
        )

        with pytest.raises(SystemExit):
            cli_app()
        s = capsys.readouterr()
        assert "Get a list of options for how to handle missing S1 data" in s.out


def test_missing_data_options_missing_arg(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["opera-utils", "disp-s1-missing-data-options"])

        with pytest.raises(SystemExit):
            cli_app()
        s = capsys.readouterr()
        assert "arguments are required: STR" in s.err


def test_invalid_command(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", ["opera-utils", "nonexistent-command"])

        with pytest.raises(SystemExit):
            cli_app()
        s = capsys.readouterr()
        assert "invalid choice" in s.err


def test_invalid_option(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    with monkeypatch.context() as m:
        frame_id = "123"
        m.setattr(
            sys,
            "argv",
            ["opera-utils", "disp-s1-frame-bbox", frame_id, "--nonexistent-option"],
        )

        with pytest.raises(SystemExit):
            cli_app()
        s = capsys.readouterr()
        assert "Unrecognized options" in s.err
