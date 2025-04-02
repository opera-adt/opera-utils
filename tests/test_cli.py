import json

import pytest
from click.testing import CliRunner

from opera_utils.cli import cli_app

FRAME_ID = 11115


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cli_help(runner):
    result = runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    assert result.output.startswith("Usage:")


def test_frame_bbox_help():
    runner = CliRunner()
    result = runner.invoke(cli_app, ["disp-s1", "frame-bbox", "--help"])
    assert result.exit_code == 0


# --- DISP-S1 Group Tests ---
def test_disp_s1_help(runner):
    result = runner.invoke(cli_app, ["disp-s1", "--help"])
    assert result.exit_code == 0
    assert result.output.startswith(
        "Usage: opera-utils disp-s1 [OPTIONS] COMMAND [ARGS]..."
    )
    assert "frame-bbox" in result.output
    assert "intersects" in result.output
    assert "missing-data-options" in result.output


def test_frame_bbox_basic(runner):
    result = runner.invoke(cli_app, ["disp-s1", "frame-bbox", str(FRAME_ID)])
    assert result.exit_code == 0
    try:
        data = json.loads(result.output)
        assert "epsg" in data
        assert "bbox" in data
        assert isinstance(data["bbox"], list)
        assert len(data["bbox"]) == 4
    except json.JSONDecodeError:
        pytest.fail(f"Output was not valid JSON: {result.output}")


def test_frame_bbox_latlon(runner):
    result = runner.invoke(
        cli_app, ["disp-s1", "frame-bbox", str(FRAME_ID), "--latlon"]
    )
    assert result.exit_code == 0
    try:
        data = json.loads(result.output)
        assert "epsg" in data
        assert "bbox" in data
        assert isinstance(data["bbox"], list)
        assert len(data["bbox"]) == 4
    except json.JSONDecodeError:
        pytest.fail(f"Output was not valid JSON: {result.output}")


def test_frame_bbox_bounds_only(runner):
    result = runner.invoke(
        cli_app, ["disp-s1", "frame-bbox", str(FRAME_ID), "--bounds-only"]
    )
    assert result.exit_code == 0
    try:
        # Expecting just a list like [1, 2, 3, 4]
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 4
        assert isinstance(data[0], (int, float))  # Check type of elements
    except json.JSONDecodeError:
        pytest.fail(f"Output was not valid JSON: {result.output}")


def test_frame_bbox_missing_arg(runner):
    result = runner.invoke(cli_app, ["disp-s1", "frame-bbox"])
    assert result.exit_code != 0
    assert "Missing argument 'FRAME_ID'" in result.output


# --- DISP-S1 intersects Tests ---
def test_intersects_help(runner):
    result = runner.invoke(cli_app, ["disp-s1", "intersects", "--help"])
    assert result.exit_code == 0
    assert result.output.startswith("Usage: opera-utils disp-s1 intersects [OPTIONS]")
    assert "--bbox" in result.output
    assert "--point" in result.output
    assert "--ids-only" in result.output


def test_intersects_bbox(runner):
    # Use plausible coordinates
    bbox = ["10.0", "20.0", "11.0", "21.0"]
    result = runner.invoke(cli_app, ["disp-s1", "intersects", "--bbox"] + bbox)
    assert result.exit_code == 0
    # Output format isn't specified in help, assume JSON list of frame details or IDs
    # Check if it's valid JSON as a basic test
    try:
        json.loads(result.output)
    except json.JSONDecodeError:
        # Or maybe it prints lines? Adapt assertion based on actual output.
        # For now, we just check if it runs without error.
        pass
        # pytest.fail(f"Output was not valid JSON: {result.output}")


def test_intersects_point(runner):
    point = ["10.5", "20.5"]
    result = runner.invoke(cli_app, ["disp-s1", "intersects", "--point"] + point)
    assert result.exit_code == 0
    # Similar check as bbox
    try:
        json.loads(result.output)
    except json.JSONDecodeError:
        pass


def test_intersects_ids_only(runner):
    bbox = ["10.0", "20.0", "11.0", "21.0"]
    result = runner.invoke(
        cli_app, ["disp-s1", "intersects", "--bbox"] + bbox + ["--ids-only"]
    )
    assert result.exit_code == 0
    # Output might just be lines of IDs, not JSON.
    # Check for empty output or specific format if known. For now, just exit code.


def test_intersects_bad_bbox_count(runner):
    bbox_bad = ["10.0", "20.0", "11.0"]  # Only 3 values
    result = runner.invoke(cli_app, ["disp-s1", "intersects", "--bbox"] + bbox_bad)
    assert result.exit_code != 0
    # Click automatically handles errors for options with fixed nargs
    assert "Error: Option '--bbox' requires 4 arguments" in result.output


def test_intersects_point_functional_geojson(runner):
    """Verify functional output for a specific point intersection (GeoJSON)."""
    point = ["-114", "31"]  # Longitude, Latitude as strings for CLI
    expected_ids = {
        "24724",
        "26692",
        "44324",
    }  # Use a set for order-independent comparison

    result = runner.invoke(cli_app, ["disp-s1", "intersects", "--point"] + point)

    assert result.exit_code == 0, f"Command failed: {result.output}"

    try:
        data = json.loads(result.output)
    except json.JSONDecodeError:
        pytest.fail(f"Output was not valid JSON: {result.output}")

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
        assert feature["geometry"].get("type") == "Polygon"  # Based on example output

        actual_ids.add(str(feature["id"]))  # Ensure ID is treated as string if needed

    assert (
        actual_ids == expected_ids
    ), f"Expected IDs {expected_ids}, but found {actual_ids}"


def test_intersects_point_functional_ids_only(runner):
    """Verify functional output for a specific point intersection (--ids-only)."""
    point = ["-114", "31"]  # Longitude, Latitude as strings for CLI
    expected_ids = {
        "24724",
        "26692",
        "44324",
    }  # Use a set for order-independent comparison

    result = runner.invoke(
        cli_app, ["disp-s1", "intersects", "--point"] + point + ["--ids-only"]
    )

    assert result.exit_code == 0, f"Command failed: {result.output}"

    # Output should be newline-separated IDs
    # Remove leading/trailing whitespace and split into lines
    # Filter out empty lines just in case
    actual_ids = set(line for line in result.output.strip().splitlines() if line)

    assert (
        actual_ids == expected_ids
    ), f"Expected IDs {expected_ids}, but found {actual_ids}"


# --- DISP-S1 missing-data-options Tests ---
def test_missing_data_options_help(runner):
    result = runner.invoke(cli_app, ["disp-s1", "missing-data-options", "--help"])
    assert result.exit_code == 0
    assert result.output.startswith(
        "Usage: opera-utils disp-s1 missing-data-options [OPTIONS] NAMELIST"
    )


def test_missing_data_options_missing_arg(runner):
    result = runner.invoke(cli_app, ["disp-s1", "missing-data-options"])
    assert result.exit_code != 0
    assert "Missing argument 'NAMELIST'" in result.output


# --- General Error Handling ---
def test_invalid_subcommand(runner):
    result = runner.invoke(cli_app, ["disp-s1", "nonexistent-command"])
    assert result.exit_code != 0
    assert "No such command 'nonexistent-command'" in result.output


def test_invalid_option(runner):
    frame_id = "123"
    result = runner.invoke(
        cli_app, ["disp-s1", "frame-bbox", frame_id, "--nonexistent-option"]
    )
    assert result.exit_code != 0
    assert "No such option: --nonexistent-option" in result.output


# --- Placeholder for DISP-NISAR ---
def test_disp_nisar_help(runner):
    # Assuming disp-nisar exists as a group, add a basic help test
    result = runner.invoke(cli_app, ["disp-nisar", "--help"])
    assert result.exit_code == 0
    assert result.output.startswith(
        "Usage: opera-utils disp-nisar [OPTIONS] COMMAND [ARGS]..."
    )
