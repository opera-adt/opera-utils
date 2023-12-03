from click.testing import CliRunner

from opera_utils.cli import cli_app


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    assert result.output.startswith("Usage:")


def test_frame_bbox_help():
    runner = CliRunner()
    result = runner.invoke(cli_app, ["frame-bbox", "--help"])
    assert result.exit_code == 0
