from pathlib import Path

from opera_utils._utils import scratch_directory


def test_scratch_directory(tmp_path):
    test_dir = tmp_path / "test_dir"
    with scratch_directory(test_dir) as d:
        (d / "test1.txt").write_text("asf")
    assert not Path(test_dir).exists()

    with scratch_directory(test_dir, delete=True) as d:
        (d / "test1.txt").write_text("asdf")
    assert Path(test_dir).exists()
    assert Path(test_dir / "test1.txt").read_text() == "asdf"


def test_scratch_directory_already_exists(tmp_path):
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    assert Path(test_dir).exists()

    with scratch_directory(test_dir, delete=True) as d:
        (d / "test1.txt").write_text("asdf")
    assert Path(test_dir).exists()

    with scratch_directory(test_dir, delete=False) as d:
        (d / "test1.txt").write_text("asdf")
    assert Path(test_dir).exists()
