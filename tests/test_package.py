from __future__ import annotations

import importlib.metadata

import opera_utils as m


def test_version():
    assert importlib.metadata.version("opera_utils") == m.__version__
