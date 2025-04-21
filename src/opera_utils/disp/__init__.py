"""OPERA DISP-S1 utilities for reading and analyzing displacement data."""

from __future__ import annotations

from . import _reader as reader
from ._product import DispProduct, DispProductStack
from ._search import search

# Remote access is based on optional dependencies
try:
    from ._remote import open_file, open_h5
except ImportError:
    pass


__all__ = [
    "DispProduct",
    "DispProductStack",
    "open_h5",
    "open_file",
    "reader",
    "search",
]
