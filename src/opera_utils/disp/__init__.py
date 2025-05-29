"""OPERA DISP-S1 utilities for reading and analyzing displacement data."""

from __future__ import annotations

from ._product import DispProduct, DispProductStack
from ._remote import open_file, open_h5
from ._search import search
from ._xarray import create_rebased_stack

__all__ = [
    "DispProduct",
    "DispProductStack",
    "create_rebased_stack",
    "open_file",
    "open_h5",
    "search",
]
