"""
OPERA DISP-S1 utilities for reading and analyzing displacement data.
"""

from __future__ import annotations

from ._product import DispProduct, DispProductStack
from ._utils import get_frame_coordinates

# Remote access is based on optional dependencies
try:
    from ._remote import open_h5
except ImportError:
    pass

# Import reader module for higher-level API
from . import _reader as reader

__all__ = [
    "DispProduct",
    "DispProductStack",
    "open_h5",
    "get_frame_coordinates",
    "reader",
]
