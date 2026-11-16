"""OPERA Studio: a local web UI to search, subset and download OPERA products.

Run ``python -m opera_utils.studio`` and open the printed address.

The studio covers DISP-S1 (with rebasing and point re-referencing), CSLC-S1,
NISAR (GSLC, RSLC, GUNW), TROPO and the DISP-S1 / CSLC-S1 static layers.
Searches run in the server process; downloads run as subprocesses (one per
job) so they can be cancelled, and so Earthdata credentials entered in the UI
reach every download path through a private ``.netrc``.
"""

from __future__ import annotations

__all__ = ["main"]


def main() -> None:
    """Start the studio server (see ``python -m opera_utils.studio --help``)."""
    from .server import main as _main

    _main()
