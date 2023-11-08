from __future__ import annotations

import re

__all__ = [
    "OPERA_DATASET_NAME",
    "OPERA_IDENTIFICATION",
    "OPERA_BURST_RE",
]

# Specific to OPERA CSLC products:
OPERA_DATASET_NAME = "/data/VV"
OPERA_IDENTIFICATION = "/identification"

# It should match either or these within a filename:
# t087_185684_iw2 (which comes from COMPASS)
# T087-165495-IW3 (which is the official product naming scheme)
# e.g.
# OPERA_L2_CSLC-S1_T078-165495-IW3_20190906T232711Z_20230101T100506Z_S1A_VV_v1.0.h5

OPERA_BURST_RE = re.compile(
    r"[tT](?P<track>\d{3})[-_](?P<burst_id>\d{6})[-_](?P<subswath>iw[1-3])",
    re.IGNORECASE,
)
