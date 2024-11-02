from __future__ import annotations

import re

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
CSLC_S1_FILE_REGEX = (
    r"(?P<project>OPERA)_"
    r"(?P<level>L2)_"
    r"(?P<product_type>CSLC-S1)_"
    r"(?P<burst_id>T\d{3}-\d+-IW\d)_"
    r"(?P<start_datetime>\d{8}T\d{6}Z)_"
    r"(?P<end_datetime>\d{8}T\d{6}Z)_"
    r"(?P<sensor>S1[AB])_"
    r"(?P<polarization>VV|HH)_"
    r"v(?P<product_version>\d+\.\d+)"
)
# https://github.com/opera-adt/COMPASS/blob/16a3c1da2a5db69b9e2007d798a1110d3a6c5f9f/src/compass/utils/runconfig.py#L316-L318
# {burst_id_str}_{date_str}
COMPASS_FILE_REGEX = r"(?P<burst_id>t\d{3}_\d+_iw\d)_(?P<start_datetime>\d{8}).h5"
