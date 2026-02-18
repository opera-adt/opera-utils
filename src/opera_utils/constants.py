from __future__ import annotations

import re
from enum import Enum


class UrlType(str, Enum):
    """Choices for the URL type for product access (S3 or HTTPS)."""

    S3 = "s3"
    HTTPS = "https"

    def __str__(self) -> str:
        return str(self.value)


SPEED_OF_LIGHT = 299_792_458

# Specific to OPERA CSLC products:
OPERA_DATASET_NAME = "/data/VV"
OPERA_IDENTIFICATION = "/identification"
NISAR_BOUNDING_POLYGON = "/science/LSAR/identification/boundingPolygon"

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
    r"(?P<generation_datetime>\d{8}T\d{6}Z)_"
    r"(?P<sensor>S1[ABCDE])_"
    r"(?P<polarization>VV|HH)_"
    r"v(?P<product_version>\d+\.\d+)"
)
COMPRESSED_CSLC_S1_FILE_REGEX = (
    r"(?P<project>OPERA)_"
    r"(?P<level>L2)_"
    r"(?P<is_compressed>COMPRESSED-)?"
    r"(?P<product_type>CSLC-S1)_"
    r"(?P<frame_id>F\d+)?_?"
    r"(?P<burst_id>T\d{3}-\d+-IW\d)_"
    r"(?P<start_datetime>\d{8}T\d{6}Z)_"
    r"(?P<ministack_start_datetime>\d{8}T\d{6}Z)_"
    r"(?P<ministack_stop_datetime>\d{8}T\d{6}Z)_"
    r"(?P<generation_datetime>\d{8}T\d{6}Z)_"
    r"(?P<polarization>VV|HH)_"
    r"v(?P<product_version>\d+\.\d+)"
)

# Old simulated NISAR format (kept for backwards compatibility)
NISAR_FILE_REGEX = (
    r"(?P<project>NISAR)_"
    r"(?P<level>L2)_"
    r"(?P<product_type>GSLC)_"
    r"(?P<sensor>NI)_"
    r"(?P<frame_id>F\d{3})_"
    r"(?P<start_datetime>\d{8}T\d{6}Z)_"
    r"(?P<generation_datetime>\d{8}T\d{6}Z)_"
    r"(?P<sensor_repeat>NI)_"
    r"(?P<polarization>VV|HH)_"
    r"v(?P<product_version>\d+\.\d+)"
)
# Official NISAR SDS naming convention:
# NISAR_L2_PR_GSLC_002_151_A_011_4005_DHDH_A_20251015T155040_ \
#                         20251015T155057_X05006_N_P_J_990.h5
# Format: NISAR_<level>_<proc_type>_<product>_<cycle>_<rel_orbit>_\
#        <direction>_<frame>_<scene>_<pol_mode>_<?>_<start>_<end>_<crid>_...
NISAR_SDS_FILE_REGEX = (
    r"(?P<project>NISAR)_"
    r"(?P<level>L\d)_"
    r"(?P<processing_type>[A-Z]{2})_"
    r"(?P<product_type>GSLC|GCOV|GUNW|RSLC|RIFG|RUNW|GOFF|ROFF)_"
    r"(?P<cycle>\d{3})_"
    r"(?P<relative_orbit>\d{3})_"
    r"(?P<direction>[AD])_"
    r"(?P<frame>\d{3})_"
    r"(?P<scene_id>\d{4})_"
    r"(?P<polarization_mode>[A-Z]{4})_"
    r"(?P<freq_pol>[A-Z])_"
    r"(?P<start_datetime>\d{8}T\d{6})_"
    r"(?P<end_datetime>\d{8}T\d{6})_"
    r"(?P<crid>[A-Z0-9]{6})_"
    r"(?P<field1>[A-Z])_"
    r"(?P<field2>[A-Z])_"
    r"(?P<field3>[A-Z])_"
    r"(?P<counter>\d{3})"
)
# https://github.com/opera-adt/COMPASS/blob/16a3c1da2a5db69b9e2007d798a1110d3a6c5f9f/src/compass/utils/runconfig.py#L316-L318
# {burst_id_str}_{date_str}
COMPASS_FILE_REGEX = r"(?P<burst_id>t\d{3}_\d+_iw\d)_(?P<start_datetime>\d{8}).h5"

# OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v1.0_20241219T231545Z.nc  # noqa: E501
DISP_FILE_REGEX = re.compile(
    r"OPERA_L3_DISP-"
    r"(?P<sensor>(S1|NI))_"
    r"(?P<acquisition_mode>IW)_"  # TODO: What's NISAR's?
    r"F(?P<frame_id>\d{5})_"
    r"(?P<polarization>(VV|HH))_"
    r"(?P<reference_datetime>\d{8}T\d{6}Z)_"
    r"(?P<secondary_datetime>\d{8}T\d{6}Z)_"
    r"v(?P<version>[\d.]+)_"
    r"(?P<generation_datetime>\d{8}T\d{6}Z)",
)
