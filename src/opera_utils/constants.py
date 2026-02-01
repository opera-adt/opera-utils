from __future__ import annotations

import re

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

# Simplified NISAR GSLC filename regex (for backwards compatibility with parse_filename)
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

# Full NISAR GSLC filename regex (for the real-world product naming convention)
# Example: NISAR_L2_PR_GSLC_004_076_A_022_2005_QPDH_A_20251103T110514_20251103T110549_X05007_N_F_J_001.h5
# Format: NISAR_L2_<mode>_GSLC_<cycleNumber>_<relOrbitNumber>_<orbitDir>_<trackFrameNumber>_
#         <subSwathID>_<pols>_<lookDir>_<startDateTime>_<endDateTime>_<compositeReleaseID>_
#         <processingLevel>_<coverageIndicator>_<majorVersion>_<minorVersion>.h5
NISAR_GSLC_FILE_REGEX = re.compile(
    r"(?P<project>NISAR)_"
    r"(?P<level>L2)_"
    r"(?P<mode>[A-Z]{2})_"  # PR=Polar, etc.
    r"(?P<product_type>GSLC)_"
    r"(?P<cycle_number>\d{3})_"
    r"(?P<relative_orbit_number>\d{3})_"
    r"(?P<orbit_direction>[AD])_"  # A=Ascending, D=Descending
    r"(?P<track_frame_number>\d{3})_"
    r"(?P<subswath_id>\d{4})_"
    r"(?P<polarizations>[A-Z]{2,4})_"  # e.g. QPDH (quad-pol dual HV)
    r"(?P<look_direction>[AD])_"  # A=Ascending (right-looking), D=Descending (left-looking)
    r"(?P<start_datetime>\d{8}T\d{6})_"
    r"(?P<end_datetime>\d{8}T\d{6})_"
    r"(?P<composite_release_id>[A-Z]\d{5})_"
    r"(?P<processing_level>[A-Z])_"
    r"(?P<coverage_indicator>[A-Z])_"
    r"(?P<major_version>[A-Z])_"
    r"(?P<minor_version>\d{3})"
)

# NISAR GSLC HDF5 dataset paths
NISAR_GSLC_ROOT = "/science/LSAR/GSLC"
NISAR_GSLC_GRIDS = f"{NISAR_GSLC_ROOT}/grids"
NISAR_GSLC_IDENTIFICATION = "/science/LSAR/identification"
NISAR_GSLC_METADATA = "/science/LSAR/GSLC/metadata"

# Valid polarizations for NISAR GSLC
NISAR_POLARIZATIONS = ("HH", "VV", "HV", "VH", "RH", "RV", "LH", "LV")
# Valid frequencies for NISAR GSLC
NISAR_FREQUENCIES = ("A", "B")
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
