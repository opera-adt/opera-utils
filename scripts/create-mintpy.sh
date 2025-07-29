#!/usr/bin/env bash
set -eo pipefail

# Require these to be set by the workflow
: "${BBOX:?Environment variable BBOX is required (e.g. \"minlon minlat maxlon maxlat\")}"
: "${FRAME_ID:?Environment variable FRAME_ID is required}"
: "${START:?Environment variable START (YYYY-MM-DD) is required}"
: "${END:?Environment variable END (YYYY-MM-DD) is required}"

# Optional / defaulted inputs
NUM_WORKERS=${NUM_WORKERS:-4}
REFERENCE_METHOD=${REFERENCE_METHOD:-BORDER}
OUTPUT_NAME=${OUTPUT_NAME:-disp-output-${FRAME_ID}.nc}

echo "Inputs:"
echo "  BBOX=$BBOX"
echo "  FRAME_ID=$FRAME_ID"
echo "  START=$START"
echo "  END=$END"
echo "  NUM_WORKERS=$NUM_WORKERS"
echo "  REFERENCE_METHOD=$REFERENCE_METHOD"
echo "  OUTPUT_NAME=$OUTPUT_NAME"
echo

echo "Downloading NetCDF subset..."
opera-utils disp-s1-download \
    --output-dir subset-ncs \
    --bbox $BBOX \
    --frame-id "$FRAME_ID" \
    --start-datetime "$START" \
    --end-datetime "$END" \
    --num-workers "$NUM_WORKERS"

echo "Reformatting to ${OUTPUT_NAME}..."
opera-utils disp-s1-reformat \
    --drop-vars connected_component_labels shp_counts persistent_scatterer_mask timeseries_inversion_residuals \
    --reference-method "$REFERENCE_METHOD" \
    --output-name "$OUTPUT_NAME" \
    --input-files subset-ncs/OPERA_L3_DISP-S1*.nc

echo "Converting to MintPy format..."
python -m opera_utils.disp.mintpy "$OUTPUT_NAME" \
    --sample-disp-nc "$(ls subset-ncs/OPERA_L3_DISP-S1*.nc | head -1)" \
    --outdir mintpy
