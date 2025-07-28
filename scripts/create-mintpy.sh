set -eo pipefail

export BBOX="-119.81 35.4 -119.63 35.56"
export NUM_WORKERS=5
export FRAME_ID=36541
export OUTPUT_NAME="disp-output-${FRAME_ID}.nc"
export START="2017-01-01"
export END="2017-01-01"
export REFERENCE_METHOD="BORDER"

echo "Downloading NetCDF subset..."
opera-utils disp-s1-download \
    --output-dir subset-ncs \
    --bbox $BBOX \
    --frame-id $FRAME_ID \
    --start-datetime $START \
    --end-datetime $END \
    --num-workers $NUM_WORKERS

echo "Reformatting to ${OUTPUT_NAME}"
opera-utils disp-s1-reformat \
    --drop-vars connected_component_labels shp_counts persistent_scatterer_mask timeseries_inversion_residuals \
    --reference-method $REFERENCE_METHOD \
    --output-name $OUTPUT_NAME \
    --input-files subset-ncs/OPERA_L3_DISP-S1*.nc

echo "Converting to MintPy format..."
python -m opera_utils.disp.mintpy $OUTPUT_NAME \
    --sample-disp-nc $(ls subset-ncs/OPERA_L3_DISP-S1*.nc | head -1)
