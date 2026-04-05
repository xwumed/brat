#!/bin/bash
# BrAT feature extraction for UCSF-PDGM dataset
# GPU-only, does not conflict with DUNE (CPU)

cd /home/xuewei/MRI/brat

DATA_DIR="/home/xuewei/MRI/UCSF_PDGM_flat"
OUTPUT_DIR="/home/xuewei/MRI/UCSF_PDGM_flat_brat"
WEIGHTS="/home/xuewei/MRI/brat/brat_4m_densenet169.bin"
TOTAL=501

PYTHON="/home/xuewei/MRI/brat/.venv/bin/python"

echo "$(date): Starting BrAT UCSF-PDGM extraction ($TOTAL patients)"

$PYTHON run_inference.py \
    --weights "$WEIGHTS" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" 2>&1

exit_code=$?
done_count=$(find "$OUTPUT_DIR" -name "*.pt" 2>/dev/null | wc -l)

if [ "$exit_code" -eq 0 ]; then
    echo "$(date): BrAT finished normally. $done_count/$TOTAL completed."
else
    echo "$(date): BrAT exited with code $exit_code. $done_count/$TOTAL completed."
fi
