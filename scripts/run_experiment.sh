#!/bin/bash
#
# Run 600 simulations in parallel using GNU parallel
#

set -e

# Configuration
SEQUENCE_FILE="data/15000copy_cen178.seq"
OUTPUT_DIR="experiment_output"
NUM_RUNS=600

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting 600 simulations using GNU parallel..."
echo "Output directory: $OUTPUT_DIR"
echo "Sequence file: $SEQUENCE_FILE"
echo ""

# Generate run IDs and run in parallel
seq 0 $((NUM_RUNS - 1)) | parallel --progress --eta \
  "python3 scripts/experiment_sim.py \
    --sequence-file $SEQUENCE_FILE \
    --output-dir $OUTPUT_DIR \
    --run-id {}"

echo ""
echo "All simulations completed!"
echo "Checking results..."

# Count successful runs (those with 6M generation file and no collapse marker)
TOTAL_RUNS=$(ls -1 "$OUTPUT_DIR"/run*_6000000gen.fa 2>/dev/null | wc -l)
COLLAPSED_RUNS=$(ls -1 "$OUTPUT_DIR"/run*_COLLAPSED.txt 2>/dev/null | wc -l)
SUCCESSFUL_RUNS=$((TOTAL_RUNS - COLLAPSED_RUNS))

echo "Total runs with 6M generation: $TOTAL_RUNS"
echo "Collapsed runs: $COLLAPSED_RUNS"
echo "Successful runs: $SUCCESSFUL_RUNS"
