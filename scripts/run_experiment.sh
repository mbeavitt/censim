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

echo "=========================================="
echo "  EXPERIMENT: 600 SIMULATIONS"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Sequence file:    $SEQUENCE_FILE"
echo "Total runs:       $NUM_RUNS"
echo ""
echo "Starting parallel simulations..."
echo ""

# Generate run IDs and run in parallel
# Using --bar for cleaner progress display, --eta for time estimate
seq 0 $((NUM_RUNS - 1)) | parallel --bar --eta \
  "python3 scripts/experiment_sim.py \
    --sequence-file $SEQUENCE_FILE \
    --output-dir $OUTPUT_DIR \
    --run-id {} 2>&1 | grep -E '(Completed|Collapsed|Saved)' || true"

echo ""
echo "=========================================="
echo "  ALL SIMULATIONS COMPLETED"
echo "=========================================="
echo ""
echo "Analyzing results..."
echo ""

# Count successful runs (those with 6M generation file and no collapse marker)
TOTAL_RUNS=$(ls -1 "$OUTPUT_DIR"/run*_6000000gen.fa 2>/dev/null | wc -l)
COLLAPSED_RUNS=$(ls -1 "$OUTPUT_DIR"/run*_COLLAPSED.txt 2>/dev/null | wc -l)
SUCCESSFUL_RUNS=$((TOTAL_RUNS - COLLAPSED_RUNS))

echo "Results:"
echo "  Total runs reaching 6M gen: $TOTAL_RUNS"
echo "  Collapsed runs:              $COLLAPSED_RUNS"
echo "  Successful runs:             $SUCCESSFUL_RUNS"
echo ""
echo "Ready for analysis! Run:"
echo "  python3 scripts/analyze_experiment.py"
echo "=========================================="
