#!/bin/bash
# Quick test of censim container - runs in seconds
set -e

echo "=========================================="
echo "Quick CenSim Container Test"
echo "=========================================="

# Configuration
CONTAINER="${CONTAINER:-./censim.sif}"
SEQUENCE_FILE="./data/15000copy_cen178.seq"
TEST_OUTPUT=$(mktemp -d -t censim-test-XXXXXX)

echo "Container: ${CONTAINER}"
echo "Output directory: ${TEST_OUTPUT}"

# Verify container exists
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found: ${CONTAINER}"
    echo "Build it with: sudo singularity build censim.sif censim.def"
    exit 1
fi

# Run simulation with a few chunks, no D2
echo
echo "Running 3000 generations (3 chunks)..."
echo "(with --no-correlation-dimension for speed)"
echo

time singularity exec \
    --bind $(pwd):$(pwd) \
    "${CONTAINER}" \
    censim-run \
        --sequence-file "${SEQUENCE_FILE}" \
        --output-dir "${TEST_OUTPUT}" \
        --max-generations 3000 \
        --no-correlation-dimension

echo
echo "=========================================="
echo "Test Results:"
echo "=========================================="

# Check outputs
echo "Generated files:"
ls -lh "${TEST_OUTPUT}"/fasta/
ls -lh "${TEST_OUTPUT}"/records/
ls -lh "${TEST_OUTPUT}"/cenh3/

echo
echo "Final FASTA output:"
head -n 2 "${TEST_OUTPUT}"/fasta/0001000generation.out.fa

echo
echo "=========================================="
echo "Test PASSED!"
echo "Output saved to: ${TEST_OUTPUT}"
echo "=========================================="
