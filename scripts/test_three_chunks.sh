#!/bin/bash
# Test script that simulates the 3-chunk workflow
# This runs 3 separate censim-run invocations to simulate separate SLURM jobs
# Each "chunk" resumes from the previous one
set -e

echo "=========================================="
echo "CenSim 3-Chunk Resume Test"
echo "=========================================="
echo "This simulates 3 separate SLURM jobs with checkpointing"
echo

# Configuration
CONTAINER="${CONTAINER:-./censim.sif}"
SEQUENCE_FILE="./data/15000copy_cen178.seq"
OUTPUT_DIR="./output/three_chunk_test_$(date +%Y%m%d_%H%M%S)"
CHUNK_SIZE=50000

echo "Configuration:"
echo "  Container: ${CONTAINER}"
echo "  Initial sequence: ${SEQUENCE_FILE}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Chunk size: ${CHUNK_SIZE} generations"
echo "=========================================="

# Verify container exists
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found: ${CONTAINER}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ============================================
# CHUNK 1: 0 -> 50,000 generations
# ============================================
echo
echo "=========================================="
echo "CHUNK 1: Running 0 -> 50,000 generations"
echo "=========================================="
CURRENT_SEQUENCE="${SEQUENCE_FILE}"

time singularity exec \
    --bind $(pwd):$(pwd) \
    "${CONTAINER}" \
    censim-run \
        --sequence-file "${CURRENT_SEQUENCE}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-generations ${CHUNK_SIZE} \
        --checkpoint-interval 1000 \
        --no-correlation-dimension

# Check chunk 1 completed
CHUNK1_OUTPUT="${OUTPUT_DIR}/fasta/0050000generation.out.fa"
if [ ! -f "${CHUNK1_OUTPUT}" ]; then
    echo "ERROR: Chunk 1 failed - output not found: ${CHUNK1_OUTPUT}"
    exit 1
fi
echo "✓ Chunk 1 complete: ${CHUNK1_OUTPUT}"

# ============================================
# CHUNK 2: 50,000 -> 100,000 generations
# ============================================
echo
echo "=========================================="
echo "CHUNK 2: Running 50,000 -> 100,000 generations"
echo "=========================================="
echo "Resuming from: ${CHUNK1_OUTPUT}"

# For the next chunk, we start from the last checkpoint
# But censim-run always starts from generation 0, so we need to:
# 1. Use the final FASTA from chunk 1 as input
# 2. Save to a new directory
# 3. Rename outputs to match generation numbers

OUTPUT_DIR_CHUNK2="${OUTPUT_DIR}/chunk2_temp"
time singularity exec \
    --bind $(pwd):$(pwd) \
    "${CONTAINER}" \
    censim-run \
        --sequence-file "${CHUNK1_OUTPUT}" \
        --output-dir "${OUTPUT_DIR_CHUNK2}" \
        --max-generations ${CHUNK_SIZE} \
        --checkpoint-interval 1000 \
        --no-correlation-dimension

# Move and rename outputs to continue generation numbering
for gen in $(seq 1000 1000 ${CHUNK_SIZE}); do
    OLD_GEN=$(printf "%07d" ${gen})
    NEW_GEN=$(printf "%07d" $((gen + CHUNK_SIZE)))

    if [ -f "${OUTPUT_DIR_CHUNK2}/fasta/${OLD_GEN}generation.out.fa" ]; then
        mv "${OUTPUT_DIR_CHUNK2}/fasta/${OLD_GEN}generation.out.fa" \
           "${OUTPUT_DIR}/fasta/${NEW_GEN}generation.out.fa"
        mv "${OUTPUT_DIR_CHUNK2}/records/${OLD_GEN}generation.record.txt" \
           "${OUTPUT_DIR}/records/${NEW_GEN}generation.record.txt"
        mv "${OUTPUT_DIR_CHUNK2}/cenh3/${OLD_GEN}generation.cenh3.txt" \
           "${OUTPUT_DIR}/cenh3/${NEW_GEN}generation.cenh3.txt"
    fi
done
rm -rf "${OUTPUT_DIR_CHUNK2}"

CHUNK2_OUTPUT="${OUTPUT_DIR}/fasta/0100000generation.out.fa"
if [ ! -f "${CHUNK2_OUTPUT}" ]; then
    echo "ERROR: Chunk 2 failed - output not found: ${CHUNK2_OUTPUT}"
    exit 1
fi
echo "✓ Chunk 2 complete: ${CHUNK2_OUTPUT}"

# ============================================
# CHUNK 3: 100,000 -> 150,000 generations
# ============================================
echo
echo "=========================================="
echo "CHUNK 3: Running 100,000 -> 150,000 generations"
echo "=========================================="
echo "Resuming from: ${CHUNK2_OUTPUT}"

OUTPUT_DIR_CHUNK3="${OUTPUT_DIR}/chunk3_temp"
time singularity exec \
    --bind $(pwd):$(pwd) \
    "${CONTAINER}" \
    censim-run \
        --sequence-file "${CHUNK2_OUTPUT}" \
        --output-dir "${OUTPUT_DIR_CHUNK3}" \
        --max-generations ${CHUNK_SIZE} \
        --checkpoint-interval 1000 \
        --no-correlation-dimension

# Move and rename outputs
for gen in $(seq 1000 1000 ${CHUNK_SIZE}); do
    OLD_GEN=$(printf "%07d" ${gen})
    NEW_GEN=$(printf "%07d" $((gen + CHUNK_SIZE * 2)))

    if [ -f "${OUTPUT_DIR_CHUNK3}/fasta/${OLD_GEN}generation.out.fa" ]; then
        mv "${OUTPUT_DIR_CHUNK3}/fasta/${OLD_GEN}generation.out.fa" \
           "${OUTPUT_DIR}/fasta/${NEW_GEN}generation.out.fa"
        mv "${OUTPUT_DIR_CHUNK3}/records/${OLD_GEN}generation.record.txt" \
           "${OUTPUT_DIR}/records/${NEW_GEN}generation.record.txt"
        mv "${OUTPUT_DIR_CHUNK3}/cenh3/${OLD_GEN}generation.cenh3.txt" \
           "${OUTPUT_DIR}/cenh3/${NEW_GEN}generation.cenh3.txt"
    fi
done
rm -rf "${OUTPUT_DIR_CHUNK3}"

CHUNK3_OUTPUT="${OUTPUT_DIR}/fasta/0150000generation.out.fa"
if [ ! -f "${CHUNK3_OUTPUT}" ]; then
    echo "ERROR: Chunk 3 failed - output not found: ${CHUNK3_OUTPUT}"
    exit 1
fi
echo "✓ Chunk 3 complete: ${CHUNK3_OUTPUT}"

# ============================================
# SUMMARY
# ============================================
echo
echo "=========================================="
echo "✓ All 3 chunks completed successfully!"
echo "=========================================="
echo "Generated checkpoints:"
CHECKPOINT_COUNT=$(ls -1 "${OUTPUT_DIR}"/fasta/*.fa 2>/dev/null | wc -l)
echo "  Total FASTA files: ${CHECKPOINT_COUNT}"
echo
echo "Checkpoints per chunk:"
echo "  Chunk 1 (0-50k):     $(ls -1 "${OUTPUT_DIR}"/fasta/00[0-4]*.fa 2>/dev/null | wc -l) files"
echo "  Chunk 2 (50k-100k):  $(ls -1 "${OUTPUT_DIR}"/fasta/00[5-9]*.fa 2>/dev/null | wc -l) files"
echo "  Chunk 3 (100k-150k): $(ls -1 "${OUTPUT_DIR}"/fasta/01[0-4]*.fa 2>/dev/null | wc -l) files"
echo
echo "Final sequence size:"
wc -c "${CHUNK3_OUTPUT}"
echo
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
