#!/bin/bash
#SBATCH -J censim_test          # Job name
#SBATCH -A HENDERSON-SL3-CPU    # Account (update as needed)
#SBATCH -p icelake              # Partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00         # 30 min for testing
#SBATCH --mem=2G
#SBATCH -o logs/test_%j.out
#SBATCH -e logs/test_%j.err

# Test script for running censim in a container with checkpoint/resume
set -e

echo "=========================================="
echo "CenSim Containerized HPC Test"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo "=========================================="

# Configuration
CONTAINER="${CONTAINER:-./censim.sif}"
SEQUENCE_FILE="${SEQUENCE_FILE:-./data/15000copy_cen178.seq}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/test_run}"
MAX_GENERATIONS="${MAX_GENERATIONS:-5000}"

echo "Configuration:"
echo "  Container: ${CONTAINER}"
echo "  Sequence: ${SEQUENCE_FILE}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Max generations: ${MAX_GENERATIONS}"
echo "=========================================="

# Verify container exists
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found: ${CONTAINER}"
    exit 1
fi

# Verify sequence file exists
if [ ! -f "${SEQUENCE_FILE}" ]; then
    echo "ERROR: Sequence file not found: ${SEQUENCE_FILE}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# Load singularity (if on a system that requires module load)
# module load singularity

# Run the simulation
echo "Starting simulation..."
singularity exec \
    --bind $(pwd):$(pwd) \
    "${CONTAINER}" \
    censim-run \
        --sequence-file "${SEQUENCE_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-generations "${MAX_GENERATIONS}" \
        --no-correlation-dimension

EXIT_CODE=$?

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="

# Show results
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Simulation completed successfully"
    echo
    echo "Generated outputs:"
    ls -lh "${OUTPUT_DIR}"/fasta/
    echo
    echo "Final sequence:"
    head -n 2 "${OUTPUT_DIR}"/fasta/*generation.out.fa | tail -n 1 | cut -c1-80
    echo "..."
else
    echo "✗ Simulation failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi
