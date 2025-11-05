#!/bin/bash
#SBATCH -J censim_checkpoint_test
#SBATCH -A HENDERSON-SL3-CPU    # Update as needed
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00         # 10 min should be plenty for 3 chunks
#SBATCH --mem=2G
#SBATCH -o logs/checkpoint_test_%j.out
#SBATCH -e logs/checkpoint_test_%j.err

# Test checkpointing with 3 chunks (150,000 generations total)
set -e

echo "=========================================="
echo "CenSim Checkpoint Test (3 chunks)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo "=========================================="

# Configuration
CONTAINER="${CONTAINER:-./censim.sif}"
SEQUENCE_FILE="${SEQUENCE_FILE:-./data/15000copy_cen178.seq}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/checkpoint_test}"
CHUNK_SIZE=50000
MAX_GENERATIONS=150000  # 3 chunks

echo "Configuration:"
echo "  Container: ${CONTAINER}"
echo "  Sequence: ${SEQUENCE_FILE}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Chunk size: ${CHUNK_SIZE} generations"
echo "  Max generations: ${MAX_GENERATIONS} (3 chunks)"
echo "=========================================="

# Verify files exist
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found: ${CONTAINER}"
    exit 1
fi

if [ ! -f "${SEQUENCE_FILE}" ]; then
    echo "ERROR: Sequence file not found: ${SEQUENCE_FILE}"
    exit 1
fi

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# Load singularity if needed
# module load singularity

# Run simulation with checkpointing
# The tool will save every 1000 generations, and we're running to 150k
echo "Running simulation (3 chunks of ${CHUNK_SIZE} generations each)..."
echo "This will checkpoint every 1000 generations..."
echo

time singularity exec \
    --bind $(pwd):$(pwd) \
    "${CONTAINER}" \
    censim-run \
        --sequence-file "${SEQUENCE_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-generations "${MAX_GENERATIONS}" \
        --checkpoint-interval 1000 \
        --no-correlation-dimension

EXIT_CODE=$?

echo
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="

# Show results
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Test completed successfully"
    echo
    echo "Generated checkpoints (should have 150):"
    CHECKPOINT_COUNT=$(ls -1 "${OUTPUT_DIR}"/fasta/*.fa 2>/dev/null | wc -l)
    echo "  FASTA files: ${CHECKPOINT_COUNT}"
    echo
    echo "Checkpoints per chunk:"
    echo "  Chunk 1 (0-50k):    $(ls -1 "${OUTPUT_DIR}"/fasta/00[0-4]*.fa 2>/dev/null | wc -l) files"
    echo "  Chunk 2 (50k-100k): $(ls -1 "${OUTPUT_DIR}"/fasta/00[5-9]*.fa 2>/dev/null | wc -l) files"
    echo "  Chunk 3 (100k-150k): $(ls -1 "${OUTPUT_DIR}"/fasta/01[0-4]*.fa 2>/dev/null | wc -l) files"
    echo
    echo "Sample outputs:"
    ls -lh "${OUTPUT_DIR}"/fasta/ | head -n 10
    echo "..."
    ls -lh "${OUTPUT_DIR}"/fasta/ | tail -n 5
else
    echo "✗ Test failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi
