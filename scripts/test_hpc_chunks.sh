#!/bin/bash
#SBATCH -J censim_chunk_test
#SBATCH -A HENDERSON-SL3-CPU    # Update as needed
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00         # 30 min for 3 chunks
#SBATCH --mem=2G
#SBATCH -o logs/chunk_test_%j.out
#SBATCH -e logs/chunk_test_%j.err

# Test HPC checkpointing workflow with 3 chunks
# This uses hpc_runner.py which handles checkpoint/resume
# Each time this job runs, it does ONE chunk, then needs to be resubmitted
set -e

echo "=========================================="
echo "CenSim HPC Chunk Test"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo "=========================================="

# Configuration
SEQUENCE_FILE="${SEQUENCE_FILE:-./data/15000copy_cen178.seq}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/chunk_test}"
RUN_ID="${RUN_ID:-0}"
CHUNK_SIZE=50000
MAX_GENERATIONS=150000  # 3 chunks total

echo "Configuration:"
echo "  Sequence: ${SEQUENCE_FILE}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Run ID: ${RUN_ID}"
echo "  Chunk size: ${CHUNK_SIZE} generations"
echo "  Max generations: ${MAX_GENERATIONS} (3 chunks)"
echo "=========================================="

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# Load Python (adjust for your system)
# module load python/3.9

# Run the HPC runner with checkpointing
# This will:
# - Check for existing checkpoint
# - Resume from checkpoint if exists
# - Run one chunk (50k generations)
# - Save checkpoint
# - Exit (ready to be resubmitted for next chunk)

echo "Running HPC runner..."
python3 scripts/hpc_runner.py \
    --sequence-file "${SEQUENCE_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-id "${RUN_ID}" \
    --chunk-size "${CHUNK_SIZE}" \
    --max-generations "${MAX_GENERATIONS}" \
    --no-correlation-dimension

EXIT_CODE=$?

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="

# Check status and determine if we need to continue
RUN_DIR="${OUTPUT_DIR}/run_$(printf '%03d' ${RUN_ID})"

if [ -f "${RUN_DIR}/COMPLETE" ]; then
    echo "✓ Run ${RUN_ID} is COMPLETE (all 3 chunks done)"
    echo
    echo "Final outputs:"
    ls -lh "${RUN_DIR}"/fasta/ | tail -n 5
    exit 0
elif [ -f "${RUN_DIR}/COLLAPSED" ]; then
    echo "⚠ Run ${RUN_ID} COLLAPSED"
    exit 0
elif [ ${EXIT_CODE} -eq 0 ]; then
    # Chunk completed successfully
    CHECKPOINT="${RUN_DIR}/checkpoints/state.pkl"
    if [ -f "${CHECKPOINT}" ]; then
        # Check current generation to see if we're done
        CURRENT_GEN=$(python3 -c "import pickle; print(pickle.load(open('${CHECKPOINT}', 'rb'))['generation'])")
        echo "Checkpoint at generation: ${CURRENT_GEN}"

        if [ ${CURRENT_GEN} -ge ${MAX_GENERATIONS} ]; then
            echo "✓ All chunks complete!"
        else
            echo "→ Need to continue - please resubmit this job"
            echo
            echo "Resubmit with:"
            echo "  sbatch scripts/test_hpc_chunks.sh"
        fi
    fi
    exit 0
else
    echo "✗ Job failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi
