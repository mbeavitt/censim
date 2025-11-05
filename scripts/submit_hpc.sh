#!/bin/bash
#SBATCH -J censim_evo          # Job name
#SBATCH -A HENDERSON-SL3-CPU   # Account
#SBATCH -p icelake             # Partition (icelake has good availability)
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # Number of tasks
#SBATCH --cpus-per-task=1      # CPUs per task (single-threaded)
#SBATCH --time=12:00:00        # 12 hour time limit
#SBATCH --mem=1G               # Memory per node (conservative, you said <1GB)
#SBATCH --array=0-299          # Job array for 300 individuals
#SBATCH -o logs/run_%A_%a.out  # Standard output (%A=job ID, %a=array index)
#SBATCH -e logs/run_%A_%a.err  # Standard error
#SBATCH --mail-type=FAIL       # Email on failure only
##SBATCH --mail-user=mab282@cam.ac.uk  # Uncomment and update if you want email

# Exit on error
set -e

# Print job information
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo "=========================================="

# Load Python module (adjust version as needed for CSD3)
module purge
module load python/3.9  # Adjust based on what's available

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SEQUENCE_FILE="${SEQUENCE_FILE:-${PROJECT_ROOT}/data/input.seq}"  # Set via environment or default
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output}"
CHUNK_SIZE="${CHUNK_SIZE:-60000}"
MAX_GENERATIONS="${MAX_GENERATIONS:-6000000}"

# Options for simulation
D2_BIAS="${D2_BIAS:-}"  # Set to "--d2-bias" to enable
D2_BIAS_STRENGTH="${D2_BIAS_STRENGTH:-1.0}"
NO_CD="${NO_CD:-}"  # Set to "--no-correlation-dimension" to disable

# Use array task ID as run ID
RUN_ID=${SLURM_ARRAY_TASK_ID}

echo "Configuration:"
echo "  Run ID: ${RUN_ID}"
echo "  Sequence file: ${SEQUENCE_FILE}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Chunk size: ${CHUNK_SIZE} generations"
echo "  Max generations: ${MAX_GENERATIONS}"
echo "=========================================="

# Verify files exist
if [ ! -f "${SEQUENCE_FILE}" ]; then
    echo "ERROR: Sequence file not found: ${SEQUENCE_FILE}"
    exit 1
fi

# Activate virtual environment if using one
# Uncomment if you installed in a venv:
# source ~/.venv/censim/bin/activate

# Create log directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs"

# Run the simulation using censim-hpc command (installed via pip install -e .)
censim-hpc \
    --sequence-file "${SEQUENCE_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-id "${RUN_ID}" \
    --chunk-size "${CHUNK_SIZE}" \
    --max-generations "${MAX_GENERATIONS}" \
    ${D2_BIAS} \
    ${NO_CD}

# Check exit status
EXIT_CODE=$?

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="

# Determine if we need to resubmit
RUN_DIR="${OUTPUT_DIR}/run_$(printf '%03d' ${RUN_ID})"

if [ -f "${RUN_DIR}/COMPLETE" ]; then
    echo "Run ${RUN_ID} is COMPLETE"
    exit 0
elif [ -f "${RUN_DIR}/COLLAPSED" ]; then
    echo "Run ${RUN_ID} COLLAPSED (stopping)"
    exit 0
elif [ ${EXIT_CODE} -eq 0 ]; then
    # Chunk completed successfully, need to continue
    CHECKPOINT="${RUN_DIR}/checkpoints/state.pkl"
    if [ -f "${CHECKPOINT}" ]; then
        echo "Chunk completed, need to continue..."
        # Resubmit this specific array task
        sbatch --array=${RUN_ID} "${BASH_SOURCE[0]}"
        echo "Resubmitted job for run ${RUN_ID}"
    fi
    exit 0
else
    echo "ERROR: Job failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi
