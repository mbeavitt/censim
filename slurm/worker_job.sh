#!/bin/bash
#SBATCH --job-name=censim_worker
#SBATCH --array=1-400
#SBATCH --time=00:05:00
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --account=HENDERSON-SL3-CPU
#SBATCH --output=logs/worker_%A_%a.out
#SBATCH --error=logs/worker_%A_%a.err

# SLURM worker script for centromere simulation on CSD3
# This runs 1000 generations and exits. Submit multiple times to continue.
#
# Usage:
#   sbatch slurm/worker_job.sh
#
# Each task in the array (1-400) represents an independent simulation run.

# ==============================================================================
# CONFIGURATION - Edit these variables before first submission
# ==============================================================================

# Path to censim project root
PROJECT_ROOT="/home/mab282/rds/hpc-work/censim"

# Input sequence file (same for all runs)
SEQUENCE_FILE="${PROJECT_ROOT}/data/input.seq"

# Output and checkpoint directories (on RDS, not home!)
OUTPUT_DIR="${PROJECT_ROOT}/output"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"

# Simulation parameters
MAX_GENERATIONS=6000000
NO_CORRELATION_DIM=""  # Set to "--no-correlation-dimension" to disable
D2_BIAS=""  # Set to "--d2-bias" to enable
D2_BIAS_STRENGTH=1.0

# Python environment (adjust if using conda/virtualenv)
# Leave blank to use system python
PYTHON_CMD="python3"

# ==============================================================================
# END CONFIGURATION
# ==============================================================================

# Load any required modules
# module load python/3.9  # Uncomment and adjust if needed

# Create log directory
mkdir -p logs

# Use SLURM_ARRAY_TASK_ID as the run ID
RUN_ID=${SLURM_ARRAY_TASK_ID}

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Run ID: ${RUN_ID}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Started: $(date)"
echo "=========================================="

# Check if this is a new run or resume
cd "${PROJECT_ROOT}"

LATEST_CHECKPOINT=$("${PYTHON_CMD}" -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}/src')
from censim.checkpoint import SimulationCheckpoint
mgr = SimulationCheckpoint('${RUN_ID}', '${CHECKPOINT_DIR}')
gen, path = mgr.get_latest_checkpoint()
print(gen)
" 2>/dev/null)

# Build command
CMD="${PROJECT_ROOT}/scripts/hpc_worker.py"
CMD="${CMD} --run-id ${RUN_ID}"
CMD="${CMD} --checkpoint-dir ${CHECKPOINT_DIR}"
CMD="${CMD} --output-dir ${OUTPUT_DIR}"
CMD="${CMD} --max-generations ${MAX_GENERATIONS}"

if [ -n "${NO_CORRELATION_DIM}" ]; then
    CMD="${CMD} ${NO_CORRELATION_DIM}"
fi

if [ -n "${D2_BIAS}" ]; then
    CMD="${CMD} ${D2_BIAS} --d2-bias-strength ${D2_BIAS_STRENGTH}"
fi

# Resume from checkpoint or start new
if [ "${LATEST_CHECKPOINT}" != "0" ]; then
    echo "Resuming from generation ${LATEST_CHECKPOINT}"
    CMD="${CMD} --resume"
else
    echo "Starting new simulation"
    CMD="${CMD} --sequence-file ${SEQUENCE_FILE}"
fi

# Run the worker
echo "Command: ${PYTHON_CMD} ${CMD}"
echo ""

"${PYTHON_CMD}" ${CMD}

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="

exit ${EXIT_CODE}
