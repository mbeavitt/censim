#!/bin/bash
# Automatic resubmission script for continuous progress
# This will keep resubmitting jobs until all 400 runs reach completion
#
# Usage:
#   nohup ./slurm/auto_resubmit.sh &
#   # Check progress: tail -f nohup.out

PROJECT_ROOT="/home/mab282/rds/hpc-work/censim"
MAX_GENERATIONS=6000000
SLEEP_TIME=600  # Wait 10 minutes between checks (jobs take ~50 sec + queue time)

echo "Starting auto-resubmit loop at $(date)"
echo "Will check every $SLEEP_TIME seconds until all runs complete"
echo "=============================================="

cd "${PROJECT_ROOT}"

iteration=1
while true; do
    echo ""
    echo "[Iteration $iteration] Checking progress at $(date)"

    # Check how many runs are incomplete
    INCOMPLETE=$(python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from censim.checkpoint import SimulationCheckpoint
from pathlib import Path

checkpoint_dir = Path('checkpoints')
max_gen = 6000000

incomplete = 0
for run_id in range(1, 401):
    mgr = SimulationCheckpoint(str(run_id), checkpoint_dir)
    gen, path = mgr.get_latest_checkpoint()
    if gen < max_gen:
        incomplete += 1

print(incomplete)
EOF
)

    COMPLETE=$((400 - INCOMPLETE))

    echo "Progress: $COMPLETE / 400 complete ($INCOMPLETE remaining)"

    if [ "$INCOMPLETE" -eq "0" ]; then
        echo "=============================================="
        echo "All 400 simulations complete!"
        echo "Finished at $(date)"
        exit 0
    fi

    # Check if there are already queued/running jobs
    RUNNING=$(squeue -u mab282 -h -t PENDING,RUNNING -n censim_worker | wc -l)

    if [ "$RUNNING" -gt "0" ]; then
        echo "Found $RUNNING jobs already queued/running, skipping submission"
    else
        echo "No jobs in queue, submitting new batch..."
        JOB_ID=$(sbatch slurm/worker_job.sh | awk '{print $4}')
        echo "Submitted job array: $JOB_ID"
    fi

    echo "Sleeping for $SLEEP_TIME seconds..."
    iteration=$((iteration + 1))
    sleep $SLEEP_TIME
done
