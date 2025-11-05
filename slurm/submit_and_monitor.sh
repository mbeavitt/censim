#!/bin/bash
# Helper script to submit jobs and monitor progress on CSD3
#
# Usage:
#   ./slurm/submit_and_monitor.sh submit     # Submit job array
#   ./slurm/submit_and_monitor.sh status     # Check status of all runs
#   ./slurm/submit_and_monitor.sh progress   # Show detailed progress
#   ./slurm/submit_and_monitor.sh resubmit   # Resubmit for incomplete runs

PROJECT_ROOT="/home/mab282/rds/hpc-work/censim"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
MAX_GENERATIONS=6000000

case "$1" in
    submit)
        echo "Submitting job array (400 runs)..."
        cd "${PROJECT_ROOT}"
        sbatch slurm/worker_job.sh
        ;;

    status)
        echo "Job Status:"
        echo "==========="
        squeue -u mab282 --format="%.18i %.12P %.20j %.8u %.8T %.10M %.6D %R"
        ;;

    progress)
        echo "Simulation Progress:"
        echo "===================="
        cd "${PROJECT_ROOT}"
        python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from censim.checkpoint import SimulationCheckpoint
from pathlib import Path

checkpoint_dir = Path('checkpoints')
max_gen = 6000000

complete = 0
incomplete = 0
not_started = 0

print(f"{'Run ID':<10} {'Generation':<15} {'Progress':<10} {'Status':<15}")
print("-" * 60)

for run_id in range(1, 401):
    mgr = SimulationCheckpoint(str(run_id), checkpoint_dir)
    gen, path = mgr.get_latest_checkpoint()

    if gen == 0:
        status = "Not started"
        not_started += 1
    elif gen >= max_gen:
        status = "Complete"
        complete += 1
    else:
        status = "In progress"
        incomplete += 1

    progress = f"{gen / max_gen * 100:.1f}%"

    # Only print first 20 and last 20 for brevity, plus any incomplete
    if run_id <= 20 or run_id > 380 or status != "Complete":
        print(f"{run_id:<10} {gen:<15,} {progress:<10} {status:<15}")
    elif run_id == 21:
        print("...")

print("-" * 60)
print(f"\nSummary:")
print(f"  Complete:     {complete:>3} / 400")
print(f"  In progress:  {incomplete:>3} / 400")
print(f"  Not started:  {not_started:>3} / 400")
EOF
        ;;

    resubmit)
        echo "Checking which runs need to continue..."
        cd "${PROJECT_ROOT}"

        INCOMPLETE=$(python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from censim.checkpoint import SimulationCheckpoint
from pathlib import Path

checkpoint_dir = Path('checkpoints')
max_gen = 6000000

incomplete = []
for run_id in range(1, 401):
    mgr = SimulationCheckpoint(str(run_id), checkpoint_dir)
    gen, path = mgr.get_latest_checkpoint()
    if gen < max_gen:
        incomplete.append(run_id)

print(len(incomplete))
EOF
)

        if [ "$INCOMPLETE" -eq "0" ]; then
            echo "All runs are complete!"
        else
            echo "Found $INCOMPLETE incomplete runs. Resubmitting job array..."
            sbatch slurm/worker_job.sh
        fi
        ;;

    *)
        echo "Usage: $0 {submit|status|progress|resubmit}"
        echo ""
        echo "Commands:"
        echo "  submit    - Submit the job array (400 parallel runs)"
        echo "  status    - Show SLURM job status"
        echo "  progress  - Show detailed progress of all 400 runs"
        echo "  resubmit  - Resubmit jobs to continue incomplete runs"
        echo ""
        echo "Workflow:"
        echo "  1. ./slurm/submit_and_monitor.sh submit"
        echo "  2. Wait for jobs to finish (2 hours)"
        echo "  3. ./slurm/submit_and_monitor.sh progress"
        echo "  4. ./slurm/submit_and_monitor.sh resubmit"
        echo "  5. Repeat steps 2-4 until all runs complete (~6000 iterations)"
        exit 1
        ;;
esac
