# HPC Simulation on CSD3

This directory contains scripts for running large-scale centromere simulations on Cambridge CSD3 using SLURM job arrays.

## Overview

The HPC workflow uses **checkpoint-based execution**:
- Each worker runs 1000 generations and saves a checkpoint
- Workers automatically resume from the last checkpoint
- Submit the job array multiple times until all runs reach 6,000,000 generations
- Run 400 independent simulations in parallel

## Quick Start

### 1. Setup on CSD3

First, copy your project to RDS storage (not home directory):

```bash
# Login to CSD3
ssh mab282@login.hpc.cam.ac.uk

# Copy project to RDS (adjust paths as needed)
cd ~/rds/hpc-work
git clone <your-censim-repo> censim
cd censim

# Install dependencies (if not already done)
pip install --user numpy numba

# Create necessary directories
mkdir -p data checkpoints output/fasta output/records logs
```

### 2. Prepare Input Data

Copy your input sequence file to the data directory:

```bash
cp /path/to/your/input.seq ~/rds/hpc-work/censim/data/input.seq
```

### 3. Configure Worker Script

Edit `slurm/worker_job.sh` and update these variables:

```bash
PROJECT_ROOT="/home/mab282/rds/hpc-work/censim"
SEQUENCE_FILE="${PROJECT_ROOT}/data/input.seq"
MAX_GENERATIONS=6000000
```

Optional: Enable D2 bias or disable correlation dimension calculations:

```bash
D2_BIAS="--d2-bias"              # Enable D2 bias
D2_BIAS_STRENGTH=1.5             # Increase bias strength
NO_CORRELATION_DIM="--no-correlation-dimension"  # Disable for speed
```

### 4. Submit Jobs

```bash
cd ~/rds/hpc-work/censim

# Submit the job array (400 parallel runs)
./slurm/submit_and_monitor.sh submit
```

### 5. Monitor Progress

```bash
# Check SLURM queue status
./slurm/submit_and_monitor.sh status

# Check detailed simulation progress
./slurm/submit_and_monitor.sh progress
```

### 6. Continue Simulations

After jobs complete (each runs 1000 generations = ~2 hours), resubmit:

```bash
./slurm/submit_and_monitor.sh resubmit
```

Repeat until all simulations reach 6,000,000 generations.

## Workflow Details

### Job Array Structure

- **Array size**: 1-400 (400 independent simulations)
- **Time limit**: 5 minutes per submission (50 sec runtime + buffer)
- **Partition**: icelake (can change to cclake or sapphire)
- **Resources**: 1 CPU, 2GB RAM per task
- **Account**: HENDERSON-SL3-CPU

Each array task gets a unique `SLURM_ARRAY_TASK_ID` (1-400) which becomes the `run_id`.

The short time limit (5 minutes) ensures:
- Fast queue throughput
- Minimal wasted resources if jobs fail
- Can resubmit frequently without concern

### Checkpoint System

Checkpoints are saved as compressed JSON files:

```
checkpoints/run_1_gen_1000.ckpt.gz
checkpoints/run_1_gen_2000.ckpt.gz
...
checkpoints/run_400_gen_6000000.ckpt.gz
```

Each checkpoint contains:
- Current generation number
- Full DNA sequence
- Complete mutation history
- Metadata (collapse status, sequence length, etc.)

**Note**: CENH3 occupancy is NOT saved (as requested).

### Output Files

Output files are written every 1000 generations:

```
output/fasta/run_1_gen_1000.fa
output/fasta/run_1_gen_2000.fa
...
output/records/run_1_gen_1000.txt
output/records/run_1_gen_2000.txt
...
```

### Iteration Estimate

- Total generations: 6,000,000
- Generations per job: 1,000
- Time per 1000 generations: ~50 seconds (~0.014 CPU hours)
- Jobs needed per run: 6,000
- Total compute time per run: ~84 CPU hours
- Total for 400 runs: ~33,600 CPU hours

**Cost estimate**: With your current balance of 199,861 CPU hours, you can easily run all 400 complete simulations (~33,600 hours) with ~166,000 hours remaining.

### Submission Strategy

Since each 1000-generation job takes only ~50 seconds, you can run many iterations quickly:
- ~72 jobs per hour per run (if submitted continuously)
- ~83 hours to complete one full simulation if run sequentially
- With 400 parallel runs, expect to submit the job array ~6,000 times total

**Recommended approach**: Set up a simple loop to auto-resubmit every 2 hours, or manually resubmit a few times per day until complete.

## Advanced Usage

### Running a Subset of Simulations

To run fewer than 400 simulations, edit `worker_job.sh`:

```bash
#SBATCH --array=1-20    # Run only 20 simulations instead of 400
```

### Customizing Parameters

Different simulations can use different parameters by editing `hpc_worker.py` to read run-specific config files.

### Checking Individual Run Status

```bash
# Check checkpoint for run 42
python3 << EOF
import sys
sys.path.insert(0, 'src')
from censim.checkpoint import SimulationCheckpoint

mgr = SimulationCheckpoint('42', 'checkpoints')
gen, path = mgr.get_latest_checkpoint()
print(f"Run 42 is at generation {gen:,}")
if path:
    ckpt = mgr.load_checkpoint(path)
    print(f"Sequence length: {len(ckpt['sequence']):,} bp")
    print(f"Total mutations: {len(ckpt['mutation_records']):,}")
EOF
```

### Cleaning Up Old Checkpoints

To save space, remove intermediate checkpoints (keep only latest):

```bash
cd ~/rds/hpc-work/censim

# For each run, keep only the most recent checkpoint
python3 << EOF
import sys
sys.path.insert(0, 'src')
from censim.checkpoint import SimulationCheckpoint
from pathlib import Path

for run_id in range(1, 401):
    mgr = SimulationCheckpoint(str(run_id), 'checkpoints')
    mgr.cleanup_old_checkpoints(keep_last_n=1)
    print(f"Cleaned run {run_id}")
EOF
```

## Troubleshooting

### Jobs Fail with "No module named 'censim'"

Make sure you're using the correct Python path and the project structure is intact:

```bash
cd ~/rds/hpc-work/censim
ls -la src/censim/
```

### Jobs Fail with "No such file or directory"

Check that `PROJECT_ROOT` in `worker_job.sh` matches your actual path:

```bash
echo $HOME/rds/hpc-work/censim
```

### Array Collapsed Early

Check the checkpoint metadata for collapsed runs:

```bash
python3 -c "
import sys, json, gzip
sys.path.insert(0, 'src')
from censim.checkpoint import SimulationCheckpoint

mgr = SimulationCheckpoint('1', 'checkpoints')
ckpt = mgr.load_checkpoint()
print('Collapsed:', ckpt['metadata']['collapsed'])
print('Final generation:', ckpt['generation'])
"
```

### Out of Memory Errors

Increase memory allocation in `worker_job.sh`:

```bash
#SBATCH --mem=8000    # 8GB instead of 4GB
```

### Check Account Balance

```bash
mybalance
```

## File Structure

```
censim/
├── slurm/
│   ├── README.md                    # This file
│   ├── worker_job.sh                # Main SLURM job script
│   └── submit_and_monitor.sh        # Helper script
├── scripts/
│   ├── hpc_worker.py                # Worker that runs 1000 generations
│   └── full_sim.py                  # Original single-run script
├── src/
│   └── censim/
│       ├── checkpoint.py            # Checkpoint management
│       ├── simulation.py            # Core simulation logic
│       ├── correlation_dimension.py
│       └── identity.py
├── data/
│   └── input.seq                    # Input sequence file
├── checkpoints/                     # Checkpoint files
├── output/
│   ├── fasta/                       # Output sequences
│   └── records/                     # Mutation records
└── logs/                            # SLURM output logs
```

## Contact

For CSD3-specific issues, contact: support@hpc.cam.ac.uk

For simulation issues, check logs in `logs/worker_JOBID_TASKID.err`
