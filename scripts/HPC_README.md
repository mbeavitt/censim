# HPC Evolutionary Simulation Setup

This directory contains scripts for running 300 independent evolutionary simulations on CSD3 with automatic checkpointing and resumption.

## Cost Estimate

- **Total runtime**: ~20,000 CPU hours for 300 runs × 6M generations
- **Your available**: 199,285 CPU hours
- **Usage**: ~10% of allocation

## Files

- `hpc_runner.py` - Main simulation runner with checkpointing
- `submit_hpc.sh` - Slurm submission script
- `manage_runs.py` - Helper for job management and monitoring

## Quick Start

### 1. Initial Submission

```bash
cd /home/mbeavitt/Code/Python/censim/scripts

# Make scripts executable
chmod +x hpc_runner.py submit_hpc.sh manage_runs.py

# Submit all 300 runs
python3 manage_runs.py submit \
    --sequence-file /path/to/your/input.seq \
    --output-dir /home/mbeavitt/rds/hpc-work/censim_output
```

### 2. Check Status

```bash
# Quick summary
python3 manage_runs.py status --output-dir /home/mbeavitt/rds/hpc-work/censim_output

# Verbose (show all runs)
python3 manage_runs.py status --output-dir /home/mbeavitt/rds/hpc-work/censim_output --verbose
```

### 3. Resubmit Incomplete Runs

If some runs fail or don't complete:

```bash
# List incomplete runs
python3 manage_runs.py resubmit \
    --sequence-file /path/to/your/input.seq \
    --output-dir /home/mbeavitt/rds/hpc-work/censim_output \
    --list

# Resubmit them
python3 manage_runs.py resubmit \
    --sequence-file /path/to/your/input.seq \
    --output-dir /home/mbeavitt/rds/hpc-work/censim_output
```

## How It Works

### Checkpointing

- Each run saves its state every 1000 generations
- If a job is interrupted (time limit, node failure), it resumes from the last checkpoint
- Checkpoint files are stored in `run_XXX/checkpoints/state.pkl`

### Automatic Continuation

- Each job runs for up to 12 hours (chunk size: 60,000 generations ≈ 40 minutes)
- When a chunk completes, the job automatically resubmits itself for the next chunk
- This continues until 6M generations is reached or the array collapses

### Output Structure

```
output_dir/
├── run_000/
│   ├── fasta/
│   │   ├── 0001000generation.out.fa
│   │   ├── 0002000generation.out.fa
│   │   └── ...
│   ├── records/
│   │   ├── 0001000generation.record.txt
│   │   └── ...
│   ├── cenh3/
│   │   ├── 0001000generation.cenh3.txt
│   │   └── ...
│   ├── checkpoints/
│   │   └── state.pkl
│   ├── metadata.json
│   └── COMPLETE or COLLAPSED (when done)
├── run_001/
│   └── ...
└── run_299/
    └── ...
```

## Advanced Options

### Enable D2 Bias

```bash
python3 manage_runs.py submit \
    --sequence-file /path/to/input.seq \
    --output-dir /home/mbeavitt/rds/hpc-work/censim_output \
    --d2-bias \
    --d2-bias-strength 1.5
```

### Disable Correlation Dimension (faster)

```bash
python3 manage_runs.py submit \
    --sequence-file /path/to/input.seq \
    --output-dir /home/mbeavitt/rds/hpc-work/censim_output \
    --no-correlation-dimension
```

### Change Chunk Size

```bash
python3 manage_runs.py submit \
    --sequence-file /path/to/input.seq \
    --output-dir /home/mbeavitt/rds/hpc-work/censim_output \
    --chunk-size 120000  # 2 chunks per job instead of 1
```

## Manual Job Control

### Submit a specific run manually

```bash
sbatch --array=42 \
    --export=ALL,SEQUENCE_FILE=/path/to/input.seq,OUTPUT_DIR=/path/to/output \
    submit_hpc.sh
```

### Check job status

```bash
squeue -u mab282  # Show your running jobs
sacct -u mab282 --starttime=today  # Show today's job history
```

### Cancel jobs

```bash
scancel <job_id>  # Cancel specific job
scancel -u mab282  # Cancel all your jobs
```

## Troubleshooting

### Check logs

Logs are in `logs/run_<jobid>_<arrayid>.out` and `.err` files.

```bash
# Recent errors
tail logs/run_*_042.err

# Successful completion messages
grep COMPLETE logs/run_*.out
```

### Verify checkpoint

```python
import pickle
with open('output/run_000/checkpoints/state.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f"Generation: {data['generation']}")
    print(f"Timestamp: {data['timestamp']}")
```

### Resume from specific generation

```bash
python3 hpc_runner.py \
    --sequence-file input.seq \
    --output-dir output \
    --run-id 0 \
    --start-generation 3000000
```

## Monitoring Best Practices

1. **Initial check (after 1 hour)**: Make sure jobs are running
   ```bash
   squeue -u mab282
   python3 manage_runs.py status --output-dir <output>
   ```

2. **Daily check**: Monitor progress
   ```bash
   python3 manage_runs.py status --output-dir <output>
   ```

3. **After 3 days**: Check for stalled runs and resubmit if needed
   ```bash
   python3 manage_runs.py resubmit --sequence-file <seq> --output-dir <output>
   ```

## Expected Timeline

- Chunk duration: ~40 minutes
- Chunks per run: 100 (6M / 60k)
- Total wall-clock per run: ~67 hours
- With auto-resubmission: ~3-4 days for completion (depending on queue)

## Notes

- Jobs run on `icelake` partition (good availability, 1 CPU, <1GB RAM)
- Uses `HENDERSON-SL3-CPU` account (12-hour time limit)
- Each run is independent (embarrassingly parallel)
- Checkpoint files are ~MB in size (mainly sequence data)
- Total disk usage will be substantial (estimate ~TB for all outputs)
