# Testing CenSim Container

This directory contains test scripts for the containerized CenSim tool.

## Quick Tests (Local)

### Basic functionality test
Runs 3000 generations (3 checkpoints) in seconds:
```bash
./scripts/test_quick.sh
```

## HPC Tests (SLURM)

### Checkpoint test (3 chunks)
Tests the checkpointing system by running 150,000 generations in 3 chunks of 50,000:
```bash
# Submit to SLURM
sbatch scripts/test_checkpoint.sh

# Or run locally (if you have the resources)
bash scripts/test_checkpoint.sh
```

This test:
- Runs 150,000 generations (3 chunks of 50k each)
- Saves checkpoints every 1000 generations (150 total checkpoints)
- Uses `--no-correlation-dimension` so each chunk completes in seconds
- Verifies all checkpoints are created correctly

### Single containerized run
Run a basic containerized simulation on HPC:
```bash
sbatch scripts/hpc_run_containerized.sh
```

## Environment Variables

All scripts support these environment variables:

- `CONTAINER` - Path to censim.sif (default: `./censim.sif`)
- `SEQUENCE_FILE` - Input sequence file (default: `./data/15000copy_cen178.seq`)
- `OUTPUT_DIR` - Output directory (default: varies by script)

Example:
```bash
CONTAINER=/path/to/censim.sif \
OUTPUT_DIR=/scratch/my_test \
sbatch scripts/test_checkpoint.sh
```

## Expected Results

### test_quick.sh
- Runtime: ~5-10 seconds
- Outputs: 3 checkpoint files (1000, 2000, 3000 generations)
- Should complete without errors

### test_checkpoint.sh
- Runtime: ~30-60 seconds (depending on system)
- Outputs: 150 checkpoint files (every 1000 generations up to 150k)
- Should create files in three "chunks":
  - Chunk 1: 0-50,000 generations (50 files)
  - Chunk 2: 50,000-100,000 generations (50 files)
  - Chunk 3: 100,000-150,000 generations (50 files)

## Building the Container First

Before running tests, build the container:

```bash
# With sudo
sudo singularity build censim.sif censim.def

# Or with fakeroot
singularity build --fakeroot censim.sif censim.def
```

## Troubleshooting

### Container not found
```bash
# Make sure you're in the project root
ls censim.sif

# If missing, build it
sudo singularity build censim.sif censim.def
```

### Bind mount issues
Add explicit binds if needed:
```bash
singularity exec --bind /scratch:/scratch censim.sif censim-run ...
```

### SLURM account/partition
Update the `#SBATCH -A` and `#SBATCH -p` lines in the scripts to match your HPC system.
