# CenSim Singularity Container

This document explains how to build and use the CenSim Singularity container.

## Building the Container

From the project root directory:

```bash
# Build the container (requires sudo/root or --fakeroot)
sudo singularity build censim.sif censim.def

# Or with fakeroot (if available on your system):
singularity build --fakeroot censim.sif censim.def
```

Build time: ~5-10 minutes

## Using the Container

### Basic Usage

```bash
# Get help
singularity exec censim.sif censim-run --help

# Or use the runscript (same as censim-run)
singularity run censim.sif --help
./censim.sif --help  # if executable
```

### Running a Simulation

```bash
# Basic run
singularity exec censim.sif censim-run \
    --sequence-file /path/to/input.seq \
    --output-dir /path/to/output \
    --max-generations 1000000

# Fast run without correlation dimension (100,000x faster)
singularity exec censim.sif censim-run \
    --sequence-file /path/to/input.seq \
    --output-dir /path/to/output \
    --max-generations 1000000 \
    --no-correlation-dimension

# With D2 bias
singularity exec censim.sif censim-run \
    --sequence-file /path/to/input.seq \
    --output-dir /path/to/output \
    --d2-bias \
    --d2-bias-strength 1.5
```

### Mounting Directories

Singularity automatically binds your `$HOME` directory, but you may need to explicitly bind other locations:

```bash
# Bind additional directories
singularity exec \
    --bind /scratch:/scratch \
    --bind /data:/mnt/data \
    censim.sif censim-run \
    -s /mnt/data/input.seq \
    -o /scratch/output
```

### Quick Test

The container includes test data. Run a quick test (3-4 chunks, should complete in seconds):

```bash
# Create temporary output directory
mkdir -p /tmp/censim-test

# Run quick test - 3000 generations (3 chunks)
singularity exec censim.sif censim-run \
    --sequence-file /opt/censim/data/15000copy_cen178.seq \
    --output-dir /tmp/censim-test \
    --max-generations 3000 \
    --no-correlation-dimension

# Check results
ls -lh /tmp/censim-test/fasta/
```

## Using with HPC Systems

### CSD3 (Cambridge)

```bash
# Load singularity module
module load singularity

# Run in a job
singularity exec censim.sif censim-run \
    -s $INPUT_FILE \
    -o $OUTPUT_DIR \
    -g 6000000 \
    --no-correlation-dimension
```

### With the HPC Scripts

The `scripts/` directory contains HPC orchestration scripts that work with the containerized `censim-run`:

1. **Update `scripts/submit_hpc.sh`** to use the container:
   ```bash
   # Instead of:
   censim-hpc --sequence-file ...

   # Use:
   singularity exec /path/to/censim.sif censim-run \
       --sequence-file "${SEQUENCE_FILE}" \
       --output-dir "${OUTPUT_DIR}" \
       --max-generations "${MAX_GENERATIONS}" \
       ${NO_CD}
   ```

2. The HPC scripts (`scripts/hpc_runner.py`, `manage_runs.py`) can orchestrate multiple container-based runs

## Container Contents

- **CenSim package**: Installed in `/opt/censim/`
- **Command**: `censim-run` available in PATH
- **Test data**: `/opt/censim/data/test_100copy_cen178.seq`
- **Python**: 3.10 with numpy, scipy, numba

## Options Reference

```
censim-run --help

Options:
  -s, --sequence-file FILE      Input sequence file (.seq) [required]
  -o, --output-dir DIR          Output directory (default: ./output)
  -g, --max-generations N       Maximum generations (default: 6000000)
  --checkpoint-interval N       Save every N generations (default: 1000)
  --no-correlation-dimension    Disable D2 calculation (much faster)
  --d2-bias                     Bias insertions to high D2 regions
  --d2-bias-strength FLOAT      Strength of D2 bias (default: 1.0)
```

## Performance Notes

- **With correlation dimension**: ~1-2 hours per 1000 generations (for 15,000 copy array)
- **Without correlation dimension (`--no-correlation-dimension`)**: ~1-2 seconds per 1000 generations
- For evolutionary simulations, use `--no-correlation-dimension` unless you specifically need D2 analysis

## Troubleshooting

### Permission Issues
```bash
# Singularity binds $HOME by default, but may have issues with other paths
# Use explicit --bind
singularity exec --bind /path/to/data:/data censim.sif ...
```

### Python Import Errors
```bash
# Verify installation
singularity exec censim.sif python3 -c "import censim; print(censim.__version__)"
```

### Out of Memory
```bash
# Reduce checkpoint interval to save more frequently
singularity exec censim.sif censim-run ... --checkpoint-interval 500
```
