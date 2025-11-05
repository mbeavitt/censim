# CenSim - Centromere Evolution Simulator

A Python package for simulating centromere evolution with repeat dynamics, CENH3 occupancy, and correlation dimension analysis.

## Features

- Efficient repeat-based sequence representation
- CENH3 occupancy tracking
- Correlation dimension analysis
- HPC-ready with automatic checkpointing
- Supports 300-parallel evolutionary runs

## Installation

### From source

```bash
cd /home/mbeavitt/Code/Python/censim
pip install -e .
```

### With explicit dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Single simulation run

```bash
censim-run --sequence-file data/input.seq --output-dir output
```

### HPC parallel runs (300 individuals)

See [scripts/HPC_README.md](scripts/HPC_README.md) for detailed HPC instructions.

Quick start:
```bash
# Submit all 300 runs
censim-manage submit \
    --sequence-file data/input.seq \
    --output-dir ~/rds/hpc-work/censim_output

# Check status
censim-manage status --output-dir ~/rds/hpc-work/censim_output
```

## Package Structure

```
censim/
├── src/censim/
│   ├── __init__.py           # Package initialization
│   ├── simulation.py         # Core simulation logic
│   ├── identity.py           # Identity calculations
│   ├── correlation_dimension.py  # Correlation dimension analysis
│   └── scripts/
│       ├── full_sim.py       # Single run script
│       ├── hpc_runner.py     # HPC runner with checkpointing
│       └── manage_runs.py    # Job management helper
├── scripts/
│   ├── submit_hpc.sh         # Slurm submission script
│   └── HPC_README.md         # HPC documentation
├── pyproject.toml            # Modern Python packaging
└── requirements.txt          # Dependencies
```

## Command-Line Tools

After installation, three commands are available:

- `censim-run` - Run a single simulation
- `censim-hpc` - Run HPC simulation with checkpointing
- `censim-manage` - Manage HPC job submissions

## Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0
- numba >= 0.54.0

## Development

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

This includes:
- pytest (testing)
- black (code formatting)
- ruff (linting)
- ipython & jupyter (interactive development)

## Usage Examples

### Python API

```python
from censim import read_sequence, introduce_mutations

# Read initial sequence
sequence = read_sequence("data/input.seq")

# Run 1000 generations
mutated_seq, records, cenh3, collapsed, d_hist, d_latest = introduce_mutations(
    sequence,
    start_generation=0,
    num_generations=1000,
    compute_correlation_dim=True,
    use_d2_bias=False
)
```

### Command Line

```bash
# Basic run
censim-run -s input.seq -o output

# With D2 bias
censim-run -s input.seq -o output --d2-bias --d2-bias-strength 1.5

# Disable correlation dimension (faster)
censim-run -s input.seq -o output --no-correlation-dimension

# Custom generation count
censim-run -s input.seq -o output --max-generations 1000000
```

## HPC Usage

For running 300 parallel evolutionary simulations on CSD3:

1. Install the package: `pip install -e .`
2. See [scripts/HPC_README.md](scripts/HPC_README.md) for detailed instructions
3. Estimated cost: ~20,000 CPU hours (~10% of typical allocation)

## License

MIT

## Authors

M.A. Beavitt (mab282@cam.ac.uk)
