# Installation Guide for CenSim

## For Local Development

```bash
cd /home/mbeavitt/Code/Python/censim

# Install in editable mode
pip install -e .

# Verify installation
censim-run --help
censim-hpc --help
censim-manage --help
```

## For HPC (CSD3)

### Step 1: Load Python module

```bash
module purge
module load python/3.9  # or whatever version is available
```

### Step 2: Install package

```bash
cd /home/mbeavitt/Code/Python/censim

# Install in user directory
pip install --user -e .

# OR create a virtual environment (recommended)
python -m venv ~/.venv/censim
source ~/.venv/censim/bin/activate
pip install -e .
```

### Step 3: Verify installation

```bash
censim-hpc --help
censim-manage --help
```

### Step 4: Test locally before submitting

```bash
# Quick test run
censim-hpc \
    --sequence-file data/input.seq \
    --output-dir test_output \
    --run-id 0 \
    --chunk-size 1000 \
    --max-generations 5000 \
    --no-correlation-dimension
```

### Step 5: Submit HPC jobs

Once verified, submit your jobs:

```bash
censim-manage submit \
    --sequence-file /path/to/input.seq \
    --output-dir ~/rds/hpc-work/censim_output
```

## Updating the Package

If you modify the source code:

```bash
cd /home/mbeavitt/Code/Python/censim
pip install -e . --force-reinstall --no-deps
```

Or if installed in editable mode (`-e`), changes are automatically reflected (except for scripts in pyproject.toml).

## Troubleshooting

### "Command not found: censim-run"

The command-line tools are defined in `pyproject.toml`. Make sure you've installed the package:

```bash
pip install -e .
```

### Import errors

Make sure you're in the right directory or have installed the package:

```bash
# From anywhere:
python -c "import censim; print(censim.__file__)"
```

### Module not found on HPC

On HPC, you may need to add to your `~/.bashrc`:

```bash
# Add to ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH"
```

Then reload:
```bash
source ~/.bashrc
```

### NumPy/Numba issues on HPC

If you encounter NumPy or Numba compatibility issues on CSD3:

```bash
# Try loading specific versions
module load python/3.9
pip install --user numpy==1.23.0 scipy==1.9.0 numba==0.56.0
```

## Development Setup

For development work with testing and linting:

```bash
cd /home/mbeavitt/Code/Python/censim

# Install with dev dependencies
pip install -e .
pip install -r requirements-dev.txt

# Run tests (when available)
pytest

# Format code
black src/

# Lint code
ruff check src/
```

## Uninstallation

```bash
pip uninstall censim
```
