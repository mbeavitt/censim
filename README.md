# Replicated Assemblies Centromere Study

Centromere simulation package (early development).

## Usage

### Setup
```bash
conda env create -f requirements.yaml
conda activate censim
```

### Running Simulations

Use the provided scripts to run simulations:

```bash
# Run a full simulation
python scripts/full_sim.py --sequence-file ./data/15000copy_cen178.seq \
                           --pos-file ./data/15000copy_cen178.178bp.bed.pos \
                           --output-dir ./output

# Run a test simulation (default 1000 generations)
python scripts/test_sim.py [generations]

# Profile simulation performance
bash scripts/test_sim.sh [generations]
bash scripts/profile_optimized.sh
```

### Using as a Library

Import and use the simulation functions directly:

```python
from censim.simulation import read_sequence, read_pos_file, introduce_mutations
import random
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load data
sequence = read_sequence("./data/15000copy_cen178.seq")
unit_data = read_pos_file("./data/15000copy_cen178.178bp.bed.pos")

# Run simulation
mutated_seq, records, adjusted_pos, cenh3, collapsed = introduce_mutations(
    sequence, 0, 1000, unit_data
)
```

### Testing
```bash
pytest
```