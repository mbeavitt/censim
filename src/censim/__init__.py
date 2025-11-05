"""
CenSim - Centromere Evolution Simulator

A package for simulating centromere evolution with repeat dynamics,
CENH3 occupancy, and correlation dimension analysis.
"""

__version__ = "0.1.0"

from .simulation import (
    RepeatSequence,
    read_sequence,
    introduce_mutations,
)

from .identity import (
    all_vs_all_identity_numba,
)

from .correlation_dimension import (
    hamming_distance_matrix,
    sliding_window_local_correlation,
    estimate_D2_from_C_r_batch,
)

__all__ = [
    # Simulation
    "RepeatSequence",
    "read_sequence",
    "introduce_mutations",
    # Identity
    "all_vs_all_identity_numba",
    # Correlation dimension
    "hamming_distance_matrix",
    "sliding_window_local_correlation",
    "estimate_D2_from_C_r_batch",
]
