"""Python wrapper for C kmer variance analysis using ctypes."""

import ctypes
import numpy as np
from pathlib import Path

# Load the shared library
lib_path = Path(__file__).parent / "kmer_c" / "libkmer.so"
if not lib_path.exists():
    lib_path = Path(__file__).parent / "kmer_c" / "libkmer.dylib"  # macOS
if not lib_path.exists():
    raise FileNotFoundError(
        "Compiled library not found. Run: cd censim/src/censim/kmer_c && make shared"
    )

lib = ctypes.CDLL(str(lib_path))

# Define bit256_t structure matching C definition
class Bit256(ctypes.Structure):
    _fields_ = [("v", ctypes.c_uint64 * 4)]

# Define function signatures
lib.find_4mers_from_bytes.argtypes = [
    ctypes.POINTER(ctypes.c_char),  # sequence bytes
    ctypes.c_int,                    # sequence length
    ctypes.POINTER(Bit256)           # output bit vector
]
lib.find_4mers_from_bytes.restype = None

lib.sliding_window_diversity_consecutive.argtypes = [
    ctypes.POINTER(Bit256),  # repeat_array
    ctypes.c_int,            # start_idx
    ctypes.c_int             # window_size
]
lib.sliding_window_diversity_consecutive.restype = ctypes.c_double

lib.sliding_window_diversity_allpairs.argtypes = [
    ctypes.POINTER(Bit256),  # repeat_array
    ctypes.c_int,            # start_idx
    ctypes.c_int             # window_size
]
lib.sliding_window_diversity_allpairs.restype = ctypes.c_double


def compute_kmer_fingerprints(repeat_units):
    """Compute k-mer fingerprints for a list of repeat units.

    Args:
        repeat_units: List of bytearrays (each 178bp) from RepeatSequence.units

    Returns:
        numpy array of Bit256 structures (one per repeat unit)
    """
    num_units = len(repeat_units)
    repeat_array = (Bit256 * num_units)()

    # Initialize to zero
    for i in range(num_units):
        for j in range(4):
            repeat_array[i].v[j] = 0

    # Process each repeat unit
    for i, unit in enumerate(repeat_units):
        # Convert bytearray to ctypes char array
        unit_bytes = bytes(unit)
        unit_len = len(unit_bytes)
        lib.find_4mers_from_bytes(
            ctypes.c_char_p(unit_bytes),
            unit_len,
            ctypes.byref(repeat_array[i])
        )

    return repeat_array


def sliding_window_analysis(repeat_units, window_size=100, method="consecutive",
                           max_exact_size=1000, scale_factor=30):
    """Perform sliding window diversity analysis on repeat units with adaptive subsampling.

    Args:
        repeat_units: List of bytearrays (each 178bp) from RepeatSequence.units
        window_size: Size of sliding window (default: 100)
        method: "consecutive" or "allpairs" (default: "consecutive")
        max_exact_size: threshold for exact computation (default: 1000)
        scale_factor: scaling factor for subsampling (default: 30)

    Returns:
        positions: Array of window center positions (in original repeat unit space)
        diversity: Array of diversity values at each position
    """
    num_units = len(repeat_units)

    # Calculate adaptive subsampling (same as identity.py)
    if num_units <= max_exact_size:
        subsample_every = 1
    else:
        # Use sqrt-based subsampling for gentler downsampling
        import math
        subsample_every = max(1, int(math.sqrt(num_units / scale_factor)))

    # Subsample if needed
    if subsample_every == 1:
        # Exact computation
        sampled_units = repeat_units
        sampled_indices = np.arange(num_units)
    else:
        # Subsampled computation
        sampled_indices = np.arange(0, num_units, subsample_every)
        sampled_units = [repeat_units[i] for i in sampled_indices]

    # Compute k-mer fingerprints on sampled units
    repeat_array = compute_kmer_fingerprints(sampled_units)
    num_sampled = len(sampled_units)
    num_windows = num_sampled - window_size + 1

    if num_windows <= 0:
        return np.array([]), np.array([])

    positions_sampled = np.zeros(num_windows, dtype=np.int32)
    diversity = np.zeros(num_windows, dtype=np.float64)

    # Choose function based on method
    if method == "consecutive":
        func = lib.sliding_window_diversity_consecutive
    elif method == "allpairs":
        func = lib.sliding_window_diversity_allpairs
    else:
        raise ValueError(f"Unknown method: {method}. Use 'consecutive' or 'allpairs'")

    # Compute diversity for each window on sampled data
    for i in range(num_windows):
        center = i + window_size // 2
        positions_sampled[i] = center
        diversity[i] = func(repeat_array, i, window_size)

    # Map positions back to original repeat unit space
    positions = sampled_indices[positions_sampled]

    return positions, diversity


def analyze_repeat_sequence(repeat_seq, window_size=100, method="consecutive",
                           max_exact_size=1000, scale_factor=30):
    """Analyze a RepeatSequence object directly with adaptive subsampling.

    Args:
        repeat_seq: RepeatSequence object from censim.simulation
        window_size: Size of sliding window (default: 100)
        method: "consecutive" or "allpairs" (default: "consecutive")
        max_exact_size: threshold for exact computation (default: 1000)
        scale_factor: scaling factor for subsampling (default: 30)

    Returns:
        positions: Array of window center positions (in repeat units)
        diversity: Array of diversity values at each position
    """
    return sliding_window_analysis(repeat_seq.units, window_size, method,
                                   max_exact_size, scale_factor)
