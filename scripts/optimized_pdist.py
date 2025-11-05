#!/usr/bin/env python3
"""
Optimized pdist implementations for sequence comparison.

This module provides faster alternatives to scipy.pdist for DNA sequence comparison.
"""

import numpy as np
from numba import njit, prange
import time


@njit(parallel=True, fastmath=True)
def hamming_distance_matrix_numba(seq_array):
    """
    Compute pairwise Hamming distances using numba with parallelization.

    Args:
        seq_array: (N, L) uint8 array where N is number of sequences, L is length

    Returns:
        distance_matrix: (N, N) float32 array with pairwise Hamming distances (0-1 range)
    """
    N, L = seq_array.shape
    distances = np.zeros((N, N), dtype=np.float32)

    # Parallel loop over upper triangle
    for i in prange(N):
        for j in range(i + 1, N):
            # Count mismatches
            mismatches = 0
            for k in range(L):
                if seq_array[i, k] != seq_array[j, k]:
                    mismatches += 1

            # Normalize by length
            dist = mismatches / L
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


@njit(parallel=True, fastmath=True)
def hamming_identity_matrix_numba(seq_array):
    """
    Compute pairwise Hamming identity directly (faster than distance then subtract).

    Args:
        seq_array: (N, L) uint8 array where N is number of sequences, L is length

    Returns:
        identity_matrix: (N, N) float32 array with pairwise identities (0-1 range)
    """
    N, L = seq_array.shape
    identity = np.zeros((N, N), dtype=np.float32)

    # Set diagonal to 1.0
    for i in range(N):
        identity[i, i] = 1.0

    # Parallel loop over upper triangle
    for i in prange(N):
        for j in range(i + 1, N):
            # Count matches
            matches = 0
            for k in range(L):
                if seq_array[i, k] == seq_array[j, k]:
                    matches += 1

            # Normalize by length
            ident = matches / L
            identity[i, j] = ident
            identity[j, i] = ident

    return identity


def convert_sequences_fast(repeats):
    """
    Fast conversion of string sequences to uint8 numpy array.

    Args:
        repeats: list of sequence strings (all same length)

    Returns:
        seq_array: (N, L) uint8 array
    """
    if len(repeats) == 0:
        return np.array([], dtype=np.uint8)

    # Get dimensions
    N = len(repeats)
    L = len(repeats[0])

    # Pre-allocate array
    seq_array = np.empty((N, L), dtype=np.uint8)

    # Convert strings to bytes efficiently
    for i, seq in enumerate(repeats):
        # Use numpy's frombuffer for fast conversion
        seq_array[i] = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)

    return seq_array


def all_vs_all_identity_numba(repeats, max_exact_size=1000, scale_factor=30):
    """
    Compute all vs all identity matrix using numba (faster than scipy).

    Args:
        repeats: list of sequences (strings)
        max_exact_size: array size threshold for exact computation
        scale_factor: controls subsampling rate for large arrays

    Returns:
        tuple of (identity_matrix, subsample_every, subset_repeats)
    """
    import math

    n_repeats = len(repeats)

    if n_repeats <= max_exact_size:
        subsample_every = 1
    else:
        subsample_every = max(1, int(math.sqrt(n_repeats / scale_factor)))

    if subsample_every == 1:
        seq_array = convert_sequences_fast(repeats)
        identity_matrix = hamming_identity_matrix_numba(seq_array)
        subset_repeats = repeats
    else:
        subset_indices = np.arange(0, n_repeats, subsample_every)
        subset_repeats = [repeats[i] for i in subset_indices]

        seq_array = convert_sequences_fast(subset_repeats)
        identity_matrix = hamming_identity_matrix_numba(seq_array)

    return identity_matrix.astype(np.float32), subsample_every, subset_repeats


def benchmark_pdist_methods(repeats, n_runs=5):
    """
    Benchmark different pdist implementations.

    Args:
        repeats: list of sequence strings
        n_runs: number of runs to average

    Returns:
        results: dict of timing results
    """
    from scipy.spatial.distance import pdist, squareform

    print(f"Benchmarking pdist methods on {len(repeats)} sequences...")

    results = {}

    # Method 1: scipy pdist (current implementation)
    print("\n1. Testing scipy pdist...")
    times = []
    for i in range(n_runs):
        t0 = time.time()
        seq_array = np.array([[ord(c) for c in seq] for seq in repeats], dtype=np.uint8)
        distances = pdist(seq_array, metric='hamming')
        identity_matrix = 1 - squareform(distances)
        t1 = time.time()
        times.append(t1 - t0)
        if i == 0:
            scipy_result = identity_matrix

    results['scipy'] = {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'times': times
    }
    print(f"   Avg time: {np.mean(times):.4f}s ± {np.std(times):.4f}s")

    # Method 2: numba (optimized)
    print("\n2. Testing numba (with warmup)...")

    # Warmup run
    seq_array = convert_sequences_fast(repeats)
    _ = hamming_identity_matrix_numba(seq_array)

    # Actual benchmark
    times = []
    for i in range(n_runs):
        t0 = time.time()
        seq_array = convert_sequences_fast(repeats)
        identity_matrix = hamming_identity_matrix_numba(seq_array)
        t1 = time.time()
        times.append(t1 - t0)
        if i == 0:
            numba_result = identity_matrix

    results['numba'] = {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'times': times
    }
    print(f"   Avg time: {np.mean(times):.4f}s ± {np.std(times):.4f}s")

    # Check correctness
    if np.allclose(scipy_result, numba_result, rtol=1e-5):
        print("\n✓ Results match!")
    else:
        print("\n✗ WARNING: Results differ!")
        print(f"   Max difference: {np.max(np.abs(scipy_result - numba_result))}")

    # Calculate speedup
    speedup = results['scipy']['avg_time'] / results['numba']['avg_time']
    print(f"\nSpeedup: {speedup:.2f}x")

    return results


if __name__ == "__main__":
    # Test with sample data
    print("Testing optimized pdist implementations...")

    # Generate sample sequences
    np.random.seed(42)
    n_seqs = 500
    seq_len = 178
    bases = ['A', 'C', 'G', 'T']

    repeats = []
    for _ in range(n_seqs):
        seq = ''.join(np.random.choice(bases, size=seq_len))
        repeats.append(seq)

    results = benchmark_pdist_methods(repeats, n_runs=5)
