#!/usr/bin/env python3
"""
Sequence identity computation for centromere analysis.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from numba import njit, prange


@njit(fastmath=True)
def _hamming_identity_matrix_numba_serial(seq_array):
    """
    Compute pairwise Hamming identity using numba serial (single-threaded).

    Args:
        seq_array: (N, L) uint8 numpy array where N is number of sequences, L is length

    Returns:
        identity_matrix: (N, N) float32 numpy array with pairwise identities (0-1 range)
    """
    N, L = seq_array.shape
    identity = np.zeros((N, N), dtype=np.float32)

    # Set diagonal to 1.0
    for i in range(N):
        identity[i, i] = 1.0

    # Compute upper triangle serially
    for i in range(N):
        for j in range(i + 1, N):
            matches = 0
            for k in range(L):
                if seq_array[i, k] == seq_array[j, k]:
                    matches += 1
            ident = matches / L
            identity[i, j] = ident
            identity[j, i] = ident  # Mirror to lower triangle

    return identity


@njit(parallel=True, fastmath=True)
def _hamming_identity_matrix_numba_parallel(seq_array):
    """
    Compute pairwise Hamming identity using numba parallel (46x faster than scipy).

    Args:
        seq_array: (N, L) uint8 numpy array where N is number of sequences, L is length

    Returns:
        identity_matrix: (N, N) float32 numpy array with pairwise identities (0-1 range)
    """
    N, L = seq_array.shape
    identity = np.zeros((N, N), dtype=np.float32)

    # Set diagonal to 1.0
    for i in range(N):
        identity[i, i] = 1.0

    # Compute upper triangle in parallel
    for i in prange(N):
        for j in range(i + 1, N):
            matches = 0
            for k in range(L):
                if seq_array[i, k] == seq_array[j, k]:
                    matches += 1
            ident = matches / L
            identity[i, j] = ident
            identity[j, i] = ident  # Mirror to lower triangle

    return identity


def _convert_sequences_fast(repeats):
    """Fast conversion of string sequences to uint8 numpy array."""
    if len(repeats) == 0:
        return np.array([], dtype=np.uint8)

    N = len(repeats)
    L = len(repeats[0])

    seq_array = np.empty((N, L), dtype=np.uint8)
    for i, seq in enumerate(repeats):
        seq_array[i] = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)

    return seq_array


def all_vs_all_identity_numba(repeats, max_exact_size=1000, scale_factor=30, use_parallel=False):
    """
    Compute all vs all identity matrix using numba (serial or parallel).

    For small arrays (<= max_exact_size), computes exact identity.
    For large arrays, subsamples to reduce computation time.

    Args:
        repeats: list of sequences (strings)
        max_exact_size: array size threshold for exact computation (default: 1000)
        scale_factor: scaling factor for subsampling (default: 30)
        use_parallel: whether to use parallel numba (default: False)

    Returns:
        numpy array of shape (n_repeats, n_repeats) with pairwise identities
    """
    n_repeats = len(repeats)

    # Calculate adaptive subsampling
    if n_repeats <= max_exact_size:
        subsample_every = 1
    else:
        # Use sqrt-based subsampling for gentler downsampling
        import math
        subsample_every = max(1, int(math.sqrt(n_repeats / scale_factor)))

    # Choose compute function
    compute_func = _hamming_identity_matrix_numba_parallel if use_parallel else _hamming_identity_matrix_numba_serial

    if subsample_every == 1:
        # Exact computation with numba
        seq_array = _convert_sequences_fast(repeats)
        identity_matrix = compute_func(seq_array)
    else:
        # Subsampled computation
        subset_indices = np.arange(0, n_repeats, subsample_every)
        subset_repeats = [repeats[i] for i in subset_indices]
        seq_array = _convert_sequences_fast(subset_repeats)
        identity_matrix = compute_func(seq_array)

    return identity_matrix


def all_vs_all_identity_scipy(repeats, max_exact_size=1000, scale_factor=30):
    """
    Compute all vs all identity matrix using scipy's pdist with adaptive subsampling.

    For small arrays (<= max_exact_size), computes exact identity.
    For large arrays, subsamples to ~max_exact_size sequences.

    Args:
        repeats: list of sequences (strings)
        max_exact_size: array size threshold for exact computation (default: 1000)
        scale_factor: scaling factor for subsampling (default: 30)

    Returns:
        numpy array of shape (n_repeats, n_repeats) with pairwise identities
    """
    n_repeats = len(repeats)

    # Calculate adaptive subsampling
    if n_repeats <= max_exact_size:
        subsample_every = 1
    else:
        # Use sqrt-based subsampling for gentler downsampling
        import math
        subsample_every = max(1, int(math.sqrt(n_repeats / scale_factor)))
        subset_size = n_repeats // subsample_every

    if subsample_every == 1:
        # Exact computation
        # Faster string-to-array conversion using numpy's vectorized operations
        seq_len = len(repeats[0])
        seq_array = np.empty((n_repeats, seq_len), dtype=np.uint8)
        for i, seq in enumerate(repeats):
            seq_array[i] = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        distances = pdist(seq_array, metric='hamming')
        identity_matrix = 1 - squareform(distances)
    else:
        # Subsampled computation (no interpolation)
        subset_indices = np.arange(0, n_repeats, subsample_every)
        subset_repeats = [repeats[i] for i in subset_indices]

        # Faster string-to-array conversion using numpy's vectorized operations
        seq_len = len(subset_repeats[0])
        seq_array = np.empty((len(subset_repeats), seq_len), dtype=np.uint8)
        for i, seq in enumerate(subset_repeats):
            seq_array[i] = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        distances = pdist(seq_array, metric='hamming')
        identity_matrix = 1 - squareform(distances)

    return identity_matrix.astype(np.float32)
