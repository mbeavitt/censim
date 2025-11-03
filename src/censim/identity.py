#!/usr/bin/env python3
"""
Sequence identity computation for centromere analysis.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


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
