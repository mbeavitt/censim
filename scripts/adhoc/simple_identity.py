#!/usr/bin/env python3
import numpy as np
import sys
from scipy.spatial.distance import pdist, squareform

def all_vs_all_identity_scipy(repeats, max_exact_size=1000, scale_factor=30):
    """
    Compute all vs all identity matrix using scipy's pdist with adaptive subsampling.
    """
    n_repeats = len(repeats)

    # Calculate adaptive subsampling
    if n_repeats <= max_exact_size:
        subsample_every = 1
    else:
        import math
        subsample_every = max(1, int(math.sqrt(n_repeats / scale_factor)))

    if subsample_every == 1:
        # Exact computation
        seq_array = np.array([[ord(c) for c in seq] for seq in repeats], dtype=np.uint8)
        distances = pdist(seq_array, metric='hamming')
        identity_matrix = 1 - squareform(distances)
    else:
        # Subsampled computation
        subset_indices = np.arange(0, n_repeats, subsample_every)
        subset_repeats = [repeats[i] for i in subset_indices]

        seq_array = np.array([[ord(c) for c in seq] for seq in subset_repeats], dtype=np.uint8)
        distances = pdist(seq_array, metric='hamming')
        identity_matrix = 1 - squareform(distances)

    return identity_matrix.astype(np.float32)

def box_count(matrix):
    """Calculate fractal dimension using box counting algorithm."""
    n = matrix.shape[0]
    max_box_size = n // 2
    box_sizes = []

    # Generate box sizes: powers of 2
    size = 1
    while size <= max_box_size:
        box_sizes.append(size)
        size *= 2

    # Add intermediate sizes
    for i in range(len(box_sizes) - 1):
        mid = (box_sizes[i] + box_sizes[i+1]) // 2
        if mid not in box_sizes and mid > box_sizes[i]:
            box_sizes.append(mid)

    box_sizes = sorted(box_sizes)

    counts = []
    scales = []

    for box_size in box_sizes:
        count = 0
        for i in range(0, n, box_size):
            for j in range(0, n, box_size):
                box = matrix[i:min(i+box_size, n), j:min(j+box_size, n)]
                if np.any(box):
                    count += 1
        counts.append(count)
        scales.append(1.0 / box_size)

    # Fit log-log plot
    log_scales = np.log(scales)
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_scales, log_counts, 1)

    return coeffs[0]

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) != 3:
        print("Usage: python simple_identity.py <file> <threshold>")
        print("\nExamples:")
        print("  python simple_identity.py data/sequences.fa 0.95")
        print("  python simple_identity.py data/sequences.fa 0.90")
        sys.exit(1)

    # Parse arguments
    filename = sys.argv[1]
    identity_threshold = float(sys.argv[2])

    # Load data
    try:
        with open(filename, "r") as file:
            contents = file.read().strip()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)

    # Parse sequences
    # Assume fixed-length repeats (like identity.py)
    repeat_len = 178
    n_repeats = int(len(contents) / repeat_len)

    # Slice into repeats
    repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]

    # Compute identity matrix
    identity_matrix = all_vs_all_identity_scipy(repeats, scale_factor=30, max_exact_size=1000)

    # Binarize at threshold
    identity_matrix = (identity_matrix >= identity_threshold).astype(float)

    # Compute fractal dimension
    fractal_dimension = box_count(identity_matrix)

    # Output result
    print(f"Fractal dimension: {fractal_dimension:.4f}")
