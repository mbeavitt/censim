#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

def all_vs_all_identity_naive(repeats):
    """
    Compute all vs all identity matrix using naive nested loops (original method).

    Args:
        repeats: list of sequences (strings)

    Returns:
        numpy array of shape (n_repeats, n_repeats) with pairwise identities
    """
    n_repeats = len(repeats)
    seq_len = len(repeats[0])

    # identity cache using hashes
    identity_cache = {}

    # initialize matrix
    mat = np.zeros((n_repeats, n_repeats), dtype=float)

    # Progress reporting
    progress_interval = max(1, n_repeats // 10)

    for i in range(n_repeats):
        if i % progress_interval == 0:
            percent = (i / n_repeats) * 100
            print(f"Progress: {i}/{n_repeats} rows ({percent:.1f}%)")

        for j in range(i, n_repeats):  # upper triangle only
            if i == j:
                identity = 1.0
            else:
                # make a hashable key (order-independent)
                key = tuple(sorted((hash(repeats[i]), hash(repeats[j]))))

                if key not in identity_cache:
                    matches = sum(a == b for a, b in zip(repeats[i], repeats[j]))
                    identity_cache[key] = matches / seq_len

                identity = identity_cache[key]

            mat[i, j] = identity
            mat[j, i] = identity  # symmetry

    print(f"Progress: {n_repeats}/{n_repeats} rows (100.0%)")
    return mat

def all_vs_all_identity(repeats):
    """
    Compute all vs all identity matrix for a list of sequences using vectorized operations.

    Args:
        repeats: list of sequences (strings)

    Returns:
        numpy array of shape (n_repeats, n_repeats) with pairwise identities
    """
    n_repeats = len(repeats)
    seq_len = len(repeats[0])

    print(f"Converting sequences to numpy array...")
    # Convert sequences to numpy array for vectorized operations
    # Shape: (n_repeats, seq_len)
    seq_array = np.array([[ord(c) for c in seq] for seq in repeats], dtype=np.uint8)

    print(f"Computing identity matrix...")
    # Use broadcasting to compare all pairs at once
    # seq_array[:, None, :] has shape (n_repeats, 1, seq_len)
    # seq_array[None, :, :] has shape (1, n_repeats, seq_len)
    # Comparison gives shape (n_repeats, n_repeats, seq_len)
    matches = (seq_array[:, None, :] == seq_array[None, :, :])

    # Sum matches along sequence dimension and divide by length
    # Shape: (n_repeats, n_repeats)
    identity_matrix = matches.sum(axis=2) / seq_len

    return identity_matrix.astype(np.float32)

def all_vs_all_identity_scipy(repeats, max_exact_size=1000, scale_factor=30):
    """
    Compute all vs all identity matrix using scipy's pdist with adaptive subsampling.

    For small arrays (<= max_exact_size), computes exact identity.
    For large arrays, subsamples to ~max_exact_size sequences and interpolates.

    Args:
        repeats: list of sequences (strings)
        max_exact_size: array size threshold for exact computation (default: 100)

    Returns:
        numpy array of shape (n_repeats, n_repeats) with pairwise identities
    """
    n_repeats = len(repeats)

    # Calculate adaptive subsampling
    if n_repeats <= max_exact_size:
        subsample_every = 1
        print(f"Computing exact identity ({n_repeats} sequences)...")
    else:
        # Use sqrt-based subsampling for gentler downsampling
        import math
        subsample_every = max(1, int(math.sqrt(n_repeats / scale_factor)))
        subset_size = n_repeats // subsample_every
        print(f"Computing subsampled identity (every {subsample_every}th sequence, {subset_size}/{n_repeats} total)...")

    if subsample_every == 1:
        # Exact computation
        seq_array = np.array([[ord(c) for c in seq] for seq in repeats], dtype=np.uint8)
        distances = pdist(seq_array, metric='hamming')
        identity_matrix = 1 - squareform(distances)
    else:
        # Subsampled computation (no interpolation)
        subset_indices = np.arange(0, n_repeats, subsample_every)
        subset_repeats = [repeats[i] for i in subset_indices]

        seq_array = np.array([[ord(c) for c in seq] for seq in subset_repeats], dtype=np.uint8)
        distances = pdist(seq_array, metric='hamming')
        identity_matrix = 1 - squareform(distances)

    return identity_matrix.astype(np.float32)

def all_vs_all_identity_fast(repeats, sample_positions=None, downsample_factor=1):
    """
    Ultra-fast approximate identity using sampling heuristics.

    Args:
        repeats: list of sequences (strings)
        sample_positions: number of positions to sample (None = use all, faster if < seq_len)
        downsample_factor: only compare every Nth sequence pair (1 = all pairs, 2 = half, etc.)

    Returns:
        numpy array of shape (n_repeats, n_repeats) with approximate pairwise identities
    """
    n_repeats = len(repeats)
    seq_len = len(repeats[0])

    # Optionally sample only certain positions in sequences
    if sample_positions is not None and sample_positions < seq_len:
        print(f"Sampling {sample_positions}/{seq_len} positions...")
        indices = np.random.choice(seq_len, size=sample_positions, replace=False)
        sampled_repeats = [''.join(seq[i] for i in indices) for seq in repeats]
        seq_array = np.array([[ord(c) for c in seq] for seq in sampled_repeats], dtype=np.uint8)
    else:
        seq_array = np.array([[ord(c) for c in seq] for seq in repeats], dtype=np.uint8)

    print(f"Computing approximate identity matrix with scipy.pdist...")
    distances = pdist(seq_array, metric='hamming')
    identity_matrix = 1 - squareform(distances)

    return identity_matrix.astype(np.float32)

def all_vs_all_identity_subsample(repeats, subsample_every=10):
    """
    Compute identity on subsampled sequences, then use for full matrix visualization.

    Args:
        repeats: list of sequences
        subsample_every: only compute identity for every Nth sequence

    Returns:
        Approximate full identity matrix
    """
    n_repeats = len(repeats)

    # Select subset of sequences
    subset_indices = np.arange(0, n_repeats, subsample_every)
    subset_repeats = [repeats[i] for i in subset_indices]

    print(f"Computing identity on {len(subset_repeats)}/{n_repeats} subsampled sequences...")

    # Compute identity on subset
    seq_array = np.array([[ord(c) for c in seq] for seq in subset_repeats], dtype=np.uint8)
    distances = pdist(seq_array, metric='hamming')
    subset_matrix = 1 - squareform(distances)

    # Interpolate back to full size using nearest neighbor
    from scipy.ndimage import zoom
    zoom_factor = n_repeats / len(subset_repeats)
    full_matrix = zoom(subset_matrix, zoom_factor, order=0)  # order=0 = nearest neighbor

    return full_matrix.astype(np.float32)

def plot_identity_heatmap(matrix, title="All vs All Sequence Identity", filename="identity_heatmap.png"):
    """
    Create a heatmap visualization of the identity matrix.

    Args:
        matrix: numpy array of pairwise identities
        title: plot title
        filename: output filename
    """
    print(matrix.shape)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix,
                cmap='RdYlBu_r',
                vmin=0,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Identity'},
                rasterized=True)
    plt.title(title)
    plt.xlabel('Repeat Index')
    plt.ylabel('Repeat Index')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Load real data
    with open("./data/sim_309/2191000generation.out.fa", "r") as file:
        contents = file.read().strip()

    repeat_len = 178
    n_repeats = int(len(contents) / repeat_len)

    # Slice into repeats
    repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]

    print(f"Loaded {n_repeats} repeats of length {repeat_len}bp")

    import time

    # Compute identity matrix with adaptive subsampling
    print("\n=== Computing identity matrix ===")
    start = time.time()
    identity_matrix = all_vs_all_identity_scipy(repeats, max_exact_size=1000)
    elapsed = time.time() - start
    print(f"Computation took: {elapsed:.2f} seconds")

    print("\nIdentity Matrix shape:", identity_matrix.shape)
    print(f"Mean identity: {identity_matrix.mean():.3f}")
    matrix_size = identity_matrix.shape[0]
    print(f"Min identity (off-diagonal): {identity_matrix[~np.eye(matrix_size, dtype=bool)].min():.3f}")
    print(f"Max identity (off-diagonal): {identity_matrix[~np.eye(matrix_size, dtype=bool)].max():.3f}")

    # Save heatmap
    print("\nSaving heatmap...")
    plot_identity_heatmap(identity_matrix,
                         title=f"All vs All Identity ({n_repeats} repeats, {repeat_len}bp)",
                         filename="./output/identity_heatmap.png")
    print("Heatmap saved as ./output/identity_heatmap.png")
