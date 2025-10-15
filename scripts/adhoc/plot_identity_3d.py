#!/usr/bin/env python3
"""
3D visualization of sequence identity matrix.

Creates two 3D surface plots:
1. Similarity values (identity scores)
2. Distance values (1 - similarity)
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time


def all_vs_all_identity_scipy(repeats, max_exact_size=1000, scale_factor=30):
    """
    Compute all vs all identity matrix using scipy's pdist with adaptive subsampling.

    For small arrays (<= max_exact_size), computes exact identity.
    For large arrays, subsamples sequences and interpolates.

    Args:
        repeats: list of sequences (strings)
        max_exact_size: array size threshold for exact computation (default: 1000)
        scale_factor: controls subsampling rate for large arrays (default: 30)

    Returns:
        numpy array of shape (n_repeats, n_repeats) with pairwise identities
    """
    from scipy.spatial.distance import pdist, squareform

    n_repeats = len(repeats)

    # Calculate adaptive subsampling
    if n_repeats <= max_exact_size:
        subsample_every = 1
        print(f"Computing exact identity ({n_repeats} sequences)...")
    else:
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


def plot_identity_3d(matrix, title="3D Identity Surface", filename="identity_3d.png",
                     z_label="Similarity", colormap='viridis'):
    """
    Create a 3D surface plot of the identity matrix.

    Args:
        matrix: numpy array of pairwise identities
        title: plot title
        filename: output filename
        z_label: label for z-axis
        colormap: matplotlib colormap to use
    """
    n = matrix.shape[0]

    # Create coordinate matrices for X and Y
    x = np.arange(0, n, 1)
    y = np.arange(0, n, 1)
    X, Y = np.meshgrid(x, y)

    # Z is the identity/distance values
    Z = matrix

    # Use actual min/max from data
    z_min = Z.min()
    z_max = Z.max()

    # Create 3D plot
    fig = plt.figure(figsize=(14, 6))

    # First subplot: 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=colormap,
                           linewidth=0, antialiased=True,
                           vmin=z_min, vmax=z_max, alpha=0.9)

    ax1.set_xlabel('Repeat Index (i)', fontsize=10)
    ax1.set_ylabel('Repeat Index (j)', fontsize=10)
    ax1.set_zlabel(z_label, fontsize=10)
    ax1.set_title(title, fontsize=12, pad=20)
    ax1.set_zlim(z_min, z_max)

    # Add colorbar
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.1)

    # Adjust viewing angle for better visualization
    ax1.view_init(elev=30, azim=45)

    # Second subplot: top-down view (heatmap style)
    ax2 = fig.add_subplot(122)
    im = ax2.imshow(Z, cmap=colormap, origin='lower',
                   extent=[0, n, 0, n], vmin=z_min, vmax=z_max,
                   interpolation='nearest', aspect='auto')
    ax2.set_xlabel('Repeat Index (i)', fontsize=10)
    ax2.set_ylabel('Repeat Index (j)', fontsize=10)
    ax2.set_title('Top-down View', fontsize=12)

    # Add colorbar
    fig.colorbar(im, ax=ax2, shrink=0.8, aspect=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"3D plot saved as {filename}")


def main():
    """Main function to load data and create 3D visualizations."""

    # Load data
    print("Loading sequence data...")
    with open("./data/2191000generation.out.fa", "r") as file:
        contents = file.read().strip()

    repeat_len = 178
    n_repeats = int(len(contents) / repeat_len)

    # Slice into repeats
    repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]

    print(f"Loaded {n_repeats} repeats of length {repeat_len}bp")

    # Compute identity matrix
    print("\n=== Computing identity matrix ===")
    start = time.time()
    identity_matrix = all_vs_all_identity_scipy(repeats, scale_factor=30, max_exact_size=1000)
    elapsed = time.time() - start
    print(f"Computation took: {elapsed:.2f} seconds")

    print("\nIdentity Matrix Statistics:")
    print(f"  Shape: {identity_matrix.shape}")
    print(f"  Mean identity: {identity_matrix.mean():.4f}")

    matrix_size = identity_matrix.shape[0]
    off_diagonal_mask = ~np.eye(matrix_size, dtype=bool)
    print(f"  Min identity (off-diagonal): {identity_matrix[off_diagonal_mask].min():.4f}")
    print(f"  Max identity (off-diagonal): {identity_matrix[off_diagonal_mask].max():.4f}")

    # Create distance matrix (1 - similarity)
    distance_matrix = 1.0 - identity_matrix

    print("\nDistance Matrix Statistics:")
    print(f"  Mean distance: {distance_matrix.mean():.4f}")
    print(f"  Min distance (off-diagonal): {distance_matrix[off_diagonal_mask].min():.4f}")
    print(f"  Max distance (off-diagonal): {distance_matrix[off_diagonal_mask].max():.4f}")

    # Create output directory if needed
    import os
    os.makedirs('./output', exist_ok=True)

    # Plot 1: Similarity (identity) values
    print("\n=== Creating 3D similarity plot ===")
    plot_identity_3d(
        identity_matrix,
        title=f"Sequence Similarity (Identity)\n{n_repeats} repeats, {repeat_len}bp",
        filename="./output/identity_similarity_3d.png",
        z_label="Similarity (Identity)",
        colormap='viridis'
    )

    # Plot 2: Distance (1 - similarity) values
    print("=== Creating 3D distance plot ===")
    plot_identity_3d(
        distance_matrix,
        title=f"Sequence Distance (1 - Identity)\n{n_repeats} repeats, {repeat_len}bp",
        filename="./output/identity_distance_3d.png",
        z_label="Distance (1 - Similarity)",
        colormap='plasma'
    )

    print("\n=== Visualization complete ===")
    print("Generated files:")
    print("  - ./output/identity_similarity_3d.png")
    print("  - ./output/identity_distance_3d.png")


if __name__ == "__main__":
    main()
