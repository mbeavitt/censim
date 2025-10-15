#!/usr/bin/env python3
"""
3D box counting fractal dimension calculation for sequence identity matrix.

Treats the identity matrix as a 3D structure where:
- X and Y are the matrix coordinates (repeat indices)
- Z is the similarity value
- Applies a threshold to create a binary 3D structure
- Calculates fractal dimension using 3D box counting
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def matrix_to_3d_points(matrix):
    """
    Convert identity matrix to 3D point coordinates.

    Args:
        matrix: 2D identity matrix (n x n)

    Returns:
        3D array of shape (n*n, 3) with columns [i, j, similarity]
    """
    n = matrix.shape[0]

    # Create coordinate arrays
    i_coords, j_coords = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')

    # Flatten and combine
    points_3d = np.column_stack([
        i_coords.ravel(),
        j_coords.ravel(),
        matrix.ravel()
    ])

    return points_3d


def box_count_3d(points_3d, box_sizes=None):
    """
    Perform 3D box counting on a point cloud.

    Args:
        points_3d: array of shape (n_points, 3) with columns [x, y, z]
        box_sizes: list of box sizes to test (default: powers of 2)

    Returns:
        tuple of (box_sizes, scales, counts) for fractal dimension calculation
    """
    # Get bounds of the point cloud
    mins = points_3d.min(axis=0)
    maxs = points_3d.max(axis=0)
    ranges = maxs - mins
    max_range = ranges.max()

    print(f"\nPoint cloud bounds:")
    print(f"  X (i): min={mins[0]:.4f}, max={maxs[0]:.4f}, range={ranges[0]:.4f}")
    print(f"  Y (j): min={mins[1]:.4f}, max={maxs[1]:.4f}, range={ranges[1]:.4f}")
    print(f"  Z (similarity): min={mins[2]:.4f}, max={maxs[2]:.4f}, range={ranges[2]:.4f}")
    print(f"  Max range: {max_range:.4f}")

    # Generate box sizes if not provided
    if box_sizes is None:
        box_sizes = []
        size = max_range / 256  # Start small
        while size <= max_range / 2:
            box_sizes.append(size)
            size *= 2

        # Add intermediate sizes for better fitting
        for i in range(len(box_sizes) - 1):
            mid = (box_sizes[i] + box_sizes[i+1]) / 2
            if mid not in box_sizes and mid > box_sizes[i]:
                box_sizes.append(mid)

        box_sizes = sorted(box_sizes)

    print(f"\nTesting 3D box sizes: {[f'{s:.4f}' for s in box_sizes]}")

    counts = []
    scales = []

    for box_size in box_sizes:
        # Calculate which box each point belongs to
        box_indices = ((points_3d - mins) / box_size).astype(int)

        # Count unique boxes that contain points
        unique_boxes = np.unique(box_indices, axis=0)
        count = len(unique_boxes)

        counts.append(count)
        scales.append(1.0 / box_size)

        # Print box dimensions
        print(f"Box size {box_size:.4f}: {count} boxes")
        print(f"  Box dimensions: X={box_size:.4f}, Y={box_size:.4f}, Z={box_size:.4f}")
        print(f"  Number of divisions: X={int(ranges[0]/box_size)+1}, Y={int(ranges[1]/box_size)+1}, Z={int(ranges[2]/box_size)+1}")

    return box_sizes, scales, counts


def calculate_fractal_dimension(scales, counts):
    """
    Calculate fractal dimension from box counting results.

    Args:
        scales: list of scales (1/box_size)
        counts: list of box counts

    Returns:
        tuple of (fractal_dimension, r_squared, coeffs)
    """
    log_scales = np.log(scales)
    log_counts = np.log(counts)

    # Linear regression in log-log space
    coeffs = np.polyfit(log_scales, log_counts, 1)
    fractal_dimension = coeffs[0]

    # Calculate R²
    fitted_log_counts = coeffs[0] * log_scales + coeffs[1]
    ss_res = np.sum((log_counts - fitted_log_counts) ** 2)
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return fractal_dimension, r_squared, coeffs


def plot_results(points_3d, box_sizes, scales, counts, fractal_dimension,
                r_squared, n_repeats, repeat_len):
    """
    Create visualization of 3D box counting results.

    Args:
        points_3d: 3D point cloud array
        box_sizes: list of box sizes tested
        scales: list of scales (1/box_size)
        counts: list of box counts
        fractal_dimension: calculated fractal dimension
        r_squared: R² value of the fit
        n_repeats: number of repeats
        repeat_len: length of each repeat
    """
    log_scales = np.log(scales)
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_scales, log_counts, 1)

    fig = plt.figure(figsize=(16, 5))

    # Plot 1: 3D structure visualization
    ax1 = fig.add_subplot(131, projection='3d')

    ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
               c=points_3d[:, 2], cmap='viridis', marker='.', s=1, alpha=0.5)
    ax1.set_xlabel('Repeat Index (i)')
    ax1.set_ylabel('Repeat Index (j)')
    ax1.set_zlabel('Similarity')
    ax1.set_title('3D Identity Surface')

    # Plot 2: Log-log plot
    ax2 = fig.add_subplot(132)
    ax2.loglog(scales, counts, 'bo-', label='Data', markersize=8)
    fit_counts = np.exp(coeffs[1]) * np.array(scales) ** coeffs[0]
    ax2.loglog(scales, fit_counts, 'r--', linewidth=2,
              label=f'Fit: D={fractal_dimension:.4f}\nR²={r_squared:.4f}')
    ax2.set_xlabel('Scale (1/box size)', fontsize=11)
    ax2.set_ylabel('Number of boxes', fontsize=11)
    ax2.set_title('3D Box Counting Method', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Box count vs box size
    ax3 = fig.add_subplot(133)
    ax3.plot(box_sizes, counts, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Box Size', fontsize=11)
    ax3.set_ylabel('Number of Boxes', fontsize=11)
    ax3.set_title('Box Count vs Box Size', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    plt.suptitle(f'3D Fractal Dimension: {fractal_dimension:.4f} | {n_repeats} repeats, {repeat_len}bp',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = './output/fractal_3d_box_count.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved as {output_file}")


def main():
    """Main function to perform 3D box counting fractal dimension analysis."""

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

    # Convert to 3D points
    print("\n=== Converting to 3D point cloud ===")
    points_3d = matrix_to_3d_points(identity_matrix)
    print(f"3D point cloud shape: {points_3d.shape}")
    print(f"Total points: {len(points_3d):,}")
    print(f"Original ranges:")
    print(f"  X range: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}]")
    print(f"  Y range: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}]")
    print(f"  Z range: [{points_3d[:, 2].min():.4f}, {points_3d[:, 2].max():.4f}]")

    # Normalize Z to match X/Y scale
    xy_range = max(points_3d[:, 0].max() - points_3d[:, 0].min(),
                   points_3d[:, 1].max() - points_3d[:, 1].min())
    z_range = points_3d[:, 2].max() - points_3d[:, 2].min()
    z_scale = xy_range if z_range > 0 else 1.0

    print(f"\nNormalizing Z dimension:")
    print(f"  XY range: {xy_range:.2f}")
    print(f"  Z range: {z_range:.4f}")
    print(f"  Z scale factor: {z_scale:.2f}")

    points_3d[:, 2] = points_3d[:, 2] * z_scale

    print(f"Normalized ranges:")
    print(f"  X range: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}]")
    print(f"  Y range: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}]")
    print(f"  Z range: [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}]")

    # Perform 3D box counting
    print("\n=== Performing 3D box counting ===")
    box_sizes, scales, counts = box_count_3d(points_3d)

    # Calculate fractal dimension
    print("\n=== Calculating fractal dimension ===")
    fractal_dimension, r_squared, coeffs = calculate_fractal_dimension(scales, counts)

    print(f"\n3D Fractal Dimension: {fractal_dimension:.4f}")
    print(f"R²: {r_squared:.4f}")
    print(f"Intercept: {coeffs[1]:.4f}")

    # Create output directory if needed
    import os
    os.makedirs('./output', exist_ok=True)

    # Plot results
    print("\n=== Creating visualizations ===")
    plot_results(points_3d, box_sizes, scales, counts, fractal_dimension,
                r_squared, n_repeats, repeat_len)

    print("\n=== Analysis complete ===")
    print(f"3D Fractal Dimension: {fractal_dimension:.4f} (R² = {r_squared:.4f})")


if __name__ == "__main__":
    main()
