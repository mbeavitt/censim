#!/usr/bin/env python3
"""
Create combined plot with rotated distance matrix on top and fractal dimension heatmap below.
Calculates fractal dimension in sliding windows across multiple identity thresholds.
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from pathlib import Path
from numpy.lib.stride_tricks import as_strided
import re

from identity import all_vs_all_identity_scipy


def count_boxes_fast(binary_matrix, box_size):
    """
    Fast box counting using vectorized operations.

    Args:
        binary_matrix: 2D binary array
        box_size: size of boxes to count

    Returns:
        count: number of boxes containing at least one 1
    """
    n = binary_matrix.shape[0]

    # Calculate how many complete boxes fit
    n_boxes_per_dim = n // box_size

    # Trim to fit complete boxes only
    trimmed = binary_matrix[:n_boxes_per_dim*box_size, :n_boxes_per_dim*box_size]

    # Reshape to group pixels into boxes
    # Shape becomes (n_boxes_per_dim, box_size, n_boxes_per_dim, box_size)
    reshaped = trimmed.reshape(n_boxes_per_dim, box_size, n_boxes_per_dim, box_size)

    # Sum over the box dimensions (axes 1 and 3)
    # Result shape: (n_boxes_per_dim, n_boxes_per_dim)
    box_sums = reshaped.sum(axis=(1, 3))

    # Count boxes with at least one 1
    count = np.count_nonzero(box_sums)

    return count


def calculate_window_fractal_dimension(window, threshold):
    """
    Calculate fractal dimension for a single window at a given threshold.

    Args:
        window: 2D numpy array (identity values)
        threshold: identity threshold for binarization

    Returns:
        fractal_dimension: estimated fractal dimension (or np.nan if cannot compute)
    """
    window_size = window.shape[0]

    # Binarize the window
    binary_matrix = (window >= threshold).astype(np.uint8)

    # Generate box sizes (powers of 2)
    box_sizes = []
    size = 1
    max_box_size = window_size // 2
    while size <= max_box_size:
        box_sizes.append(size)
        size *= 2

    if len(box_sizes) < 2:
        return np.nan

    counts = []
    scales = []

    for box_size in box_sizes:
        count = count_boxes_fast(binary_matrix, box_size)
        counts.append(count)
        scales.append(1.0 / box_size)

    # Fit log-log plot
    log_scales = np.log(scales)
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_scales, log_counts, 1)
    fractal_dim = coeffs[0]

    return fractal_dim


def sliding_window_fractal_heatmap(identity_matrix, window_size, thresholds):
    """
    Calculate fractal dimension in sliding windows across multiple thresholds.

    Args:
        identity_matrix: full identity matrix
        window_size: size of sliding window
        thresholds: array of identity thresholds to test

    Returns:
        positions: array of window center positions
        fractal_dims: (n_windows, n_thresholds) array of fractal dimensions
    """
    n = identity_matrix.shape[0]
    n_thresholds = len(thresholds)

    print(f"\nSliding window fractal dimension analysis:")
    print(f"Matrix size: {n}")
    print(f"Window size: {window_size}")
    print(f"Number of thresholds: {n_thresholds}")
    print(f"Threshold range: [{thresholds.min():.3f}, {thresholds.max():.3f}]")

    # Pad the matrix to allow windows at edges
    pad_size = window_size // 2
    padded_matrix = np.pad(identity_matrix, pad_size, mode='constant', constant_values=0)
    print(f"Padded matrix size: {padded_matrix.shape[0]} (added {pad_size} on each side)")

    positions = []
    fractal_dims = []

    # Slide window across the matrix
    step = 1
    total_windows = n

    for start in range(0, n, step):
        end = start + window_size

        # Extract window from padded matrix
        window = padded_matrix[start:end, start:end]

        # Calculate fractal dimension for each threshold
        window_fractal_dims = []
        for threshold in thresholds:
            fd = calculate_window_fractal_dimension(window, threshold)
            window_fractal_dims.append(fd)

        positions.append(start)
        fractal_dims.append(window_fractal_dims)

        if (start // step) % 100 == 0:
            print(f"Progress: {start}/{total_windows} windows ({100*start/total_windows:.1f}%)")

    positions = np.array(positions)
    fractal_dims = np.array(fractal_dims)  # Shape: (n_windows, n_thresholds)

    print(f"Computed {len(positions)} windows")
    print(f"Fractal dimensions shape: {fractal_dims.shape}")

    return positions, fractal_dims


def plot_matrix_and_fractal_heatmap(identity_matrix, positions, fractal_dims,
                                     thresholds, window_size, generation=None,
                                     output_file='./output/matrix_with_fractal_heatmap.png'):
    """
    Create a combined plot with rotated identity matrix on top and fractal dimension heatmap below.

    Args:
        identity_matrix: full identity matrix
        positions: window positions
        fractal_dims: (n_windows, n_thresholds) array of fractal dimension values
        thresholds: array of thresholds used
        window_size: window size used
        generation: generation number to display (optional)
        output_file: output filename
    """
    n = identity_matrix.shape[0]

    # Prepare upper triangle and rotate
    masked_matrix = np.copy(identity_matrix).astype(float)
    for i in range(n):
        for j in range(i):
            masked_matrix[i, j] = np.nan

    rotated = rotate(masked_matrix, 45, reshape=True, order=0, cval=np.nan)

    # Create figure with 2 subplots: matrix on top, heatmap on bottom
    fig, (ax_matrix, ax_heatmap) = plt.subplots(2, 1, figsize=(16, 10),
                                                 gridspec_kw={'height_ratios': [2, 1]})

    # Top: rotated matrix with viridis
    im1 = ax_matrix.imshow(rotated, cmap='viridis', interpolation='nearest', aspect='auto',
                          vmin=0.85, vmax=1.0)  # Focus on high identity range

    # Shift the image down by adjusting the y limits
    ylim = ax_matrix.get_ylim()
    ax_matrix.set_ylim(ylim[0] * 0.5, ylim[1])

    ax_matrix.set_title('Identity Matrix (45° rotation)\nDark = low identity, Light = high identity',
                       fontsize=12, fontweight='bold')
    ax_matrix.axis('off')

    # Add generation number in top left corner if provided
    if generation is not None:
        ax_matrix.text(0.02, 0.98, f'Generation: {generation}',
                      transform=ax_matrix.transAxes,
                      fontsize=10, fontweight='bold',
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Bottom: fractal dimension heatmap
    im2 = ax_heatmap.imshow(fractal_dims.T, aspect='auto', cmap='viridis',
                           interpolation='nearest', origin='lower',
                           extent=[0, n, 0, len(thresholds)])

    ax_heatmap.set_xlabel('Position (sequence index)', fontsize=11, fontweight='bold')
    ax_heatmap.set_ylabel('Identity Threshold', fontsize=11, fontweight='bold')
    ax_heatmap.set_title(f'Fractal Dimension Heatmap (window size = {window_size})',
                        fontsize=12, fontweight='bold')

    # Set y-ticks to show actual threshold values
    # Show fewer, nicer-looking ticks
    n_ticks = 5
    tick_indices = np.linspace(0, len(thresholds) - 1, n_ticks, dtype=int)
    ax_heatmap.set_yticks(tick_indices)
    ax_heatmap.set_yticklabels([f'{thresholds[i]:.2f}' for i in tick_indices])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined matrix + fractal heatmap saved as {output_file}")


def main():
    """Create combined matrix and fractal dimension heatmap plot."""

    # Configuration
    window_size = 30  # Fixed window size

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "./data/2191000generation.out.fa"

    print("=" * 70)
    print("Matrix + Fractal Dimension Heatmap Plot")
    print("=" * 70)
    print(f"Window size: {window_size}")
    print(f"Threshold range: [0.85, 0.99]")
    print(f"Data file: {data_file}")
    print()

    # Ensure output directory exists
    output_dir = Path("./output/fractal_hm")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading sequence data...")
    with open(data_file, "r") as file:
        contents = file.read().strip()

    repeat_len = 178
    n_repeats = int(len(contents) / repeat_len)
    repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]
    print(f"Loaded {n_repeats} repeats of length {repeat_len}bp")

    # Compute identity matrix
    print("\nComputing identity matrix...")
    identity_matrix = all_vs_all_identity_scipy(repeats, scale_factor=30, max_exact_size=1000)
    print(f"Matrix shape: {identity_matrix.shape}")
    print(f"Mean identity: {identity_matrix.mean():.3f}")

    # Choose thresholds in range [0.85, 0.99]
    print("\nChoosing identity thresholds in range [0.85, 0.99]...")
    n_thresholds = 15  # Use 15 thresholds for good coverage
    thresholds = np.linspace(0.92, 0.99, n_thresholds)
    print(f"Selected {len(thresholds)} thresholds")

    # Compute sliding window fractal dimensions
    print(f"\nComputing sliding window fractal dimensions...")
    positions, fractal_dims = sliding_window_fractal_heatmap(
        identity_matrix, window_size, thresholds
    )
    print(f"Computed {len(positions)} windows")

    # Print some statistics
    print("\nFractal dimension statistics (overall):")
    print(f"  Mean: {np.nanmean(fractal_dims):.4f}")
    print(f"  Std:  {np.nanstd(fractal_dims):.4f}")
    print(f"  Min:  {np.nanmin(fractal_dims):.4f}")
    print(f"  Max:  {np.nanmax(fractal_dims):.4f}")

    print("\nFractal dimension statistics by threshold:")
    for i, threshold in enumerate(thresholds):
        mean_val = np.nanmean(fractal_dims[:, i])
        std_val = np.nanstd(fractal_dims[:, i])
        print(f"  Threshold {threshold:.2f}: Mean={mean_val:.4f}, Std={std_val:.4f}")

    # Create combined plot
    print("\nCreating combined matrix + fractal heatmap plot...")
    # Extract filename stem from data file for output naming
    data_stem = Path(data_file).stem  # e.g., "2191000generation.out"
    output_file = output_dir / f"{data_stem}_fractal_heatmap.png"

    # Extract generation number from filename
    generation_match = re.search(r'(\d+)generation', data_stem)
    generation = int(generation_match.group(1)) if generation_match else None

    plot_matrix_and_fractal_heatmap(identity_matrix, positions, fractal_dims,
                                   thresholds, window_size, generation=generation,
                                   output_file=str(output_file))

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nGenerated plot:")
    print(f"  - {output_file}")


if __name__ == "__main__":
    main()
