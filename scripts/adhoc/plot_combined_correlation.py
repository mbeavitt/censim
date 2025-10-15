#!/usr/bin/env python3
"""
Create a combined plot showing rotated distance matrix and sliding window correlation.
Similar to the combined_matrix_fractal.png but for correlation-based analysis.
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from pathlib import Path

from identity import all_vs_all_identity_scipy
from correlation_dimension import (
    hamming_distance_matrix,
    sliding_window_local_correlation,
    choose_meaningful_radii
)


def plot_combined_matrix_correlation(D, positions, max_corr, r_value,
                                    window_size, threshold_equiv=None,
                                    output_file='./output/combined_matrix_correlation.png'):
    """
    Create a combined plot with rotated distance matrix and sliding window correlation.

    Args:
        D: distance matrix
        positions: window positions
        max_corr: max correlation values for windows
        r_value: the radius value being plotted
        window_size: window size used
        threshold_equiv: optional equivalent threshold for comparison
        output_file: output filename
    """
    n = D.shape[0]

    # Prepare upper triangle and rotate
    # For distance matrix, we want to show low distances (high identity) as light
    masked_matrix = np.copy(D).astype(float)
    for i in range(n):
        for j in range(i):
            masked_matrix[i, j] = np.nan

    rotated = rotate(masked_matrix, 45, reshape=True, order=0, cval=np.nan)

    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 8))

    # Top: rotated matrix
    ax1 = plt.subplot(2, 1, 1)
    # Use reversed colormap so low distances (high identity) appear dark like the binary version
    im = plt.imshow(rotated, cmap='binary_r', interpolation='nearest', aspect='auto',
                   vmin=0, vmax=1)
    # Shift the image down by adjusting the y limits
    ylim = ax1.get_ylim()
    ax1.set_ylim(ylim[0] * 0.5, ylim[1])

    if threshold_equiv is not None:
        title = f'Distance Matrix (45° rotation)\nWhite = high identity (d < {r_value:.3f}, ≈ threshold {threshold_equiv:.2f})'
    else:
        title = f'Distance Matrix (45° rotation)\nWhite = high identity'

    plt.title(title)
    plt.axis('off')

    # Bottom: sliding window correlation aligned
    ax2 = plt.subplot(2, 1, 2)
    plt.plot(positions, max_corr, 'b-', linewidth=1.5)
    plt.xlabel('Position (sequence index)')
    plt.ylabel('Max Local Correlation')
    plt.title(f'Sliding Window Max C_i(r) at r={r_value:.3f} (window={window_size})')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, n)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved as {output_file}")


def main():
    """Create combined plot for correlation-based analysis."""

    # Configuration
    if len(sys.argv) > 1:
        r_target = float(sys.argv[1])  # Target radius (distance threshold)
    else:
        r_target = 0.10  # Default: 10% distance = 90% identity

    if len(sys.argv) > 2:
        window_size = int(sys.argv[2])
    else:
        window_size = 100

    if len(sys.argv) > 3:
        data_file = sys.argv[3]
    else:
        data_file = "./data/2191000generation.out.fa"

    threshold_equiv = 1.0 - r_target

    print("=" * 70)
    print("Combined Matrix + Correlation Plot")
    print("=" * 70)
    print(f"Target distance r: {r_target:.3f} (≈ identity threshold {threshold_equiv:.2f})")
    print(f"Window size: {window_size}")
    print(f"Data file: {data_file}")
    print()

    # Ensure output directory exists
    Path("./output").mkdir(exist_ok=True)

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

    # Convert to distance
    D = hamming_distance_matrix(identity_matrix)
    print(f"Distance matrix computed (mean: {D.mean():.3f})")

    # Choose radii
    print("\nChoosing radii for analysis...")
    r_values = choose_meaningful_radii(D, n_radii=15)

    # Make sure r_target is in the list
    if r_target not in r_values:
        r_values = np.sort(np.append(r_values, r_target))

    print(f"Selected {len(r_values)} radii")

    # Compute sliding window
    print(f"\nComputing sliding window local correlation...")
    positions, mean_corr, max_corr = sliding_window_local_correlation(
        D, window_size, r_values
    )

    # Find index closest to r_target
    r_idx = np.argmin(np.abs(r_values - r_target))
    actual_r = r_values[r_idx]
    print(f"Using r = {actual_r:.3f} (closest to target {r_target:.3f})")

    max_corr_at_r = max_corr[:, r_idx]

    # Create combined plot
    print("\nCreating combined plot...")
    plot_combined_matrix_correlation(
        D, positions, max_corr_at_r, actual_r, window_size,
        threshold_equiv=threshold_equiv
    )

    # Also create versions at other scales
    print("\nCreating multi-scale combined plots...")

    # High similarity (low distance)
    r_high_sim = r_values[len(r_values)//4]
    r_idx_hs = len(r_values)//4
    plot_combined_matrix_correlation(
        D, positions, max_corr[:, r_idx_hs], r_high_sim, window_size,
        threshold_equiv=1.0-r_high_sim,
        output_file='./output/combined_matrix_correlation_high_sim.png'
    )

    # Medium similarity
    r_med_sim = r_values[len(r_values)//2]
    r_idx_ms = len(r_values)//2
    plot_combined_matrix_correlation(
        D, positions, max_corr[:, r_idx_ms], r_med_sim, window_size,
        threshold_equiv=1.0-r_med_sim,
        output_file='./output/combined_matrix_correlation_med_sim.png'
    )

    # Low similarity (high distance)
    r_low_sim = r_values[3*len(r_values)//4]
    r_idx_ls = 3*len(r_values)//4
    plot_combined_matrix_correlation(
        D, positions, max_corr[:, r_idx_ls], r_low_sim, window_size,
        threshold_equiv=1.0-r_low_sim,
        output_file='./output/combined_matrix_correlation_low_sim.png'
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  - ./output/combined_matrix_correlation.png (main)")
    print("  - ./output/combined_matrix_correlation_high_sim.png")
    print("  - ./output/combined_matrix_correlation_med_sim.png")
    print("  - ./output/combined_matrix_correlation_low_sim.png")


if __name__ == "__main__":
    main()
