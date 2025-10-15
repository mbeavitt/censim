#!/usr/bin/env python3
"""
Create multi-scale combined plots with rotated distance matrix and mean local correlation.
Focuses on meaningful r range (0.006 to 0.1) and uses viridis colormap.
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
)


def plot_multiscale_matrix_correlation(D, positions, mean_corr, r_values,
                                       window_size, n_scales=4,
                                       output_file='./output/multiscale_combined.png'):
    """
    Create a combined plot with rotated distance matrix and multiple scales of mean correlation.

    Args:
        D: distance matrix
        positions: window positions
        mean_corr: (n_windows, n_radii) array of mean correlation values
        r_values: array of radii used
        window_size: window size used
        n_scales: number of different r values to show
        output_file: output filename
    """
    n = D.shape[0]

    # Prepare upper triangle and rotate
    masked_matrix = np.copy(D).astype(float)
    for i in range(n):
        for j in range(i):
            masked_matrix[i, j] = np.nan

    rotated = rotate(masked_matrix, 45, reshape=True, order=0, cval=np.nan)

    # Create figure with 2 subplots: matrix on top, all correlation lines on bottom
    fig, (ax_matrix, ax_corr) = plt.subplots(2, 1, figsize=(16, 10),
                                              gridspec_kw={'height_ratios': [2, 1]})

    # Top: rotated matrix with viridis
    im = ax_matrix.imshow(rotated, cmap='viridis', interpolation='nearest', aspect='auto',
                          vmin=0, vmax=0.25)  # Clip at 0.25 for better contrast

    # Shift the image down by adjusting the y limits
    ylim = ax_matrix.get_ylim()
    ax_matrix.set_ylim(ylim[0] * 0.5, ylim[1])

    ax_matrix.set_title('Distance Matrix (45° rotation)\nDark = high identity, Light = low identity',
                       fontsize=12, fontweight='bold')
    ax_matrix.axis('off')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_matrix, orientation='horizontal',
                       pad=0.02, fraction=0.03, aspect=40)
    cbar.set_label('Hamming Distance', fontsize=10)

    # Select n_scales radii evenly spaced in the valid range
    selected_indices = np.linspace(0, len(r_values) - 1, n_scales, dtype=int)

    # Bottom: all correlation lines on one plot with different colors
    # Use a colormap for the lines
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_scales))

    for i, r_idx in enumerate(selected_indices):
        r = r_values[r_idx]
        identity_equiv = (1.0 - r) * 100  # Convert to percentage

        ax_corr.plot(positions, mean_corr[:, r_idx], '-', linewidth=2,
                    color=colors[i],
                    label=f'r={r:.4f} ({identity_equiv:.1f}%)')

    ax_corr.set_xlabel('Position (sequence index)', fontsize=11, fontweight='bold')
    ax_corr.set_ylabel('Mean C_i(r)', fontsize=11, fontweight='bold')
    ax_corr.set_title(f'Mean Local Correlation at Multiple Scales (window size = {window_size})',
                     fontsize=12, fontweight='bold')
    ax_corr.grid(True, alpha=0.3)
    ax_corr.set_xlim(0, n)
    ax_corr.set_ylim(bottom=0)
    ax_corr.legend(loc='upper right', fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Multi-scale combined plot saved as {output_file}")


def main():
    """Create multi-scale combined plot with focus on r ∈ [0.006, 0.1]."""

    # Configuration
    if len(sys.argv) > 1:
        window_size = int(sys.argv[1])
    else:
        window_size = 100

    if len(sys.argv) > 2:
        n_scales = int(sys.argv[2])
    else:
        n_scales = 5  # Number of different r values to show

    if len(sys.argv) > 3:
        data_file = sys.argv[3]
    else:
        data_file = "./data/2191000generation.out.fa"

    print("=" * 70)
    print("Multi-Scale Combined Plot (Matrix + Mean Correlation)")
    print("=" * 70)
    print(f"Window size: {window_size}")
    print(f"Number of scales: {n_scales}")
    print(f"Radius range: [0.006, 0.1]")
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

    # Choose radii in the focused range [0.006, 0.1]
    print("\nChoosing radii in range [0.006, 0.1]...")
    n_radii = 20  # Use more radii for smoother coverage
    r_values = np.linspace(0.006, 0.1, n_radii)

    print(f"Selected {len(r_values)} radii:")
    for i, r in enumerate(r_values):
        print(f"  r[{i:2d}] = {r:.4f} (≈{(1-r)*100:.1f}% identity)")

    # Compute sliding window
    print(f"\nComputing sliding window local correlation...")
    print(f"This may take a moment...")
    positions, mean_corr, max_corr = sliding_window_local_correlation(
        D, window_size, r_values
    )
    print(f"Computed {len(positions)} windows")

    # Create multi-scale combined plot
    print(f"\nCreating multi-scale combined plot with {n_scales} scales...")
    plot_multiscale_matrix_correlation(
        D, positions, mean_corr, r_values, window_size,
        n_scales=n_scales
    )

    # Also create a detailed plot showing all radii for reference
    print("\nCreating detailed correlation plot for all radii...")
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot every other radius to avoid overcrowding
    step = max(1, len(r_values) // 10)
    for i in range(0, len(r_values), step):
        r = r_values[i]
        identity_pct = (1.0 - r) * 100
        ax.plot(positions, mean_corr[:, i], '-', linewidth=1.5,
               label=f'r={r:.4f} ({identity_pct:.1f}%)', alpha=0.8)

    ax.set_xlabel('Position (sequence index)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean C_i(r)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Local Correlation at Multiple Scales', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_xlim(0, D.shape[0])

    plt.tight_layout()
    plt.savefig('./output/all_scales_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Detailed plot saved as ./output/all_scales_correlation.png")

    # Create a heatmap view of mean correlation across all windows and radii
    print("\nCreating heatmap of mean correlation...")
    fig, ax = plt.subplots(figsize=(16, 6))

    im = ax.imshow(mean_corr.T, aspect='auto', cmap='viridis',
                   interpolation='nearest', origin='lower', extent=[0, D.shape[0], 0, len(r_values)])

    ax.set_xlabel('Position (sequence index)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Radius index', fontsize=12, fontweight='bold')
    ax.set_title('Mean Local Correlation Heatmap (all windows × all radii)',
                fontsize=14, fontweight='bold')

    # Set y-ticks to show actual r values
    n_ticks = min(len(r_values), 10)
    tick_indices = np.linspace(0, len(r_values) - 1, n_ticks, dtype=int)
    ax.set_yticks(tick_indices)
    ax.set_yticklabels([f'{r_values[i]:.3f}' for i in tick_indices])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean C_i(r)', fontsize=11)

    plt.tight_layout()
    plt.savefig('./output/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Heatmap saved as ./output/correlation_heatmap.png")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    # Find the most interesting radius (highest variance = most informative)
    variances = np.var(mean_corr, axis=0)
    most_informative_idx = np.argmax(variances)
    most_informative_r = r_values[most_informative_idx]

    print(f"\nMost informative radius (highest variance):")
    print(f"  r = {most_informative_r:.4f} (≈{(1-most_informative_r)*100:.1f}% identity)")
    print(f"  Variance = {variances[most_informative_idx]:.6f}")

    # Statistics at this radius
    mc_at_best = mean_corr[:, most_informative_idx]
    print(f"\nMean C_i statistics at this radius:")
    print(f"  Mean:   {mc_at_best.mean():.4f}")
    print(f"  Std:    {mc_at_best.std():.4f}")
    print(f"  Min:    {mc_at_best.min():.4f}")
    print(f"  Max:    {mc_at_best.max():.4f}")

    # Find windows with highest/lowest correlation at this radius
    top_5_idx = np.argsort(mc_at_best)[-5:][::-1]
    bottom_5_idx = np.argsort(mc_at_best)[:5]

    print(f"\nTop 5 windows (most self-similar):")
    for idx in top_5_idx:
        print(f"  Position {positions[idx]:5d}: mean C_i = {mc_at_best[idx]:.4f}")

    print(f"\nBottom 5 windows (least self-similar):")
    for idx in bottom_5_idx:
        print(f"  Position {positions[idx]:5d}: mean C_i = {mc_at_best[idx]:.4f}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  - ./output/multiscale_combined.png (MAIN)")
    print("  - ./output/all_scales_correlation.png")
    print("  - ./output/correlation_heatmap.png")


if __name__ == "__main__":
    main()
