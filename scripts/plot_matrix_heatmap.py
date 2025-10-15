#!/usr/bin/env python3
"""
Create combined plot with rotated distance matrix on top and correlation heatmap below.
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from pathlib import Path
import re

from identity import all_vs_all_identity_scipy
from correlation_dimension import (
    hamming_distance_matrix,
    sliding_window_local_correlation,
)


def plot_matrix_and_heatmap(D, positions, mean_corr, r_values, window_size,
                            generation=None, output_file='./output/matrix_with_heatmap.png'):
    """
    Create a combined plot with rotated distance matrix on top and correlation heatmap below.

    Args:
        D: distance matrix
        positions: window positions
        mean_corr: (n_windows, n_radii) array of mean correlation values
        r_values: array of radii used
        window_size: window size used
        generation: generation number to display (optional)
        output_file: output filename
    """
    n = D.shape[0]

    # Prepare upper triangle and rotate
    masked_matrix = np.copy(D).astype(float)
    for i in range(n):
        for j in range(i):
            masked_matrix[i, j] = np.nan

    rotated = rotate(masked_matrix, 45, reshape=True, order=0, cval=np.nan)

    # Create figure with 2 subplots: matrix on top, heatmap on bottom
    fig, (ax_matrix, ax_heatmap) = plt.subplots(2, 1, figsize=(16, 10),
                                                 gridspec_kw={'height_ratios': [2, 1]})

    # Top: rotated matrix with viridis
    im1 = ax_matrix.imshow(rotated, cmap='viridis', interpolation='nearest', aspect='auto',
                          vmin=0, vmax=0.25)  # Clip at 0.25 for better contrast

    # Shift the image down by adjusting the y limits
    ylim = ax_matrix.get_ylim()
    ax_matrix.set_ylim(ylim[0] * 0.5, ylim[1])

    ax_matrix.set_title('Distance Matrix (45° rotation)\nDark = high identity, Light = low identity',
                       fontsize=12, fontweight='bold')
    ax_matrix.axis('off')

    # Add generation number in top left corner if provided
    if generation is not None:
        ax_matrix.text(0.02, 0.98, f'Generation: {generation}',
                      transform=ax_matrix.transAxes,
                      fontsize=10, fontweight='bold',
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#    # Add colorbar for matrix
#    cbar1 = plt.colorbar(im1, ax=ax_matrix, orientation='horizontal',
#                        pad=0.02, fraction=0.03, aspect=40)
#    cbar1.set_label('Hamming Distance', fontsize=10)

    # Bottom: correlation heatmap
    im2 = ax_heatmap.imshow(mean_corr.T, aspect='auto', cmap='viridis',
                           interpolation='nearest', origin='lower',
                           extent=[0, n, 0, len(r_values)])

    ax_heatmap.set_xlabel('Position (sequence index)', fontsize=11, fontweight='bold')
    ax_heatmap.set_ylabel('Radius (r)', fontsize=11, fontweight='bold')
    ax_heatmap.set_title(f'Mean Local Correlation Heatmap (window size = {window_size})',
                        fontsize=12, fontweight='bold')

    # Set y-ticks to show actual r values
    n_ticks = min(len(r_values), 10)
    tick_indices = np.linspace(0, len(r_values) - 1, n_ticks, dtype=int)
    ax_heatmap.set_yticks(tick_indices)
    ax_heatmap.set_yticklabels([f'{r_values[i]:.3f}' for i in tick_indices])

#    # Add colorbar for heatmap
#    cbar2 = plt.colorbar(im2, ax=ax_heatmap)
#    cbar2.set_label('Mean C_i(r)', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined matrix + heatmap saved as {output_file}")


def main():
    """Create combined matrix and heatmap plot."""

    # Configuration
    window_size = 100  # Fixed window size

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "./data/2191000generation.out.fa"

    print("=" * 70)
    print("Matrix + Correlation Heatmap Plot")
    print("=" * 70)
    print(f"Window size: {window_size}")
    print(f"Radius range: [0.006, 0.1]")
    print(f"Data file: {data_file}")
    print()

    # Ensure output directory exists
    output_dir = Path("./output/matrix_hm")
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

    # Convert to distance
    D = hamming_distance_matrix(identity_matrix)
    print(f"Distance matrix computed (mean: {D.mean():.3f})")

    # Choose radii in the focused range [0.006, 0.1]
    print("\nChoosing radii in range [0.006, 0.1]...")
    n_radii = 20  # Use more radii for smoother coverage
    r_values = np.linspace(0.006, 0.1, n_radii)
    print(f"Selected {len(r_values)} radii")

    # Compute sliding window
    print(f"\nComputing sliding window local correlation...")
    positions, mean_corr, max_corr = sliding_window_local_correlation(
        D, window_size, r_values
    )
    print(f"Computed {len(positions)} windows")

    # Create combined plot
    print("\nCreating combined matrix + heatmap plot...")
    # Extract filename stem from data file for output naming
    data_stem = Path(data_file).stem  # e.g., "2191000generation.out"
    output_file = output_dir / f"{data_stem}_matrix_heatmap.png"

    # Extract generation number from filename
    generation_match = re.search(r'(\d+)generation', data_stem)
    generation = int(generation_match.group(1)) if generation_match else None

    plot_matrix_and_heatmap(D, positions, mean_corr, r_values, window_size,
                           generation=generation, output_file=str(output_file))

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nGenerated plot:")
    print(f"  - {output_file}")


if __name__ == "__main__":
    main()
