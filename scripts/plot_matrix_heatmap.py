#!/usr/bin/env python3
"""
Create combined plot with rotated distance matrix on top and correlation heatmap below.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from pathlib import Path
import re

from censim.identity import all_vs_all_identity_scipy
from censim.correlation_dimension import (
    hamming_distance_matrix,
    sliding_window_local_correlation,
    estimate_D2_from_C_r,
)


def plot_matrix_and_heatmap(D, positions, mean_corr, r_values, window_size,
                            d_values=None, generation=None, output_file='./output/matrix_with_heatmap.png',
                            image_format='jpeg', jpeg_quality=85):
    """
    Create a combined plot with rotated distance matrix on top, correlation heatmap in middle, and D2 plot below.

    Args:
        D: distance matrix
        positions: window positions
        mean_corr: (n_windows, n_radii) array of mean correlation values
        r_values: array of radii used
        window_size: window size used
        d_values: array of correlation dimension (D2) values at each position (optional)
        generation: generation number to display (optional)
        output_file: output filename
        image_format: output format - 'png', 'jpeg', or 'webp' (default: 'jpeg')
        jpeg_quality: JPEG quality 1-100 (default: 85)
    """
    n = D.shape[0]

    # Prepare upper triangle and rotate
    masked_matrix = np.copy(D).astype(float)
    for i in range(n):
        for j in range(i):
            masked_matrix[i, j] = np.nan

    rotated = rotate(masked_matrix, 45, reshape=True, order=0, cval=np.nan)

    # Create figure with 3 subplots if d_values provided, else 2
    if d_values is not None:
        fig, (ax_matrix, ax_heatmap, ax_d) = plt.subplots(3, 1, figsize=(16, 12),
                                                            gridspec_kw={'height_ratios': [2, 1, 0.8]})
    else:
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

    # Bottom: D2 values plot (if provided)
    if d_values is not None:
        ax_d.plot(positions, d_values, 'b-', linewidth=1.5)
        ax_d.set_xlabel('Position (sequence index)', fontsize=11, fontweight='bold')
        ax_d.set_ylabel('D₂', fontsize=11, fontweight='bold')
        ax_d.set_title('Correlation Dimension (D₂) Along Sequence', fontsize=12, fontweight='bold')
        ax_d.grid(True, alpha=0.3)
        ax_d.set_xlim(0, n)

        # Add horizontal line at D2=1 for reference
        ax_d.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='D₂ = 1')
        ax_d.legend(loc='upper right')

    plt.tight_layout(pad=0.5)
    # Use fixed bbox instead of 'tight' to ensure consistent image dimensions
    save_kwargs = {'dpi': 150, 'bbox_inches': None}

    if image_format.lower() in ['jpg', 'jpeg']:
        save_kwargs['format'] = 'jpeg'
        # matplotlib uses 'pil_kwargs' to pass quality to PIL
        save_kwargs['pil_kwargs'] = {'quality': jpeg_quality, 'optimize': True}
    elif image_format.lower() == 'webp':
        save_kwargs['format'] = 'webp'
        save_kwargs['pil_kwargs'] = {'quality': jpeg_quality}
    else:  # png
        save_kwargs['format'] = 'png'

    plt.savefig(output_file, **save_kwargs)
    plt.close()
    print(f"Combined matrix + heatmap saved as {output_file} ({image_format.upper()})")


def main():
    """Create combined matrix and heatmap plot."""
    parser = argparse.ArgumentParser(
        description='Create combined plot with rotated distance matrix on top and correlation heatmap below.'
    )

    parser.add_argument(
        '-i', '--input',
        dest='data_file',
        default='./data/2191000generation.out.fa',
        help='Path to input sequence data file (default: ./data/2191000generation.out.fa)'
    )

    parser.add_argument(
        '-o', '--output',
        dest='output_dir',
        default='./output/matrix_hm',
        help='Output directory for generated plots (default: ./output/matrix_hm)'
    )

    parser.add_argument(
        '-w', '--window-size',
        type=int,
        default=100,
        help='Window size for sliding window correlation (default: 100)'
    )

    parser.add_argument(
        '--r-min',
        type=float,
        default=0.006,
        help='Minimum radius for correlation analysis (default: 0.006)'
    )

    parser.add_argument(
        '--r-max',
        type=float,
        default=0.1,
        help='Maximum radius for correlation analysis (default: 0.1)'
    )

    parser.add_argument(
        '--n-radii',
        type=int,
        default=20,
        help='Number of radii to use in range [r-min, r-max] (default: 20)'
    )

    parser.add_argument(
        '--format',
        choices=['png', 'jpeg', 'jpg', 'webp'],
        default='jpeg',
        help='Output image format (default: jpeg)'
    )

    parser.add_argument(
        '--quality',
        type=int,
        default=85,
        help='JPEG/WebP quality 1-100 (default: 85, higher = better quality)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Matrix + Correlation Heatmap Plot")
    print("=" * 70)
    print(f"Window size: {args.window_size}")
    print(f"Radius range: [{args.r_min}, {args.r_max}]")
    print(f"Data file: {args.data_file}")
    print()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading sequence data...")
    with open(args.data_file, "r") as file:
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

    # Choose radii in the specified range
    print(f"\nChoosing radii in range [{args.r_min}, {args.r_max}]...")
    r_values = np.linspace(args.r_min, args.r_max, args.n_radii)
    print(f"Selected {len(r_values)} radii")

    # Compute sliding window
    print("\nComputing sliding window local correlation...")
    positions, mean_corr = sliding_window_local_correlation(
        D, args.window_size, r_values
    )
    print(f"Computed {len(positions)} windows")

    # Compute D2 values at each window position
    print("\nComputing correlation dimension (D2) at each position...")
    d_values = []
    for i, pos in enumerate(positions):
        # Fit D2 from the correlation values at this position
        slope, intercept, mask = estimate_D2_from_C_r(r_values, mean_corr[i, :])
        d_values.append(slope if not np.isnan(slope) else 0.0)
    d_values = np.array(d_values)
    print(f"D2 range: [{np.min(d_values):.3f}, {np.max(d_values):.3f}]")
    print(f"D2 mean: {np.mean(d_values):.3f} ± {np.std(d_values):.3f}")

    # Create combined plot
    print("\nCreating combined matrix + heatmap + D2 plot...")
    # Extract filename stem from data file for output naming
    data_stem = Path(args.data_file).stem  # e.g., "2191000generation.out"

    # Determine file extension based on format
    fmt = args.format.lower()
    if fmt == 'jpg':
        fmt = 'jpeg'
    ext = 'jpg' if fmt == 'jpeg' else fmt

    output_file = output_dir / f"{data_stem}_matrix_heatmap.{ext}"

    # Extract generation number from filename
    generation_match = re.search(r'(\d+)generation', data_stem)
    generation = int(generation_match.group(1)) if generation_match else None

    plot_matrix_and_heatmap(D, positions, mean_corr, r_values, args.window_size,
                           d_values=d_values, generation=generation, output_file=str(output_file),
                           image_format=fmt, jpeg_quality=args.quality)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nGenerated plot:")
    print(f"  - {output_file}")


if __name__ == "__main__":
    main()
