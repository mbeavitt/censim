#!/usr/bin/env python3
"""
Compare threshold-based fractal dimension with correlation-based local dimension.

This script runs both analyses side-by-side to help understand the relationship
between the two approaches.
"""

import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Import from original script
from identity import all_vs_all_identity_scipy, sliding_window_fractal_dimension

# Import correlation methods
from correlation_dimension import (
    hamming_distance_matrix,
    sliding_window_local_correlation,
    choose_meaningful_radii,
    local_correlation_sum
)


def main():
    """Compare threshold-based and correlation-based methods."""

    # Configuration
    if len(sys.argv) > 1:
        threshold = float(sys.argv[1])
    else:
        threshold = 0.9

    if len(sys.argv) > 2:
        window_size = int(sys.argv[2])
    else:
        window_size = 100

    if len(sys.argv) > 3:
        data_file = sys.argv[3]
    else:
        data_file = "./data/2191000generation.out.fa"

    print("=" * 70)
    print("COMPARISON: Threshold vs Correlation Methods")
    print("=" * 70)
    print(f"Threshold: {threshold}")
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

    # Compute identity matrix once
    print("\nComputing identity matrix...")
    start = time.time()
    identity_matrix = all_vs_all_identity_scipy(repeats, scale_factor=30, max_exact_size=1000)
    elapsed = time.time() - start
    print(f"Computation took: {elapsed:.2f} seconds")
    print(f"Matrix shape: {identity_matrix.shape}")

    # ====================
    # METHOD 1: Threshold-based fractal dimension
    # ====================
    print("\n" + "=" * 70)
    print("METHOD 1: Threshold-based Fractal Dimension")
    print("=" * 70)

    # Binarize
    binary_matrix = (identity_matrix >= threshold).astype(float)
    print(f"Binarized at threshold ≥ {threshold}")
    print(f"Fraction of 1s: {binary_matrix.mean():.4f}")

    print(f"\nRunning sliding window fractal dimension...")
    start = time.time()
    positions_fd, fractal_dims = sliding_window_fractal_dimension(
        binary_matrix, window_size=window_size, threshold=threshold
    )
    elapsed_fd = time.time() - start
    print(f"Fractal dimension analysis took: {elapsed_fd:.2f} seconds")

    if len(fractal_dims) > 0:
        print(f"\nFractal dimension statistics:")
        print(f"  Mean: {np.mean(fractal_dims):.4f}")
        print(f"  Std:  {np.std(fractal_dims):.4f}")
        print(f"  Min:  {np.min(fractal_dims):.4f}")
        print(f"  Max:  {np.max(fractal_dims):.4f}")

    # ====================
    # METHOD 2: Correlation-based local dimension
    # ====================
    print("\n" + "=" * 70)
    print("METHOD 2: Correlation-based Local Dimension")
    print("=" * 70)

    # Convert to distance
    D = hamming_distance_matrix(identity_matrix)
    print(f"Distance matrix computed (mean distance: {D.mean():.3f})")

    # Choose radii - use one near the threshold
    # threshold = 0.9 means distance = 0.1
    r_threshold = 1.0 - threshold
    print(f"\nUsing r ≈ {r_threshold:.3f} (equivalent to identity threshold {threshold})")

    # Also compute at a range of radii
    r_values = choose_meaningful_radii(D, n_radii=10)
    # Make sure r_threshold is in the list
    if r_threshold not in r_values:
        r_values = np.sort(np.append(r_values, r_threshold))

    print(f"Computing sliding window local correlation at {len(r_values)} radii...")
    start = time.time()
    positions_corr, mean_corr, max_corr = sliding_window_local_correlation(
        D, window_size, r_values
    )
    elapsed_corr = time.time() - start
    print(f"Correlation analysis took: {elapsed_corr:.2f} seconds")

    # Find the index closest to r_threshold
    r_idx = np.argmin(np.abs(r_values - r_threshold))
    actual_r = r_values[r_idx]
    print(f"\nUsing r = {actual_r:.3f} (closest to {r_threshold:.3f})")

    max_corr_at_r = max_corr[:, r_idx]
    mean_corr_at_r = mean_corr[:, r_idx]

    print(f"\nLocal correlation statistics (max in window):")
    print(f"  Mean: {max_corr_at_r.mean():.4f}")
    print(f"  Std:  {max_corr_at_r.std():.4f}")
    print(f"  Min:  {max_corr_at_r.min():.4f}")
    print(f"  Max:  {max_corr_at_r.max():.4f}")

    # ====================
    # COMPARISON PLOT
    # ====================
    print("\n" + "=" * 70)
    print("CREATING COMPARISON PLOT")
    print("=" * 70)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot 1: Fractal dimension
    if len(fractal_dims) > 0:
        axes[0].plot(positions_fd, fractal_dims, 'b-', linewidth=1.5, label='Fractal Dimension')
        axes[0].set_ylabel('Fractal Dimension')
        axes[0].set_title(f'Threshold-based Fractal Dimension (threshold ≥ {threshold}, window={window_size})')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    # Plot 2: Max local correlation (most sensitive to hotspots)
    axes[1].plot(positions_corr, max_corr_at_r, 'r-', linewidth=1.5,
                label=f'Max C_i(r) at r={actual_r:.3f}')
    axes[1].set_ylabel('Max Local Correlation')
    axes[1].set_title(f'Correlation-based: Max C_i in Window (hotspot strength)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Mean local correlation (average density)
    axes[2].plot(positions_corr, mean_corr_at_r, 'g-', linewidth=1.5,
                label=f'Mean C_i(r) at r={actual_r:.3f}')
    axes[2].set_xlabel('Position (sequence index)')
    axes[2].set_ylabel('Mean Local Correlation')
    axes[2].set_title(f'Correlation-based: Mean C_i in Window (average density)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('./output/method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Comparison plot saved as ./output/method_comparison.png")

    # Create scatter plot comparing the two methods
    if len(fractal_dims) > 0 and len(max_corr_at_r) > 0:
        print("\nComputing correlation between methods...")

        # Need to ensure arrays are same length
        min_len = min(len(fractal_dims), len(max_corr_at_r))
        fd_trimmed = np.array(fractal_dims[:min_len])
        mc_trimmed = max_corr_at_r[:min_len]

        # Compute correlation
        correlation = np.corrcoef(fd_trimmed, mc_trimmed)[0, 1]
        print(f"Pearson correlation: {correlation:.4f}")

        # Scatter plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(fd_trimmed, mc_trimmed, alpha=0.5, s=20)
        ax.set_xlabel('Fractal Dimension (threshold-based)')
        ax.set_ylabel(f'Max Local Correlation (r={actual_r:.3f})')
        ax.set_title(f'Method Comparison\nPearson r = {correlation:.4f}')
        ax.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(fd_trimmed, mc_trimmed, 1)
        p = np.poly1d(z)
        x_line = np.linspace(fd_trimmed.min(), fd_trimmed.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'y = {z[0]:.3f}x + {z[1]:.3f}')
        ax.legend()

        plt.tight_layout()
        plt.savefig('./output/method_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Scatter plot saved as ./output/method_scatter.png")

    # ====================
    # MULTI-SCALE COMPARISON
    # ====================
    print("\n" + "=" * 70)
    print("MULTI-SCALE ANALYSIS")
    print("=" * 70)

    # Show how correlation varies at different scales
    print("Comparing local correlation at multiple distance thresholds...")

    # Select a few interesting radii
    r_low = r_values[len(r_values)//4]      # Low distance = high similarity
    r_mid = r_values[len(r_values)//2]      # Medium
    r_high = r_values[3*len(r_values)//4]   # High distance = lower similarity

    idx_low = np.argmin(np.abs(r_values - r_low))
    idx_mid = np.argmin(np.abs(r_values - r_mid))
    idx_high = np.argmin(np.abs(r_values - r_high))

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    # Fractal dimension
    if len(fractal_dims) > 0:
        axes[0].plot(positions_fd, fractal_dims, 'b-', linewidth=1.5)
        axes[0].set_ylabel('Fractal Dimension')
        axes[0].set_title(f'Threshold-based: Fractal Dimension (threshold ≥ {threshold})')
        axes[0].grid(True, alpha=0.3)

    # Three different radii
    axes[1].plot(positions_corr, max_corr[:, idx_low], '-', linewidth=1.5,
                label=f'r = {r_values[idx_low]:.3f} (high similarity)')
    axes[1].set_ylabel('Max C_i(r)')
    axes[1].set_title('Correlation-based: High Similarity Scale')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(positions_corr, max_corr[:, idx_mid], '-', linewidth=1.5,
                label=f'r = {r_values[idx_mid]:.3f} (medium similarity)', color='orange')
    axes[2].set_ylabel('Max C_i(r)')
    axes[2].set_title('Correlation-based: Medium Similarity Scale')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(positions_corr, max_corr[:, idx_high], '-', linewidth=1.5,
                label=f'r = {r_values[idx_high]:.3f} (low similarity)', color='red')
    axes[3].set_xlabel('Position (sequence index)')
    axes[3].set_ylabel('Max C_i(r)')
    axes[3].set_title('Correlation-based: Low Similarity Scale')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    plt.tight_layout()
    plt.savefig('./output/multiscale_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Multi-scale comparison saved as ./output/multiscale_comparison.png")

    # ====================
    # SUMMARY
    # ====================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nThreshold-based method:")
    print("  + Familiar fractal dimension interpretation")
    print("  + Single summary statistic per window")
    print("  - Requires arbitrary threshold choice")
    print("  - Loses information from continuous distance values")
    print("  - Binary transformation can miss subtle patterns")

    print("\nCorrelation-based method:")
    print("  + Uses full distance information (no threshold)")
    print("  + Multi-scale analysis reveals patterns at different similarity levels")
    print("  + Local correlation directly measures cluster density")
    print("  + Max C_i detects hotspots of self-similarity")
    print("  - Less familiar interpretation than fractal dimension")
    print("  - More computationally intensive (multiple radii)")

    print("\nRecommendation:")
    print("  Use correlation-based method as primary analysis, especially")
    print("  when you don't know the right threshold a priori. The multi-scale")
    print("  information can reveal structure missed by any single threshold.")

    print(f"\nAll comparison plots saved in ./output/")


if __name__ == "__main__":
    main()
