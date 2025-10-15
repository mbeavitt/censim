#!/usr/bin/env python3
"""
Correlation-based analysis of sequence identity without thresholding.

This script uses local correlation sums instead of arbitrary thresholds
to identify regions of high self-similarity (hotspots) in sliding windows.
"""

import numpy as np
import sys
import time
from pathlib import Path

# Import identity matrix computation from original script
from identity import all_vs_all_identity_scipy

# Import correlation-based analysis functions
from correlation_dimension import (
    hamming_distance_matrix,
    correlation_sum_from_distance_matrix,
    local_correlation_sum,
    estimate_D2_from_C_r,
    sliding_window_local_correlation,
    plot_global_correlation_sum,
    plot_local_correlation_map,
    plot_sliding_window_correlation,
    find_hotspots,
    choose_meaningful_radii,
    analyze_distance_distribution
)


def main():
    """Run correlation-based identity analysis."""

    # Configuration
    if len(sys.argv) > 1:
        window_size = int(sys.argv[1])
    else:
        window_size = 100

    if len(sys.argv) > 2:
        data_file = sys.argv[2]
    else:
        data_file = "./data/2191000generation.out.fa"

    print("=" * 70)
    print("CORRELATION-BASED SEQUENCE IDENTITY ANALYSIS")
    print("=" * 70)
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
    print("\n" + "=" * 70)
    print("STEP 1: Computing identity matrix")
    print("=" * 70)
    start = time.time()
    identity_matrix = all_vs_all_identity_scipy(repeats, scale_factor=30, max_exact_size=1000)
    elapsed = time.time() - start
    print(f"Computation took: {elapsed:.2f} seconds")
    print(f"Identity matrix shape: {identity_matrix.shape}")
    print(f"Mean identity: {identity_matrix.mean():.3f}")

    # Convert to distance matrix
    print("\nConverting to Hamming distance matrix...")
    D = hamming_distance_matrix(identity_matrix)
    print(f"Mean distance: {D.mean():.3f}")

    # Analyze distance distribution
    print("\n" + "=" * 70)
    print("STEP 2: Analyzing distance distribution")
    print("=" * 70)
    analyze_distance_distribution(D)

    # Choose meaningful radii based on distance distribution
    print("\nChoosing meaningful radii for analysis...")
    r_values = choose_meaningful_radii(D, n_radii=15)
    print(f"Selected {len(r_values)} radii:")
    for i, r in enumerate(r_values):
        print(f"  r[{i}] = {r:.4f}")

    # Global correlation sum
    print("\n" + "=" * 70)
    print("STEP 3: Computing global correlation sum")
    print("=" * 70)
    C_global = correlation_sum_from_distance_matrix(D, r_values)
    print("C(r) computed for all radii")

    # Estimate correlation dimension
    print("\nEstimating correlation dimension D2...")
    D2, intercept, fit_mask = estimate_D2_from_C_r(r_values, C_global,
                                                    fit_window=None, min_points=3)
    if not np.isnan(D2):
        print(f"Estimated D2 (correlation dimension): {D2:.4f}")
        print(f"  (Note: biological sequences may not show clean power-law scaling)")
    else:
        print("Could not estimate D2 (insufficient scaling region)")

    # Plot global correlation sum
    plot_global_correlation_sum(r_values, C_global, D2, fit_mask)

    # Local correlation map
    print("\n" + "=" * 70)
    print("STEP 4: Computing local correlation map")
    print("=" * 70)
    print("Computing C_i(r) for all sequences at all radii...")
    plot_local_correlation_map(D, r_values)

    # Find hotspots at a biologically meaningful radius
    print("\n" + "=" * 70)
    print("STEP 5: Finding hotspot sequences")
    print("=" * 70)

    # Use a radius around 5% distance (95% similarity) as example
    # User can adjust based on their biological question
    r_hotspot = 0.05
    print(f"Using r = {r_hotspot:.3f} for hotspot detection")

    hotspot_indices, local_corr = find_hotspots(D, r_hotspot, percentile=95)
    print(f"Found {len(hotspot_indices)} hotspot sequences (top 5%)")
    print(f"Hotspot local correlation range: [{local_corr[hotspot_indices].min():.3f}, "
          f"{local_corr[hotspot_indices].max():.3f}]")

    if len(hotspot_indices) > 0:
        print(f"\nTop 10 hotspot sequences:")
        top_10_idx = np.argsort(local_corr)[-10:][::-1]
        for idx in top_10_idx:
            print(f"  Sequence {idx:5d}: C_i = {local_corr[idx]:.3f}")

    # Sliding window analysis
    print("\n" + "=" * 70)
    print("STEP 6: Sliding window local correlation analysis")
    print("=" * 70)
    print(f"Window size: {window_size}")
    print(f"Computing local correlation for each window...")

    # Use a subset of radii for sliding window to save time
    r_values_window = r_values[::2]  # Every other radius
    print(f"Using {len(r_values_window)} radii for sliding window")

    start = time.time()
    positions, mean_corr, max_corr = sliding_window_local_correlation(
        D, window_size, r_values_window
    )
    elapsed = time.time() - start
    print(f"\nSliding window computation took: {elapsed:.2f} seconds")
    print(f"Computed {len(positions)} windows")

    # Plot sliding window results
    print("\nGenerating sliding window plots...")
    plot_sliding_window_correlation(positions, mean_corr, max_corr,
                                   r_values_window, selected_r_indices=None)

    # Summary statistics for sliding window at a single meaningful radius
    print("\n" + "=" * 70)
    print("STEP 7: Summary statistics")
    print("=" * 70)

    # Find the index closest to r_hotspot
    r_idx = np.argmin(np.abs(r_values_window - r_hotspot))
    actual_r = r_values_window[r_idx]
    print(f"\nSummary for r ≈ {r_hotspot:.3f} (actual: {actual_r:.3f}):")

    max_corr_at_r = max_corr[:, r_idx]
    mean_corr_at_r = mean_corr[:, r_idx]

    print(f"\nMax local correlation in windows:")
    print(f"  Mean:   {max_corr_at_r.mean():.3f}")
    print(f"  Std:    {max_corr_at_r.std():.3f}")
    print(f"  Min:    {max_corr_at_r.min():.3f}")
    print(f"  Max:    {max_corr_at_r.max():.3f}")

    print(f"\nMean local correlation in windows:")
    print(f"  Mean:   {mean_corr_at_r.mean():.3f}")
    print(f"  Std:    {mean_corr_at_r.std():.3f}")
    print(f"  Min:    {mean_corr_at_r.min():.3f}")
    print(f"  Max:    {mean_corr_at_r.max():.3f}")

    # Identify windows with highest self-similarity
    print(f"\nTop 10 windows by max local correlation:")
    top_window_idx = np.argsort(max_corr_at_r)[-10:][::-1]
    for idx in top_window_idx:
        pos = positions[idx]
        max_c = max_corr_at_r[idx]
        mean_c = mean_corr_at_r[idx]
        print(f"  Position {pos:5d}: max C_i = {max_c:.3f}, mean C_i = {mean_c:.3f}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  - ./output/distance_distribution.png")
    print("  - ./output/global_correlation_sum.png")
    print("  - ./output/local_correlation_map.png")
    print("  - ./output/sliding_window_correlation.png")
    print("\nInterpretation:")
    print("  • High max C_i in a window = hotspot of self-similarity")
    print("  • Compare patterns to your original threshold-based fractal dimension")
    print("  • Peaks in different radii reveal similarity at different scales")


if __name__ == "__main__":
    main()
