#!/usr/bin/env python3
"""
Correlation-based dimension analysis for sequence identity matrices.

This module implements local and global correlation sum methods for analyzing
self-similarity in sequence data without requiring arbitrary thresholds.
"""

import numpy as np
from numpy import log, log10
from scipy.stats import linregress
import matplotlib.pyplot as plt


def hamming_distance_matrix(identity_matrix):
    """
    Convert an identity matrix to a Hamming distance matrix.

    Args:
        identity_matrix: (N, N) matrix with values in [0, 1] representing identity

    Returns:
        distance_matrix: (N, N) matrix with values in [0, 1] representing distance
    """
    return 1.0 - identity_matrix


def correlation_sum_from_distance_matrix(D, r_values):
    """
    Global correlation sum C(r) for each r in r_values.

    Computes the fraction of all sequence pairs within distance r.

    Args:
        D: (N, N) symmetric distance matrix with zeros on diagonal
        r_values: 1D array of radii (distance thresholds)

    Returns:
        C: array of same length as r_values with correlation sums
    """
    N = D.shape[0]
    # Upper triangle distances (i < j) to avoid counting pairs twice
    iu = np.triu_indices(N, k=1)
    d = D[iu]
    r_values = np.asarray(r_values)
    C = np.array([(np.sum(d < r) * 2) / (N * (N - 1)) for r in r_values])
    return C


def local_correlation_sum(D, radius):
    """
    Local correlation sum C_i(r) for each sequence i.

    Computes the fraction of other sequences within distance r of each sequence.
    High values indicate sequences in dense clusters (hotspots of self-similarity).

    Args:
        D: (N, N) distance matrix
        radius: scalar radius (distance threshold)

    Returns:
        C_i: array of length N with local correlation sums
    """
    N = D.shape[0]
    # Count neighbors within radius, excluding self (diagonal)
    counts = np.sum((D < radius) & (~np.eye(N, dtype=bool)), axis=1)
    return counts / (N - 1)


def estimate_D2_from_C_r(r_values, C_values, fit_window=None, min_points=3):
    """
    Estimate correlation dimension D2 from log-log slope of C(r).

    In a self-similar system, C(r) ~ r^D2, so log(C) ~ D2 * log(r).

    Args:
        r_values: array of radii
        C_values: array of correlation sums
        fit_window: optional tuple (r_min, r_max) to restrict fitting range
        min_points: minimum number of points required for fit

    Returns:
        slope: estimated D2 (correlation dimension)
        intercept: y-intercept of log-log fit
        mask: boolean array indicating which points were used
    """
    r = np.asarray(r_values)
    C = np.asarray(C_values)
    mask = C > 0  # log(0) not allowed

    if fit_window is not None:
        rmin, rmax = fit_window
        mask &= (r >= rmin) & (r <= rmax)

    if np.sum(mask) < min_points:
        return np.nan, np.nan, mask

    lr = np.log(r[mask])
    lC = np.log(C[mask])
    slope, intercept, r_val, p_val, stderr = linregress(lr, lC)
    return slope, intercept, mask


def sliding_window_local_correlation(D, window_size, r_values):
    """
    Compute local correlation dimensions using a sliding window with zero-padding.

    For each window position, compute the mean and max local correlation sum
    across sequences in that window at each radius r. Uses zero-padding at edges
    to ensure full coverage from position 0 to N.

    Args:
        D: (N, N) distance matrix
        window_size: size of sliding window
        r_values: array of radii to compute local correlations at

    Returns:
        positions: array of window center positions (0 to N-1)
        mean_correlations: (n_windows, n_radii) array of mean local correlations
        max_correlations: (n_windows, n_radii) array of max local correlations
    """
    N = D.shape[0]
    r_values = np.asarray(r_values)
    n_radii = len(r_values)

    if window_size >= N:
        print("Window size >= matrix size, returning single window")
        positions = [N // 2]
        mean_corr = np.zeros((1, n_radii))
        max_corr = np.zeros((1, n_radii))
        for i, r in enumerate(r_values):
            local_c = local_correlation_sum(D, r)
            mean_corr[0, i] = np.mean(local_c)
            max_corr[0, i] = np.max(local_c)
        return positions, mean_corr, max_corr

    # Pad the distance matrix with 1s (maximum distance) to allow windows at the edges
    pad_size = window_size // 2
    padded_D = np.pad(D, pad_size, mode='constant', constant_values=1.0)

    positions = []
    mean_correlations = []
    max_correlations = []

    # Slide window with step size of 1, now covering positions 0 to N-1
    step = 1
    for pos in range(0, N, step):
        # Window in padded matrix coordinates
        start = pos
        end = pos + window_size
        positions.append(pos)

        # Extract window submatrix from padded matrix
        window = padded_D[start:end, start:end]

        # Compute local correlation at each radius
        mean_c = []
        max_c = []
        for r in r_values:
            local_c = local_correlation_sum(window, r)
            mean_c.append(np.mean(local_c))
            max_c.append(np.max(local_c))

        mean_correlations.append(mean_c)
        max_correlations.append(max_c)

        if len(positions) % 100 == 0:
            print(f"Progress: window {len(positions)}/{N}")

    return (np.array(positions),
            np.array(mean_correlations),
            np.array(max_correlations))


def plot_global_correlation_sum(r_values, C_values, D2=None, fit_mask=None,
                                 output_file='./output/global_correlation_sum.png'):
    """
    Plot global correlation sum C(r) vs r in both linear and log-log scales.

    Args:
        r_values: array of radii
        C_values: array of correlation sums
        D2: optional correlation dimension estimate to show fit line
        fit_mask: optional mask indicating which points were used for fit
        output_file: path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale
    ax1.plot(r_values, C_values, 'b-', linewidth=2)
    ax1.set_xlabel('Distance threshold r')
    ax1.set_ylabel('C(r)')
    ax1.set_title('Global Correlation Sum')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, 1.05)

    # Log-log scale
    mask = C_values > 0
    ax2.loglog(r_values[mask], C_values[mask], 'bo-', label='Data', markersize=4)

    if D2 is not None and fit_mask is not None:
        # Show fit line
        r_fit = r_values[fit_mask]
        if len(r_fit) > 0:
            # Reconstruct from slope
            slope, intercept, _ = estimate_D2_from_C_r(r_values, C_values,
                                                        fit_window=None,
                                                        min_points=3)
            C_fit = np.exp(intercept) * r_fit ** slope
            ax2.loglog(r_fit, C_fit, 'r--', linewidth=2,
                      label=f'Fit: D₂ = {D2:.3f}')

    ax2.set_xlabel('Distance threshold r')
    ax2.set_ylabel('C(r)')
    ax2.set_title('Log-Log Plot')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Global correlation plot saved as {output_file}")


def plot_local_correlation_map(D, r_values, positions=None,
                               output_file='./output/local_correlation_map.png'):
    """
    Create a heatmap showing local correlation C_i(r) for each sequence at multiple radii.

    Args:
        D: (N, N) distance matrix
        r_values: array of radii to compute local correlations
        positions: optional array of genomic positions for x-axis labels
        output_file: path to save figure
    """
    N = D.shape[0]
    r_values = np.asarray(r_values)

    # Compute local correlation for each sequence at each radius
    local_corr_matrix = np.zeros((N, len(r_values)))
    for i, r in enumerate(r_values):
        local_corr_matrix[:, i] = local_correlation_sum(D, r)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Heatmap
    im = ax1.imshow(local_corr_matrix.T, aspect='auto', cmap='viridis',
                    interpolation='nearest', origin='lower')
    ax1.set_xlabel('Sequence Index')
    ax1.set_ylabel('Distance threshold r')
    ax1.set_title('Local Correlation Sum C_i(r) - Hotspots of Self-Similarity')

    # Set y-tick labels to actual r values
    n_ticks = min(len(r_values), 10)
    tick_indices = np.linspace(0, len(r_values) - 1, n_ticks, dtype=int)
    ax1.set_yticks(tick_indices)
    ax1.set_yticklabels([f'{r_values[i]:.3f}' for i in tick_indices])

    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('C_i(r) - Fraction of neighbors within r')

    # Mean local correlation across all sequences (shows which r is meaningful)
    mean_local_corr = local_corr_matrix.mean(axis=0)
    ax2.plot(r_values, mean_local_corr, 'b-', linewidth=2)
    ax2.set_xlabel('Distance threshold r')
    ax2.set_ylabel('Mean C_i(r)')
    ax2.set_title('Average Local Correlation')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, r_values[-1])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Local correlation map saved as {output_file}")


def plot_sliding_window_correlation(positions, mean_correlations, max_correlations,
                                    r_values, selected_r_indices=None,
                                    output_file='./output/sliding_window_correlation.png'):
    """
    Plot sliding window local correlation results.

    Args:
        positions: array of window center positions
        mean_correlations: (n_windows, n_radii) array
        max_correlations: (n_windows, n_radii) array
        r_values: array of radii used
        selected_r_indices: optional list of indices into r_values to plot (plots all if None)
        output_file: path to save figure
    """
    if selected_r_indices is None:
        # Plot up to 5 radii
        n_radii_to_plot = min(5, len(r_values))
        selected_r_indices = np.linspace(0, len(r_values) - 1,
                                        n_radii_to_plot, dtype=int)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Mean local correlation
    for idx in selected_r_indices:
        ax1.plot(positions, mean_correlations[:, idx], '-',
                linewidth=1.5, label=f'r = {r_values[idx]:.3f}')
    ax1.set_ylabel('Mean C_i(r) in window')
    ax1.set_title('Sliding Window: Mean Local Correlation')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Max local correlation
    for idx in selected_r_indices:
        ax2.plot(positions, max_correlations[:, idx], '-',
                linewidth=1.5, label=f'r = {r_values[idx]:.3f}')
    ax2.set_xlabel('Position (sequence index)')
    ax2.set_ylabel('Max C_i(r) in window')
    ax2.set_title('Sliding Window: Max Local Correlation (Hotspot Strength)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sliding window correlation plot saved as {output_file}")


def find_hotspots(D, r, percentile=95):
    """
    Identify hotspot sequences with high local correlation.

    Args:
        D: (N, N) distance matrix
        r: distance threshold
        percentile: percentile threshold for defining hotspots

    Returns:
        hotspot_indices: array of sequence indices that are hotspots
        local_corr: full array of local correlation values
    """
    local_corr = local_correlation_sum(D, r)
    threshold = np.percentile(local_corr, percentile)
    hotspot_indices = np.where(local_corr >= threshold)[0]
    return hotspot_indices, local_corr


def choose_meaningful_radii(D, n_radii=10, r_min=None, r_max=None):
    """
    Choose a set of meaningful radii for analysis based on distance distribution.

    Args:
        D: (N, N) distance matrix
        n_radii: number of radii to return
        r_min: minimum radius (default: auto-detect from data)
        r_max: maximum radius (default: auto-detect from data)

    Returns:
        r_values: array of radii spanning the distance distribution
    """
    N = D.shape[0]
    # Get upper triangle distances (excluding diagonal)
    iu = np.triu_indices(N, k=1)
    distances = D[iu]

    # Determine min radius
    if r_min is None:
        min_nonzero = distances[distances > 0].min() if np.any(distances > 0) else 0.001
        r_min = min_nonzero

    # Determine max radius
    if r_max is None:
        # Use percentiles to span the distribution
        percentiles = np.linspace(1, 99, n_radii)
        r_values = np.percentile(distances, percentiles)
    else:
        # Use linear spacing in specified range
        r_values = np.linspace(r_min, r_max, n_radii)

    # Ensure first value is r_min
    r_values = np.unique(np.concatenate([[r_min], r_values]))

    # Filter to stay within bounds
    if r_max is not None:
        r_values = r_values[r_values <= r_max]

    # Make sure we have at least n_radii values
    if len(r_values) < n_radii and r_max is not None:
        r_values = np.linspace(r_min, r_max, n_radii)

    return r_values


def analyze_distance_distribution(D, output_file='./output/distance_distribution.png'):
    """
    Visualize the distance distribution in the matrix.

    Args:
        D: (N, N) distance matrix
        output_file: path to save figure
    """
    N = D.shape[0]
    iu = np.triu_indices(N, k=1)
    distances = D[iu]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1.hist(distances, bins=100, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Hamming Distance')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Pairwise Distances')
    ax1.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_dist = np.sort(distances)
    cumulative = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
    ax2.plot(sorted_dist, cumulative, 'b-', linewidth=2)
    ax2.set_xlabel('Hamming Distance')
    ax2.set_ylabel('Cumulative Fraction')
    ax2.set_title('Cumulative Distribution of Distances')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Distance distribution plot saved as {output_file}")

    # Print statistics
    print("\nDistance distribution statistics:")
    print(f"  Min:    {distances.min():.4f}")
    print(f"  25th:   {np.percentile(distances, 25):.4f}")
    print(f"  Median: {np.median(distances):.4f}")
    print(f"  75th:   {np.percentile(distances, 75):.4f}")
    print(f"  Max:    {distances.max():.4f}")
    print(f"  Mean:   {distances.mean():.4f}")
    print(f"  Std:    {distances.std():.4f}")
