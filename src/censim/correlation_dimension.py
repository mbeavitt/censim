#!/usr/bin/env python3
"""
Correlation-based dimension analysis for sequence identity matrices.

This module implements correlation sum methods for analyzing self-similarity
in sequence data without requiring arbitrary thresholds.
"""

import numpy as np
from scipy.stats import linregress


def hamming_distance_matrix(identity_matrix):
    """
    Convert an identity matrix to a Hamming distance matrix.

    Args:
        identity_matrix: (N, N) matrix with values in [0, 1] representing identity

    Returns:
        distance_matrix: (N, N) matrix with values in [0, 1] representing distance
    """
    return 1.0 - identity_matrix


def correlation_sum_from_distance_matrix(D, r_values, triu_mask=None):
    """
    Global correlation sum C(r) for each r in r_values.

    Computes the fraction of all sequence pairs within distance r.
    Optimized with sort + binary search instead of loop.

    Args:
        D: (N, N) symmetric distance matrix with zeros on diagonal
        r_values: 1D array of radii (distance thresholds)
        triu_mask: Optional pre-computed upper triangle mask for performance

    Returns:
        C: array of same length as r_values with correlation sums
    """
    N = D.shape[0]
    # Upper triangle distances (i < j) to avoid counting pairs twice
    # Use boolean mask instead of fancy indexing for better performance
    if triu_mask is None:
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    else:
        mask = triu_mask
    d = D[mask]

    # Sort distances once: O(n_pairs log n_pairs)
    d_sorted = np.sort(d)

    # Binary search for each radius: O(n_radii × log n_pairs)
    # searchsorted returns count of elements < r
    r_values = np.asarray(r_values)
    counts = np.searchsorted(d_sorted, r_values, side='left')

    # Convert counts to correlation sums
    C = (counts * 2.0) / (N * (N - 1))
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


def local_correlation_sum_naive(D, r_values):
    """
    Naive loop version - calls local_correlation_sum for each radius.

    Simple to understand but slow: repeats work for each radius.
    Complexity: O(N² × n_radii)
    Creates np.eye(N) for each radius call.

    Args:
        D: (N, N) distance matrix
        r_values: array of radii (distance thresholds)

    Returns:
        C_i: (N, len(r_values)) array where C_i[i, j] is the local correlation
             sum for sequence i at radius r_values[j]
    """
    N = D.shape[0]
    results = []

    # Call local_correlation_sum once per radius
    for r in r_values:
        result = local_correlation_sum(D, r)
        results.append(result)

    # Stack results: (n_radii, N) -> transpose to (N, n_radii)
    return np.array(results).T


def local_correlation_sum_broadcast(D, r_values):
    """
    Broadcast vectorized version - creates (N, N, n_radii) array.

    Simple but memory-intensive: creates full boolean array for all comparisons.
    Complexity: O(N² × n_radii)
    Memory: N × N × n_radii booleans (e.g., 12.5MB for N=500, n_radii=50)

    Args:
        D: (N, N) distance matrix
        r_values: array of radii (distance thresholds)

    Returns:
        C_i: (N, len(r_values)) array where C_i[i, j] is the local correlation
             sum for sequence i at radius r_values[j]
    """
    N = D.shape[0]

    # Pre-compute diagonal mask once
    D_masked = D.copy()
    np.fill_diagonal(D_masked, np.inf)

    # Vectorized comparison: (N, N, n_radii) broadcast
    counts = np.sum(D_masked[:, :, np.newaxis] < r_values, axis=1)

    return counts / (N - 1)


def local_correlation_sum_sorted(D, r_values):
    """
    Sort + binary search version.

    Sorts each row once, then uses binary search for each radius.
    Complexity: O(N² log N + N × n_radii × log N)
    Memory: N × N × 2 (original + sorted, no n_radii term)

    For N=500, n_radii=50: ~2.5M vs ~12.5M operations (5x fewer than broadcast)

    Args:
        D: (N, N) distance matrix
        r_values: array of radii (distance thresholds)

    Returns:
        C_i: (N, len(r_values)) array where C_i[i, j] is the local correlation
             sum for sequence i at radius r_values[j]
    """
    N = D.shape[0]

    # Mask diagonal with infinity
    D_masked = D.copy()
    np.fill_diagonal(D_masked, np.inf)

    # Sort each row once: O(N² log N)
    D_sorted = np.sort(D_masked, axis=1)

    # Binary search for each radius: O(N × n_radii × log N)
    counts = np.zeros((N, len(r_values)), dtype=np.int32)
    for i in range(N):
        counts[i] = np.searchsorted(D_sorted[i], r_values, side='left')

    return counts.astype(np.float64) / (N - 1)


# Active implementation (change this to switch versions)
local_correlation_sum_vectorized = local_correlation_sum_sorted


def estimate_D2_from_C_r_batch(r_values, C_matrix, fit_window=None, min_points=3, log_r=None):
    """
    Fully vectorized batch version of estimate_D2_from_C_r for multiple windows.

    Processes all windows at once using vectorized operations. Assumes all windows
    have identical valid point masks (which is typical when using same r_values).

    Args:
        r_values: array of radii (n_radii,)
        C_matrix: (n_windows, n_radii) array of correlation sums
        fit_window: optional tuple (r_min, r_max) to restrict fitting range
        min_points: minimum number of points required for fit
        log_r: optional pre-computed log(r_values) for performance

    Returns:
        slopes: (n_windows,) array of estimated D2 values (0.0 for invalid fits)
    """
    r = np.asarray(r_values)
    C = np.asarray(C_matrix)  # (n_windows, n_radii)

    # Compute mask: assume all windows have same mask (typical case)
    # Use first window to determine mask
    mask = C[0] > 0

    if fit_window is not None:
        rmin, rmax = fit_window
        mask &= (r >= rmin) & (r <= rmax)

    if np.sum(mask) < min_points:
        # If first window is invalid, return zeros for all
        return np.zeros(C.shape[0])

    # Pre-compute log(r) for masked values
    if log_r is None:
        log_r = np.log(r)
    lr = log_r[mask]  # (n_valid,)

    # Compute log(C) for all windows at once: (n_windows, n_valid)
    lC = np.log(C[:, mask])

    # Vectorized least-squares regression for all windows
    # slope = cov(x,y) / var(x), intercept = mean(y) - slope * mean(x)
    lr_mean = np.mean(lr)  # scalar
    lC_mean = np.mean(lC, axis=1)  # (n_windows,)

    # Compute slopes for all windows at once
    # numerator: sum((lr - lr_mean) * (lC - lC_mean)) for each window
    # denominator: sum((lr - lr_mean)**2) - same for all windows
    lr_centered = lr - lr_mean  # (n_valid,)
    lC_centered = lC - lC_mean[:, np.newaxis]  # (n_windows, n_valid)

    numerator = np.sum(lr_centered * lC_centered, axis=1)  # (n_windows,)
    denominator = np.sum(lr_centered**2)  # scalar

    slopes = numerator / denominator  # (n_windows,)

    # Replace any NaN values with 0.0
    slopes = np.where(np.isnan(slopes), 0.0, slopes)

    return slopes


def estimate_D2_from_C_r(r_values, C_values, fit_window=None, min_points=3, log_r=None):
    """
    Estimate correlation dimension D2 from log-log slope of C(r).

    In a self-similar system, C(r) ~ r^D2, so log(C) ~ D2 * log(r).

    Args:
        r_values: array of radii
        C_values: array of correlation sums
        fit_window: optional tuple (r_min, r_max) to restrict fitting range
        min_points: minimum number of points required for fit
        log_r: optional pre-computed log(r_values) for performance

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

    # Use pre-computed log_r if provided (faster when called repeatedly)
    if log_r is None:
        lr = np.log(r[mask])
    else:
        lr = log_r[mask]

    lC = np.log(C[mask])

    # Fast least-squares instead of full linregress (we don't need r_val, p_val, stderr)
    # slope = cov(x,y) / var(x), intercept = mean(y) - slope * mean(x)
    lr_mean = np.mean(lr)
    lC_mean = np.mean(lC)
    slope = np.sum((lr - lr_mean) * (lC - lC_mean)) / np.sum((lr - lr_mean)**2)
    intercept = lC_mean - slope * lr_mean

    return slope, intercept, mask


def sliding_window_local_correlation(D, window_size, r_values):
    """
    Compute local correlation dimensions using a sliding window with zero-padding.

    For each window position, compute the mean local correlation sum
    across sequences in that window at each radius r. Uses zero-padding at edges
    to ensure full coverage from position 0 to N.

    Args:
        D: (N, N) distance matrix
        window_size: size of sliding window
        r_values: array of radii to compute local correlations at

    Returns:
        positions: array of window center positions (0 to N-1)
        mean_correlations: (n_windows, n_radii) array of mean local correlations
    """
    N = D.shape[0]
    r_values = np.asarray(r_values)
    n_radii = len(r_values)

    if window_size >= N:
        positions = [N // 2]
        # Use direct global correlation sum (mean) - much faster!
        mean_corr = correlation_sum_from_distance_matrix(D, r_values).reshape(1, -1)  # (1, n_radii)
        return positions, mean_corr

    # Pad the distance matrix with 1s (maximum distance) to allow windows at the edges
    pad_size = window_size // 2
    padded_D = np.pad(D, pad_size, mode='constant', constant_values=1.0)

    # Pre-allocate output arrays for better performance
    n_windows = N
    positions = np.arange(0, N, 1)
    mean_correlations = np.zeros((n_windows, n_radii))

    # Pre-compute upper triangle mask once for all windows (saves 702 mask creations)
    triu_mask = np.triu(np.ones((window_size, window_size), dtype=bool), k=1)

    # Slide window with step size of 1, now covering positions 0 to N-1
    for i, pos in enumerate(positions):
        # Window in padded matrix coordinates
        start = pos
        end = pos + window_size

        # Extract window submatrix from padded matrix
        window = padded_D[start:end, start:end]

        # Compute mean correlation directly using global correlation sum (much faster!)
        mean_correlations[i, :] = correlation_sum_from_distance_matrix(window, r_values, triu_mask)

    return positions, mean_correlations


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
