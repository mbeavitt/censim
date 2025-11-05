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


def local_correlation_sum_vectorized(D, r_values):
    """
    Vectorized local correlation sum C_i(r) for multiple radii at once.

    Uses sort + binary search instead of full broadcast comparison.
    Complexity: O(N² log N + N × n_radii × log N) vs O(N² × n_radii)
    For typical case (N=500, n_radii=50): ~2.5M vs ~12.5M operations (5x faster)

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
    # searchsorted(sorted_row, r) returns count of elements < r
    counts = np.zeros((N, len(r_values)), dtype=np.int32)
    for i in range(N):
        counts[i] = np.searchsorted(D_sorted[i], r_values, side='left')

    return counts.astype(np.float64) / (N - 1)


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
        positions = [N // 2]
        # Use vectorized version - compute all radii at once
        local_c = local_correlation_sum_vectorized(D, r_values)  # (N, n_radii)
        mean_corr = np.mean(local_c, axis=0, keepdims=True)  # (1, n_radii)
        max_corr = np.max(local_c, axis=0, keepdims=True)  # (1, n_radii)
        return positions, mean_corr, max_corr

    # Pad the distance matrix with 1s (maximum distance) to allow windows at the edges
    pad_size = window_size // 2
    padded_D = np.pad(D, pad_size, mode='constant', constant_values=1.0)

    # Pre-allocate output arrays for better performance
    n_windows = N
    positions = np.arange(0, N, 1)
    mean_correlations = np.zeros((n_windows, n_radii))
    max_correlations = np.zeros((n_windows, n_radii))

    # Slide window with step size of 1, now covering positions 0 to N-1
    for i, pos in enumerate(positions):
        # Window in padded matrix coordinates
        start = pos
        end = pos + window_size

        # Extract window submatrix from padded matrix
        window = padded_D[start:end, start:end]

        # Compute local correlation for all radii at once (vectorized!)
        local_c = local_correlation_sum_vectorized(window, r_values)  # (window_size, n_radii)
        mean_correlations[i, :] = np.mean(local_c, axis=0)
        max_correlations[i, :] = np.max(local_c, axis=0)

    return positions, mean_correlations, max_correlations


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
