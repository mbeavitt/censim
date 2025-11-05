#!/usr/bin/env python3
"""
Correlation-based dimension analysis for sequence identity matrices.

This module implements correlation sum methods for analyzing self-similarity
in sequence data without requiring arbitrary thresholds.
"""

import numpy as np
from scipy.stats import linregress
from numba import njit


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


@njit(fastmath=True)
def _extract_and_sort_windows_numba_serial(padded_D, window_size, n_windows):
    """
    Extract upper triangle distances from all windows and sort them serially (single-threaded).

    Args:
        padded_D: Padded distance matrix
        window_size: Size of sliding window
        n_windows: Number of windows

    Returns:
        sorted_distances: (n_windows, n_pairs) array of sorted distances
    """
    # Number of pairs in upper triangle: window_size * (window_size - 1) / 2
    n_pairs = (window_size * (window_size - 1)) // 2
    sorted_distances = np.zeros((n_windows, n_pairs), dtype=np.float32)

    # Process all windows serially
    for i in range(n_windows):
        # Extract window
        start = i
        end = i + window_size

        # Extract upper triangle distances manually (numba doesn't support boolean indexing)
        idx = 0
        for row in range(window_size):
            for col in range(row + 1, window_size):
                sorted_distances[i, idx] = padded_D[start + row, start + col]
                idx += 1

        # Sort distances for this window (in-place)
        sorted_distances[i, :] = np.sort(sorted_distances[i, :])

    return sorted_distances


@njit(fastmath=True)
def _compute_correlation_sums_batch_serial(sorted_distances, r_values, window_size):
    """
    Compute correlation sums for all windows using pre-sorted distances (serial).

    Args:
        sorted_distances: (n_windows, n_pairs) array of sorted distances
        r_values: Array of radii
        window_size: Size of sliding window

    Returns:
        correlations: (n_windows, n_radii) array of correlation sums
    """
    n_windows = sorted_distances.shape[0]
    n_radii = len(r_values)
    correlations = np.zeros((n_windows, n_radii), dtype=np.float32)

    # Normalization factor
    norm = 2.0 / (window_size * (window_size - 1))

    # Process all windows serially
    for i in range(n_windows):
        for j in range(n_radii):
            # Binary search: count how many distances < r_values[j]
            count = np.searchsorted(sorted_distances[i, :], r_values[j])
            correlations[i, j] = count * norm

    return correlations


def sliding_window_local_correlation(D, window_size, r_values):
    """
    Compute local correlation dimensions using a sliding window with zero-padding.

    For each window position, compute the mean local correlation sum
    across sequences in that window at each radius r. Uses zero-padding at edges
    to ensure full coverage from position 0 to N.

    Optimized with numba processing: extracts and sorts all windows, then computes
    correlation sums using JIT-compiled functions.

    Args:
        D: (N, N) distance matrix
        window_size: size of sliding window
        r_values: array of radii to compute local correlations at

    Returns:
        positions: array of window center positions (0 to N-1)
        mean_correlations: (n_windows, n_radii) array of mean local correlations
    """
    N = D.shape[0]
    r_values = np.asarray(r_values, dtype=np.float32)
    n_radii = len(r_values)

    if window_size >= N:
        positions = [N // 2]
        # Use direct global correlation sum (mean) - much faster!
        mean_corr = correlation_sum_from_distance_matrix(D, r_values).reshape(1, -1)  # (1, n_radii)
        return positions, mean_corr

    # Pad the distance matrix with 1s (maximum distance) to allow windows at the edges
    pad_size = window_size // 2
    padded_D = np.pad(D, pad_size, mode='constant', constant_values=1.0).astype(np.float32)

    # Pre-allocate output arrays for better performance
    n_windows = N
    positions = np.arange(0, N, 1)

    # Extract and sort all windows (numba optimized)
    sorted_distances = _extract_and_sort_windows_numba_serial(padded_D, window_size, n_windows)

    # Compute correlation sums for all windows (numba optimized)
    mean_correlations = _compute_correlation_sums_batch_serial(sorted_distances, r_values, window_size)

    return positions, mean_correlations
