#!/usr/bin/env python3
"""
Compare Fractal Dimension vs Correlation Dimension using sliding window analysis.

This script:
1. Loads sequence data from generation 2 million
2. Computes all-vs-all distance matrix using scipy
3. Performs 100-repeat sliding window analysis for both:
   - Fractal dimension (3D box counting)
   - Correlation dimension (D2)
4. Plots both dimensions on the same line graph for comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import rotate
import time
import sys
import os
import lzma  # Faster than zlib for large windows

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from censim.correlation_dimension import (
    hamming_distance_matrix,
    correlation_sum_from_distance_matrix,
    estimate_D2_from_C_r_batch,
)
from censim.identity import all_vs_all_identity_numba


def all_vs_all_identity_scipy(repeats, max_exact_size=1000, scale_factor=30):
    """
    Compute all vs all identity matrix using optimized numba (46x faster than scipy).

    Args:
        repeats: list of sequences (strings)
        max_exact_size: array size threshold for exact computation
        scale_factor: controls subsampling rate for large arrays

    Returns:
        tuple of (identity_matrix, subsample_every, subset_repeats)
        - identity_matrix: numpy array of shape (n_repeats, n_repeats) with pairwise identities
        - subsample_every: subsampling interval (1 = no subsampling)
        - subset_repeats: the subsampled repeat sequences
    """
    n_repeats = len(repeats)

    # Calculate adaptive subsampling
    if n_repeats <= max_exact_size:
        subsample_every = 1
        print(f"Computing exact identity ({n_repeats} sequences)...")
    else:
        import math
        subsample_every = max(1, int(math.sqrt(n_repeats / scale_factor)))
        subset_size = n_repeats // subsample_every
        print(f"Computing subsampled identity (every {subsample_every}th sequence, {subset_size}/{n_repeats} total)...")

    if subsample_every == 1:
        # Exact computation using optimized numba
        identity_matrix = all_vs_all_identity_numba(repeats, max_exact_size=max_exact_size, scale_factor=scale_factor)
        subset_repeats = repeats  # No subsampling
    else:
        # Subsampled computation
        subset_indices = np.arange(0, n_repeats, subsample_every)
        subset_repeats = [repeats[i] for i in subset_indices]
        identity_matrix = all_vs_all_identity_numba(subset_repeats, max_exact_size=max_exact_size, scale_factor=scale_factor)

    return identity_matrix, subsample_every, subset_repeats


def box_count_2d(binary_matrix):
    """
    Perform 2D box counting on a binary matrix.

    Args:
        binary_matrix: 2D binary array (0s and 1s)

    Returns:
        tuple of (scales, counts)
    """
    n = binary_matrix.shape[0]

    # Box sizes to test (powers of 2)
    max_box_size = n // 2
    box_sizes = []

    size = 1
    while size <= max_box_size:
        box_sizes.append(size)
        size *= 2

    # Add intermediate sizes for better fitting
    for i in range(len(box_sizes) - 1):
        mid = (box_sizes[i] + box_sizes[i+1]) // 2
        if mid not in box_sizes and mid > box_sizes[i]:
            box_sizes.append(mid)

    box_sizes = sorted(box_sizes)

    counts = []
    scales = []

    for box_size in box_sizes:
        count = 0
        # Iterate over boxes
        for i in range(0, n, box_size):
            for j in range(0, n, box_size):
                # Extract box
                box = binary_matrix[i:min(i+box_size, n), j:min(j+box_size, n)]
                # Count if box contains any 1s
                if np.any(box):
                    count += 1

        counts.append(count)
        scales.append(1.0 / box_size)

    return scales, counts


def calculate_fractal_dimension(scales, counts):
    """
    Calculate fractal dimension from box counting results.

    Args:
        scales: list of scales (1/box_size)
        counts: list of box counts

    Returns:
        fractal_dimension value
    """
    if len(scales) < 2:
        return np.nan

    log_scales = np.log(scales)
    log_counts = np.log(counts)

    # Linear regression in log-log space
    coeffs = np.polyfit(log_scales, log_counts, 1)
    fractal_dimension = coeffs[0]

    return fractal_dimension


def sliding_window_fractal_dimension(identity_matrix, window_size=100, threshold=0.92):
    """
    Calculate fractal dimension using sliding window on identity matrix with 2D box counting.

    Args:
        identity_matrix: (N, N) identity matrix
        window_size: size of sliding window
        threshold: threshold for binarizing the identity matrix (default: 0.92)

    Returns:
        positions: center positions of windows
        fractal_dimensions: fractal dimension at each window position
    """
    N = identity_matrix.shape[0]

    if window_size >= N:
        # Single window case
        positions = [N // 2]
        binary_matrix = (identity_matrix > threshold).astype(int)
        scales, counts = box_count_2d(binary_matrix)
        fd = calculate_fractal_dimension(scales, counts)
        return positions, [fd]

    positions = []
    fractal_dimensions = []

    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        # Extract window submatrix
        window = identity_matrix[i:i+window_size, i:i+window_size]

        # Binarize using threshold
        binary_window = (window > threshold).astype(int)

        # Calculate fractal dimension for this window using 2D box counting
        try:
            scales, counts = box_count_2d(binary_window)
            fd = calculate_fractal_dimension(scales, counts)
            fractal_dimensions.append(fd)
        except Exception as e:
            print(f"Warning: Failed to compute fractal dimension at position {center}: {e}")
            fractal_dimensions.append(np.nan)

    return positions, fractal_dimensions


def sliding_window_correlation_dimension(distance_matrix, window_size=100, r_min=0.01, r_max=0.5, n_radii=50):
    """
    Calculate correlation dimension using sliding window on distance matrix.

    Args:
        distance_matrix: (N, N) distance matrix
        window_size: size of sliding window
        r_min: minimum radius for correlation sum
        r_max: maximum radius for correlation sum
        n_radii: number of radii to test

    Returns:
        positions: center positions of windows
        d2_values: correlation dimension at each window position
    """
    N = distance_matrix.shape[0]
    r_values = np.linspace(r_min, r_max, n_radii)

    if window_size >= N:
        # Single window case
        positions = [N // 2]
        C_values = correlation_sum_from_distance_matrix(distance_matrix, r_values)
        d2 = estimate_D2_from_C_r_batch(r_values, C_values.reshape(1, -1))[0]
        return positions, [d2]

    positions = []
    all_C_values = []

    # Pre-compute upper triangle mask for windows
    triu_mask = np.triu(np.ones((window_size, window_size), dtype=bool), k=1)

    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        # Extract window submatrix
        window = distance_matrix[i:i+window_size, i:i+window_size]

        # Compute correlation sum for this window
        C_values = correlation_sum_from_distance_matrix(window, r_values, triu_mask)
        all_C_values.append(C_values)

    # Batch compute D2 for all windows
    C_matrix = np.array(all_C_values)  # (n_windows, n_radii)
    d2_values = estimate_D2_from_C_r_batch(r_values, C_matrix)

    return positions, d2_values.tolist()

def calculate_kmer_entropy(sequence_string, k=4):
    """
    Calculate sequence complexity using k-mer diversity (optimized with numpy).

    Uses k=4 (tetramers) to measure sequence diversity.
    Higher diversity = more complex/random
    Lower diversity = more repetitive/structured

    Args:
        sequence_string: string sequence
        k: k-mer size (default: 4)

    Returns:
        complexity: k-mer diversity score (higher = more complex)
    """
    if len(sequence_string) < k:
        return 0.0

    # Extract non-overlapping k-mers (side-by-side windows)
    n_kmers = len(sequence_string) // k  # Integer division for non-overlapping

    # Use numpy for fast counting
    # Convert to array of k-mer hashes for fast unique counting
    kmer_list = [sequence_string[i*k:(i+1)*k] for i in range(n_kmers)]
    unique_kmers, counts = np.unique(kmer_list, return_counts=True)

    # Calculate Shannon entropy using vectorized operations
    total = counts.sum()
    probs = counts / total
    entropy = -np.sum(probs * np.log2(probs))

    # Normalize by maximum possible entropy for k-mers
    # Max entropy = log2(4^k) for DNA with 4 bases
    max_entropy = 2 * k  # log2(4^k) = k * log2(4) = k * 2
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return normalized_entropy


def calculate_lzma_compressibility(sequence_string):
    """
    Calculate compressibility using LZMA compression.

    Args:
        sequence_string: string sequence

    Returns:
        compressibility: compression ratio (0 to 1, lower = more compressible)
    """
    original_size = len(sequence_string)
    compressed_size = len(lzma.compress(sequence_string.encode(), preset=1))
    return compressed_size / original_size


def sliding_window_complexity_from_repeats(repeats, window_size=100, method='kmer'):
    """
    Calculate complexity using sliding window on subsampled repeat array.

    Args:
        repeats: list of repeat sequences (already subsampled)
        window_size: number of repeats per window
        method: 'kmer' or 'lzma'

    Returns:
        positions: center positions of windows (in repeat indices)
        complexity_values: complexity at each window position
    """
    N = len(repeats)

    if window_size >= N:
        # Single window case
        positions = [N // 2]
        window_seq = ''.join(repeats)
        if method == 'kmer':
            comp = calculate_kmer_entropy(window_seq, k=4)
        else:  # lzma
            comp = calculate_lzma_compressibility(window_seq)
        return positions, [comp]

    positions = []
    complexity_values = []

    # Slide window across repeats
    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        # Extract window of repeats and concatenate
        window_repeats = repeats[i:i+window_size]
        window_seq = ''.join(window_repeats)

        # Calculate complexity for this window
        if method == 'kmer':
            comp = calculate_kmer_entropy(window_seq, k=4)
        else:  # lzma
            comp = calculate_lzma_compressibility(window_seq)
        complexity_values.append(comp)

    return positions, complexity_values


def main():
    """Main function to compare fractal and correlation dimensions."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare complexity metrics: Fractal Dimension, Correlation Dimension, k-mer Entropy, and LZMA Compressibility'
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        default="./data/2191000generation.out.fa",
        help='Input FASTA file (default: ./data/2191000generation.out.fa)'
    )

    parser.add_argument(
        '-w', '--window-size',
        type=int,
        default=100,
        help='Window size in repeats (default: 100)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.92,
        help='Threshold for fractal dimension binarization (default: 0.92)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./output/fractal_vs_correlation_comparison.png',
        help='Output plot filename (default: ./output/fractal_vs_correlation_comparison.png)'
    )

    args = parser.parse_args()

    # Configuration
    input_file = args.input
    repeat_len = 178
    window_size = args.window_size
    threshold = args.threshold

    print("="*80)
    print("Complexity Metrics Comparison")
    print("="*80)
    print(f"Input file: {input_file}")
    print(f"Window size: {window_size} repeats")
    print(f"Fractal dimension threshold: {threshold}")

    # Load data
    print(f"\n1. Loading sequence data from {input_file}...")
    with open(input_file, "r") as file:
        contents = file.read().strip()

    n_repeats = int(len(contents) / repeat_len)
    repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]

    print(f"   Loaded {n_repeats} repeats of length {repeat_len}bp")
    print(f"   Window size: {window_size} repeats")

    # Compute identity matrix (shared cost for FD and CD)
    print(f"\n2. Computing all-vs-all identity matrix (pdist - shared by FD & CD)...")
    start = time.time()
    identity_matrix, subsample_every, subsampled_repeats = all_vs_all_identity_scipy(repeats, scale_factor=30, max_exact_size=1000)
    elapsed_pdist = time.time() - start
    print(f"   Computation took: {elapsed_pdist:.2f} seconds")
    print(f"   Matrix shape: {identity_matrix.shape}")
    print(f"   Mean identity: {identity_matrix.mean():.4f}")
    print(f"   Subsampling interval: {subsample_every} (scale factor for positions)")
    print(f"   Subsampled repeats: {len(subsampled_repeats)} repeats")

    # Convert to distance matrix for correlation dimension
    distance_matrix = hamming_distance_matrix(identity_matrix)
    print(f"   Mean distance: {distance_matrix.mean():.4f}")

    # Storage for computed metrics
    computed_metrics = {}

    # Calculate Fractal Dimension with sliding window
    print(f"\n3. Computing Fractal Dimension (metric only)...")
    start = time.time()
    fd_positions, fractal_dimensions = sliding_window_fractal_dimension(identity_matrix, window_size, threshold)
    elapsed_fd_metric = time.time() - start

    # Scale positions back up to original sequence space
    fd_positions_scaled = [pos * subsample_every for pos in fd_positions]

    elapsed_fd_total = elapsed_pdist + elapsed_fd_metric
    print(f"   Metric computation: {elapsed_fd_metric:.2f} seconds")
    print(f"   Total (pdist + metric): {elapsed_fd_total:.2f} seconds")
    print(f"   Number of windows: {len(fd_positions)}")
    print(f"   Position range (scaled): {fd_positions_scaled[0]} to {fd_positions_scaled[-1]}")
    print(f"   Mean fractal dimension: {np.nanmean(fractal_dimensions):.4f}")
    print(f"   Std fractal dimension: {np.nanstd(fractal_dimensions):.4f}")

    computed_metrics['fd'] = {
        'positions': fd_positions_scaled,
        'values': fractal_dimensions,
        'label': 'Fractal Dimension',
        'color': 'b',
        'invert': False,
        'time_total': elapsed_fd_total,
        'time_metric': elapsed_fd_metric
    }

    # Calculate Correlation Dimension with sliding window
    print(f"\n4. Computing Correlation Dimension (metric only)...")
    start = time.time()
    d2_positions, correlation_dimensions = sliding_window_correlation_dimension(
        distance_matrix, window_size, r_min=0.01, r_max=0.5, n_radii=50
    )
    elapsed_d2_metric = time.time() - start

    # Scale positions back up to original sequence space
    d2_positions_scaled = [pos * subsample_every for pos in d2_positions]

    elapsed_d2_total = elapsed_pdist + elapsed_d2_metric
    print(f"   Metric computation: {elapsed_d2_metric:.2f} seconds")
    print(f"   Total (pdist + metric): {elapsed_d2_total:.2f} seconds")
    print(f"   Number of windows: {len(d2_positions)}")
    print(f"   Position range (scaled): {d2_positions_scaled[0]} to {d2_positions_scaled[-1]}")
    print(f"   Mean correlation dimension: {np.nanmean(correlation_dimensions):.4f}")
    print(f"   Std correlation dimension: {np.nanstd(correlation_dimensions):.4f}")

    computed_metrics['cd'] = {
        'positions': d2_positions_scaled,
        'values': correlation_dimensions,
        'label': 'Correlation Dimension (D2)',
        'color': 'r',
        'invert': True,
        'time_total': elapsed_d2_total,
        'time_metric': elapsed_d2_metric
    }

    # Calculate k-mer entropy on subsampled repeats (same window size as FD/CD)
    print(f"\n5. Computing k-mer Entropy on subsampled repeats (metric only, no pdist needed)...")
    print(f"   Using same subsampled repeats as FD/CD for fair comparison")
    start = time.time()
    kmer_positions, kmer_values = sliding_window_complexity_from_repeats(subsampled_repeats, window_size, method='kmer')
    elapsed_kmer_total = time.time() - start

    # Scale positions back up to original sequence space
    kmer_positions_scaled = [pos * subsample_every for pos in kmer_positions]

    print(f"   Total computation: {elapsed_kmer_total:.2f} seconds")
    print(f"   Number of windows: {len(kmer_positions)}")
    print(f"   Position range (scaled): {kmer_positions_scaled[0]} to {kmer_positions_scaled[-1]}")
    print(f"   Mean k-mer entropy: {np.mean(kmer_values):.4f}")
    print(f"   Std k-mer entropy: {np.std(kmer_values):.4f}")

    computed_metrics['kmer'] = {
        'positions': kmer_positions_scaled,
        'values': kmer_values,
        'label': 'k-mer Entropy',
        'color': 'g',
        'invert': True,
        'time_total': elapsed_kmer_total,
        'time_metric': elapsed_kmer_total
    }

    # Calculate LZMA compression on subsampled repeats (same window size as FD/CD)
    print(f"\n6. Computing LZMA Compressibility on subsampled repeats (metric only, no pdist needed)...")
    start = time.time()
    lzma_positions, lzma_values = sliding_window_complexity_from_repeats(subsampled_repeats, window_size, method='lzma')
    elapsed_lzma_total = time.time() - start

    # Scale positions back up to original sequence space
    lzma_positions_scaled = [pos * subsample_every for pos in lzma_positions]

    print(f"   Total computation: {elapsed_lzma_total:.2f} seconds")
    print(f"   Number of windows: {len(lzma_positions)}")
    print(f"   Position range (scaled): {lzma_positions_scaled[0]} to {lzma_positions_scaled[-1]}")
    print(f"   Mean LZMA compressibility: {np.mean(lzma_values):.4f}")
    print(f"   Std LZMA compressibility: {np.std(lzma_values):.4f}")

    computed_metrics['lzma'] = {
        'positions': lzma_positions_scaled,
        'values': lzma_values,
        'label': 'LZMA Compressibility',
        'color': 'm',
        'invert': True,
        'time_total': elapsed_lzma_total,
        'time_metric': elapsed_lzma_total
    }

    # Normalize all computed metrics to [0, 1] for comparison
    print(f"\n7. Normalizing all metrics to same scale...")

    def normalize(values):
        """Normalize array to [0, 1] range."""
        values = np.array(values)
        values = values[~np.isnan(values)]  # Remove NaN values
        if len(values) == 0:
            return np.array([])
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if vmax - vmin == 0:
            return np.ones_like(values) * 0.5
        return (values - vmin) / (vmax - vmin)

    # Normalize each computed metric
    inverted_metrics = []
    for key, metric_data in computed_metrics.items():
        values = metric_data['values']

        # Apply inversion if needed
        if metric_data.get('invert', False):
            values = [-v for v in values]
            inverted_metrics.append(key)

        # Normalize
        metric_data['normalized'] = normalize(values)

        # Print range
        orig_values = metric_data['values']
        print(f"   {metric_data['label']} range: [{np.nanmin(orig_values):.4f}, {np.nanmax(orig_values):.4f}]")

    if inverted_metrics:
        print(f"   Note: {', '.join(inverted_metrics)} values inverted for comparison")

    # Plot comparison with matrix visualization
    print(f"\n8. Creating comparison plot...")

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 12))

    # Top: rotated matrix
    ax_matrix = plt.subplot(2, 1, 1)

    # Prepare rotated identity matrix (upper triangle)
    masked_matrix = np.copy(identity_matrix).astype(float)
    for i in range(identity_matrix.shape[0]):
        for j in range(i):
            masked_matrix[i, j] = np.nan

    rotated = rotate(masked_matrix, 45, reshape=True, order=0, cval=np.nan)

    im = ax_matrix.imshow(rotated, cmap='viridis', interpolation='nearest', aspect='auto',
                          vmin=0.5, vmax=1.0)  # Focus on high identity region

    # Shift the image down by adjusting the y limits
    ylim = ax_matrix.get_ylim()
    ax_matrix.set_ylim(ylim[0] * 0.5, ylim[1])

    ax_matrix.set_title('Identity Matrix (45° rotation)\nDark = low identity, Light = high identity',
                       fontsize=12, fontweight='bold')
    ax_matrix.axis('off')

    # Add generation and threshold info
    info_text = f'Generation: 2,000,000\nThreshold: {threshold}'
    ax_matrix.text(0.02, 0.98, info_text,
                  transform=ax_matrix.transAxes,
                  fontsize=10, fontweight='bold',
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Bottom: metrics comparison
    ax = plt.subplot(2, 1, 2)

    # Plot all computed metrics normalized to same scale
    for key, metric_data in computed_metrics.items():
        ax.plot(metric_data['positions'], metric_data['normalized'],
                color=metric_data['color'], linewidth=1.5,
                label=metric_data['label'], alpha=0.8)

    ax.set_xlabel('Sequence Position (Repeat Index)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Value [0-1]', fontsize=12, fontweight='bold')
    ax.set_title(f'Complexity Metrics Comparison (Normalized)\n'
                 f'Window size: {window_size} repeats | {n_repeats} total repeats',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_repeats)
    ax.set_ylim(-0.05, 1.05)

    # Add summary statistics text box with original values
    stats_lines = []
    for key, metric_data in computed_metrics.items():
        values = metric_data['values']
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        stats_lines.append(f"{key.upper()}: {mean_val:.3f} ± {std_val:.3f}")

    stats_text = '\n'.join(stats_lines)
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    output_file = './output/fractal_vs_correlation_comparison.png'
    os.makedirs('./output', exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Plot saved as {output_file}")

    # Also save data to file (only computed metrics)
    data_file = './output/fractal_vs_correlation_data.npz'
    save_dict = {
        'window_size': window_size,
        'n_repeats': n_repeats,
        'repeat_len': repeat_len
    }

    # Add data for each computed metric
    for key, metric_data in computed_metrics.items():
        save_dict[f'{key}_positions'] = metric_data['positions']
        save_dict[f'{key}_values'] = metric_data['values']

    save_dict['subsample_every'] = subsample_every

    np.savez(data_file, **save_dict)
    print(f"   Data saved as {data_file}")

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nSummary:")
    for key, metric_data in computed_metrics.items():
        values = metric_data['values']
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        print(f"  {metric_data['label']:25s}: {mean_val:.4f} ± {std_val:.4f}")

    plt.show()


if __name__ == "__main__":
    main()
