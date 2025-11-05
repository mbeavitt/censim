#!/usr/bin/env python3
"""
Investigate the impact of different parameters on Correlation Dimension (D2) calculation.

This script systematically varies key parameters and plots the results alongside
the baseline for visual comparison.

Parameters investigated:
1. Subsampling depth (scale_factor)
2. Radius range (r_min, r_max)
3. Number of radii (n_radii)
4. Window size
5. Fit window (range used for slope estimation)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import time
import sys
import os
import argparse
from numba import njit, prange
try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from censim.correlation_dimension import (
    hamming_distance_matrix,
    correlation_sum_from_distance_matrix,
    estimate_D2_from_C_r_batch,
    sliding_window_local_correlation,
)
from censim.identity import all_vs_all_identity_scipy, all_vs_all_identity_numba


# Note: Local version removed - now using centralized version from censim.identity

# Keep this for reference, but use imported version
def _UNUSED_hamming_identity_matrix_numba_parallel(seq_array):
    """
    Compute pairwise Hamming identity with parallel numba.

    Args:
        seq_array: (N, L) uint8 array where N is number of sequences, L is length

    Returns:
        identity_matrix: (N, N) float32 array with pairwise identities (0-1 range)
    """
    N, L = seq_array.shape
    identity = np.zeros((N, N), dtype=np.float32)

    # Set diagonal to 1.0
    for i in range(N):
        identity[i, i] = 1.0

    # Parallel loop over upper triangle
    for i in prange(N):
        for j in range(i + 1, N):
            # Count matches
            matches = 0
            for k in range(L):
                if seq_array[i, k] == seq_array[j, k]:
                    matches += 1

            # Normalize by length
            ident = matches / L
            identity[i, j] = ident
            identity[j, i] = ident

    return identity


@njit(fastmath=True)
def hamming_identity_matrix_numba_serial(seq_array):
    """
    Compute pairwise Hamming identity with serial numba (no parallel).

    Args:
        seq_array: (N, L) uint8 array where N is number of sequences, L is length

    Returns:
        identity_matrix: (N, N) float32 array with pairwise identities (0-1 range)
    """
    N, L = seq_array.shape
    identity = np.zeros((N, N), dtype=np.float32)

    # Set diagonal to 1.0
    for i in range(N):
        identity[i, i] = 1.0

    # Serial loop over upper triangle
    for i in range(N):
        for j in range(i + 1, N):
            # Count matches
            matches = 0
            for k in range(L):
                if seq_array[i, k] == seq_array[j, k]:
                    matches += 1

            # Normalize by length
            ident = matches / L
            identity[i, j] = ident
            identity[j, i] = ident

    return identity


# CuPy CUDA kernel for Hamming identity computation
if CUPY_AVAILABLE:
    _hamming_kernel_cupy = cp.RawKernel(r'''
    extern "C" __global__
    void hamming_identity_kernel(const unsigned char* seq_array, float* identity, int N, int L) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= N || j >= N) return;

        if (i == j) {
            identity[i * N + j] = 1.0f;
        } else if (i < j) {
            // Only compute upper triangle
            int matches = 0;
            for (int k = 0; k < L; k++) {
                if (seq_array[i * L + k] == seq_array[j * L + k]) {
                    matches++;
                }
            }
            float ident = (float)matches / (float)L;
            identity[i * N + j] = ident;
            identity[j * N + i] = ident;  // Mirror
        }
    }
    ''', 'hamming_identity_kernel')


def hamming_identity_matrix_gpu(seq_array):
    """
    Compute pairwise Hamming identity using CuPy GPU.

    Args:
        seq_array: (N, L) uint8 array where N is number of sequences, L is length

    Returns:
        identity_matrix: (N, N) float32 array with pairwise identities (0-1 range)
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy/CUDA is not available. Please use CPU version instead.")

    N, L = seq_array.shape

    # Transfer to GPU
    seq_gpu = cp.asarray(seq_array, dtype=cp.uint8)
    identity_gpu = cp.zeros((N, N), dtype=cp.float32)

    # Configure grid and block dimensions
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (N + threads_per_block[0] - 1) // threads_per_block[0],
        (N + threads_per_block[1] - 1) // threads_per_block[1]
    )

    # Launch kernel
    _hamming_kernel_cupy(
        blocks_per_grid,
        threads_per_block,
        (seq_gpu, identity_gpu, N, L)
    )

    # Transfer back to CPU
    identity = cp.asnumpy(identity_gpu)

    return identity


def convert_sequences_fast(repeats):
    """
    Fast conversion of string sequences to uint8 numpy array.

    Args:
        repeats: list of sequence strings (all same length)

    Returns:
        seq_array: (N, L) uint8 array
    """
    if len(repeats) == 0:
        return np.array([], dtype=np.uint8)

    # Get dimensions
    N = len(repeats)
    L = len(repeats[0])

    # Pre-allocate array
    seq_array = np.empty((N, L), dtype=np.uint8)

    # Convert strings to bytes efficiently
    for i, seq in enumerate(repeats):
        # Use numpy's frombuffer for fast conversion
        seq_array[i] = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)

    return seq_array


# REMOVED: Local version that returned a tuple - now using imported version from censim.identity


def sliding_window_correlation_dimension_gpu(distance_matrix, window_size=100, r_min=0.01, r_max=0.5, n_radii=50):
    """
    GPU-accelerated sliding window correlation dimension using CuPy.

    Processes all windows in batch on GPU to minimize transfer overhead.

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
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available for GPU metric computation")

    N = distance_matrix.shape[0]
    r_values = np.linspace(r_min, r_max, n_radii)

    if window_size >= N:
        positions = [N // 2]
        C_values = correlation_sum_from_distance_matrix(distance_matrix, r_values)
        d2 = estimate_D2_from_C_r_batch(r_values, C_values.reshape(1, -1))[0]
        return positions, [d2]

    # Transfer to GPU ONCE
    distance_matrix_gpu = cp.asarray(distance_matrix, dtype=cp.float32)
    r_values_gpu = cp.asarray(r_values, dtype=cp.float32)

    n_windows = N - window_size + 1
    positions = list(range(window_size // 2, N - window_size // 2 + 1))

    # Pre-compute upper triangle indices
    triu_i, triu_j = cp.triu_indices(window_size, k=1)
    n_pairs = len(triu_i)

    # Extract all windows at once (batched)
    # Shape: (n_windows, n_pairs)
    all_distances = cp.zeros((n_windows, n_pairs), dtype=cp.float32)

    for i in range(n_windows):
        window_gpu = distance_matrix_gpu[i:i+window_size, i:i+window_size]
        all_distances[i] = window_gpu[triu_i, triu_j]

    # Sort all windows in parallel on GPU
    all_distances_sorted = cp.sort(all_distances, axis=1)

    # Compute correlation sums for all windows and radii
    # Shape: (n_windows, n_radii)
    C_matrix_gpu = cp.zeros((n_windows, n_radii), dtype=cp.float32)

    # Vectorize searchsorted for all radii at once per window
    for i in range(n_windows):
        counts = cp.searchsorted(all_distances_sorted[i], r_values_gpu, side='left')
        C_matrix_gpu[i, :] = (counts * 2.0) / (window_size * (window_size - 1))

    # Transfer result back to CPU (one transfer for all data)
    C_matrix = cp.asnumpy(C_matrix_gpu)

    # Batch compute D2 on CPU
    d2_values = estimate_D2_from_C_r_batch(r_values, C_matrix)

    return positions, d2_values.tolist()


def sliding_window_correlation_dimension(distance_matrix, window_size=100, r_min=0.01, r_max=0.5, n_radii=50, use_gpu=False):
    """
    Calculate correlation dimension using sliding window on distance matrix.

    Args:
        distance_matrix: (N, N) distance matrix
        window_size: size of sliding window
        r_min: minimum radius for correlation sum
        r_max: maximum radius for correlation sum
        n_radii: number of radii to test
        use_gpu: whether to use GPU acceleration (default: False)

    Returns:
        positions: center positions of windows
        d2_values: correlation dimension at each window position
    """
    if use_gpu and CUPY_AVAILABLE:
        return sliding_window_correlation_dimension_gpu(distance_matrix, window_size, r_min, r_max, n_radii)

    N = distance_matrix.shape[0]
    r_values = np.linspace(r_min, r_max, n_radii)

    if window_size >= N:
        positions = [N // 2]
        C_values = correlation_sum_from_distance_matrix(distance_matrix, r_values)
        d2 = estimate_D2_from_C_r_batch(r_values, C_values.reshape(1, -1))[0]
        return positions, [d2]

    positions = []
    all_C_values = []

    triu_mask = np.triu(np.ones((window_size, window_size), dtype=bool), k=1)

    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        window = distance_matrix[i:i+window_size, i:i+window_size]
        C_values = correlation_sum_from_distance_matrix(window, r_values, triu_mask)
        all_C_values.append(C_values)

    C_matrix = np.array(all_C_values)
    d2_values = estimate_D2_from_C_r_batch(r_values, C_matrix)

    return positions, d2_values.tolist()


def investigate_parameter(param_name, param_values, repeats, baseline_params, n_repeats_total, subsample_every_baseline):
    """
    Investigate the impact of varying a single parameter.

    Args:
        param_name: name of parameter being varied
        param_values: list of values to test for this parameter
        repeats: full list of sequence repeats
        baseline_params: dict of baseline parameter values
        n_repeats_total: total number of repeats in original data
        subsample_every_baseline: baseline subsampling interval

    Returns:
        results: dict containing positions and d2_values for each param value
    """
    results = {}

    for param_val in param_values:
        print(f"  Testing {param_name} = {param_val}...")
        start = time.time()

        # Update parameters for this test
        test_params = baseline_params.copy()
        test_params[param_name] = param_val

        # Compute identity matrix with appropriate subsampling
        if param_name == 'scale_factor':
            identity_matrix, subsample_every, _ = all_vs_all_identity_numba(
                repeats,
                max_exact_size=1000,
                scale_factor=param_val
            )
        else:
            # Use baseline subsampling for other parameters
            identity_matrix, subsample_every, _ = all_vs_all_identity_numba(
                repeats,
                max_exact_size=1000,
                scale_factor=baseline_params.get('scale_factor', 30)
            )

        distance_matrix = hamming_distance_matrix(identity_matrix)

        # Compute D2 with updated parameters (always use CPU for metric - faster)
        positions, d2_values = sliding_window_correlation_dimension(
            distance_matrix,
            window_size=test_params.get('window_size', 100),
            r_min=test_params.get('r_min', 0.01),
            r_max=test_params.get('r_max', 0.5),
            n_radii=test_params.get('n_radii', 50),
            use_gpu=False  # Always False - CPU is faster for sliding window
        )

        # Scale positions back to original sequence space
        positions_scaled = [pos * subsample_every for pos in positions]

        elapsed = time.time() - start
        print(f"    Completed in {elapsed:.2f}s - Mean D2: {np.nanmean(d2_values):.4f}")

        results[param_val] = {
            'positions': positions_scaled,
            'values': d2_values,
            'subsample': subsample_every,
            'time': elapsed
        }

    return results


def plot_parameter_investigation(param_name, param_values, results, baseline_value, n_repeats, output_prefix):
    """
    Create comparison plot for a parameter investigation.

    Shows baseline on top, and each parameter variation below for comparison.

    Args:
        param_name: name of parameter being varied
        param_values: list of values tested
        results: dict of results from investigate_parameter
        baseline_value: the baseline parameter value
        n_repeats: total number of repeats for x-axis
        output_prefix: prefix for output filename
    """
    n_params = len(param_values)

    # Create figure with baseline on top, variations below
    fig, axes = plt.subplots(n_params + 1, 1, figsize=(14, 3 * (n_params + 1)), sharex=True)

    if n_params == 0:
        axes = [axes]

    # Find baseline data
    baseline_data = results[baseline_value]
    baseline_positions = baseline_data['positions']
    baseline_values = baseline_data['values']

    # Plot baseline on top
    ax = axes[0]
    ax.plot(baseline_positions, baseline_values, 'b-', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('D2', fontsize=10, fontweight='bold')
    ax.set_title(f'BASELINE: {param_name} = {baseline_value}\n' +
                 f'Mean D2: {np.nanmean(baseline_values):.4f} ± {np.nanstd(baseline_values):.4f}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_repeats)

    # Plot each variation
    for idx, param_val in enumerate([v for v in param_values if v != baseline_value]):
        ax = axes[idx + 1]

        data = results[param_val]
        positions = data['positions']
        values = data['values']

        # Plot variation
        ax.plot(positions, values, 'r-', linewidth=1.5, alpha=0.8, label=f'{param_name}={param_val}')

        # Overlay baseline as faint reference
        ax.plot(baseline_positions, baseline_values, 'b-', linewidth=1.0, alpha=0.2, label=f'baseline ({baseline_value})')

        ax.set_ylabel('D2', fontsize=10, fontweight='bold')
        ax.set_title(f'{param_name} = {param_val}\n' +
                     f'Mean D2: {np.nanmean(values):.4f} ± {np.nanstd(values):.4f} | ' +
                     f'Subsample: {data["subsample"]} | Time: {data["time"]:.1f}s',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlim(0, n_repeats)

    # Label bottom plot
    axes[-1].set_xlabel('Sequence Position (Repeat Index)', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save plot
    output_file = f'{output_prefix}_{param_name}_investigation.png'
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {output_file}")

    plt.close()


def benchmark_parameters(fasta_dir, n_files=2000, gen_min=2000000, gen_max=6000000):
    """
    Benchmark different parameter configurations on a large set of files.

    Args:
        fasta_dir: directory containing FASTA files
        n_files: number of files to process for benchmarking
        gen_min: minimum generation to include
        gen_max: maximum generation to include

    Returns:
        benchmark_results: dict of timing results for each configuration
    """
    import glob
    import re

    # Find all FASTA files in the generation range
    all_files = glob.glob(os.path.join(fasta_dir, '*.fa'))

    # Filter files by generation range
    filtered_files = []
    for fpath in all_files:
        fname = os.path.basename(fpath)
        match = re.match(r'(\d+)generation\.out\.fa', fname)
        if match:
            gen = int(match.group(1))
            if gen_min <= gen <= gen_max:
                filtered_files.append((gen, fpath))

    # Sort by generation and take evenly spaced samples
    filtered_files.sort()
    total_files = len(filtered_files)

    if total_files == 0:
        print("ERROR: No files found in generation range!")
        return {}

    # Sample evenly across the range
    step = max(1, total_files // n_files)
    sampled_files = [filtered_files[i][1] for i in range(0, total_files, step)][:n_files]

    print(f"\nBenchmarking on {len(sampled_files)} files from generation {gen_min} to {gen_max}")
    print(f"  Total files in range: {total_files}")
    print(f"  Sampling every {step} file(s)")

    # Parameter configurations to benchmark
    # Compare scipy (baseline), numba serial (single-threaded), and numba parallel (multi-threaded)
    configs = {
        'scipy_baseline': {'scale_factor': 25, 'r_min': 0.01, 'r_max': 0.5, 'n_radii': 50, 'window_size': 100, 'use_scipy': True},  # SciPy baseline
        'numba_serial': {'scale_factor': 25, 'r_min': 0.01, 'r_max': 0.5, 'n_radii': 50, 'window_size': 100, 'use_parallel': False},  # Numba serial (default)
        'numba_parallel': {'scale_factor': 25, 'r_min': 0.01, 'r_max': 0.5, 'n_radii': 50, 'window_size': 100, 'use_parallel': True},  # Numba parallel
    }

    results = {}

    for config_name, params in configs.items():
        print(f"\n  Testing configuration: {config_name}")
        print(f"    Parameters: {params}")

        total_time = 0
        pdist_time = 0
        metric_time = 0
        n_processed = 0

        for i, fpath in enumerate(sampled_files):
            if (i + 1) % 100 == 0:
                print(f"    Progress: {i+1}/{len(sampled_files)} files ({100*(i+1)/len(sampled_files):.1f}%)")

            try:
                # Load file (skip FASTA header)
                with open(fpath, 'r') as f:
                    lines = f.readlines()
                    # Skip header line (starts with >)
                    contents = ''.join(line.strip() for line in lines if not line.startswith('>'))

                repeat_len = 178
                n_repeats = int(len(contents) / repeat_len)
                repeats = [contents[j*repeat_len:(j+1)*repeat_len] for j in range(n_repeats)]

                # Time pdist computation
                t0 = time.time()
                if params.get('use_scipy', False):
                    # Use scipy baseline
                    identity_matrix = all_vs_all_identity_scipy(
                        repeats, max_exact_size=1000, scale_factor=params['scale_factor']
                    )
                else:
                    # Use numba (serial or parallel)
                    identity_matrix = all_vs_all_identity_numba(
                        repeats, max_exact_size=1000, scale_factor=params['scale_factor'],
                        use_parallel=params.get('use_parallel', False)
                    )
                t1 = time.time()
                pdist_time += (t1 - t0)

                distance_matrix = hamming_distance_matrix(identity_matrix)

                # Time metric computation
                t0 = time.time()
                r_values = np.linspace(params['r_min'], params['r_max'], params['n_radii'])
                positions, mean_corr = sliding_window_local_correlation(
                    distance_matrix,
                    window_size=params['window_size'],
                    r_values=r_values,
                    use_parallel=params.get('use_parallel', False)
                )
                # Compute D2 values
                log_r = np.log(r_values)
                d2_values = estimate_D2_from_C_r_batch(r_values, mean_corr, log_r=log_r)
                t1 = time.time()
                metric_time += (t1 - t0)

                n_processed += 1

            except Exception as e:
                import traceback
                print(f"    WARNING: Failed to process {fpath}: {e}")
                traceback.print_exc()
                continue

        total_time = pdist_time + metric_time
        avg_total = total_time / n_processed if n_processed > 0 else 0
        avg_pdist = pdist_time / n_processed if n_processed > 0 else 0
        avg_metric = metric_time / n_processed if n_processed > 0 else 0

        results[config_name] = {
            'params': params,
            'total_time': total_time,
            'pdist_time': pdist_time,
            'metric_time': metric_time,
            'n_files': n_processed,
            'avg_total': avg_total,
            'avg_pdist': avg_pdist,
            'avg_metric': avg_metric
        }

        print(f"    Completed: {n_processed} files")
        print(f"    Total time: {total_time:.2f}s")
        print(f"    Avg per file: {avg_total:.3f}s (pdist: {avg_pdist:.3f}s, metric: {avg_metric:.3f}s)")

    return results


def plot_benchmark_results(results, output_file):
    """
    Create summary plot of benchmark results.

    Args:
        results: dict from benchmark_parameters
        output_file: path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    config_names = list(results.keys())
    total_times = [results[c]['total_time'] for c in config_names]
    pdist_times = [results[c]['pdist_time'] for c in config_names]
    metric_times = [results[c]['metric_time'] for c in config_names]
    avg_total_times = [results[c]['avg_total'] for c in config_names]

    # Plot 1: Total time comparison
    ax = axes[0]
    bars = ax.bar(range(len(config_names)), total_times, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.set_ylabel('Total Time (s)', fontsize=11, fontweight='bold')
    ax.set_title('Total Processing Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight baseline (if exists)
    if 'baseline' in config_names:
        baseline_idx = config_names.index('baseline')
        bars[baseline_idx].set_color('darkgreen')
        bars[baseline_idx].set_alpha(0.8)
    elif 'scale_25_parallel' in config_names:
        baseline_idx = config_names.index('scale_25_parallel')
        bars[baseline_idx].set_color('darkgreen')
        bars[baseline_idx].set_alpha(0.8)

    # Plot 2: Breakdown (pdist vs metric)
    ax = axes[1]
    x = np.arange(len(config_names))
    width = 0.35
    ax.bar(x - width/2, pdist_times, width, label='pdist', color='coral', alpha=0.7)
    ax.bar(x + width/2, metric_times, width, label='metric', color='skyblue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.set_ylabel('Time (s)', fontsize=11, fontweight='bold')
    ax.set_title('Time Breakdown: pdist vs metric', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Average time per file
    ax = axes[2]
    bars = ax.bar(range(len(config_names)), avg_total_times, color='mediumpurple', alpha=0.7)
    ax.set_xticks(range(len(config_names)))
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.set_ylabel('Avg Time per File (s)', fontsize=11, fontweight='bold')
    ax.set_title('Average Processing Time per File', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight baseline (if exists)
    if 'baseline' in config_names:
        baseline_idx = config_names.index('baseline')
        bars[baseline_idx].set_color('darkgreen')
        bars[baseline_idx].set_alpha(0.8)
    elif 'scale_25_parallel' in config_names:
        baseline_idx = config_names.index('scale_25_parallel')
        bars[baseline_idx].set_color('darkgreen')
        bars[baseline_idx].set_alpha(0.8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nBenchmark summary plot saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Investigate correlation dimension parameter sensitivity'
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        default="./data/2191000generation.out.fa",
        help='Input FASTA file (default: ./data/2191000generation.out.fa)'
    )

    parser.add_argument(
        '-o', '--output-prefix',
        type=str,
        default='./output/param_investigation',
        help='Output file prefix (default: ./output/param_investigation)'
    )

    parser.add_argument(
        '--params',
        nargs='+',
        choices=['scale_factor', 'r_min', 'r_max', 'n_radii', 'window_size', 'all'],
        default=['all'],
        help='Parameters to investigate (default: all)'
    )

    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark on 2000 files from 2M-6M generations'
    )

    parser.add_argument(
        '--benchmark-dir',
        type=str,
        default='./output/fasta',
        help='Directory containing FASTA files for benchmarking (default: ./output/fasta)'
    )

    parser.add_argument(
        '--benchmark-files',
        type=int,
        default=200,
        help='Number of files to use for benchmarking (default: 200)'
    )

    args = parser.parse_args()

    # If benchmark mode, run benchmark and exit
    if args.benchmark:
        print("="*80)
        print("Correlation Dimension Parameter Benchmarking")
        print("="*80)

        results = benchmark_parameters(
            args.benchmark_dir,
            n_files=args.benchmark_files,
            gen_min=2000000,
            gen_max=6000000
        )

        if not results:
            print("\nERROR: Benchmark failed - no results")
            return

        # Print summary
        print("\n" + "="*80)
        print("Benchmark Summary")
        print("="*80)
        print(f"\n{'Config':<15} {'Total(s)':<12} {'pdist(s)':<12} {'metric(s)':<12} {'Avg/file(s)':<12}")
        print("-" * 80)

        for config_name, res in results.items():
            print(f"{config_name:<15} {res['total_time']:<12.2f} {res['pdist_time']:<12.2f} "
                  f"{res['metric_time']:<12.2f} {res['avg_total']:<12.4f}")

        # Find fastest
        fastest = min(results.items(), key=lambda x: x[1]['total_time'])
        print("\n" + "="*80)
        print(f"FASTEST: {fastest[0]} - {fastest[1]['total_time']:.2f}s total")
        print("="*80)

        # Create summary plot
        plot_benchmark_results(results, f'{args.output_prefix}_benchmark_summary.png')

        # Save detailed results to file
        results_file = f'{args.output_prefix}_benchmark_results.txt'
        with open(results_file, 'w') as f:
            f.write("Correlation Dimension Parameter Benchmark Results\n")
            f.write("="*80 + "\n\n")
            for config_name, res in results.items():
                f.write(f"Configuration: {config_name}\n")
                f.write(f"  Parameters: {res['params']}\n")
                f.write(f"  Files processed: {res['n_files']}\n")
                f.write(f"  Total time: {res['total_time']:.2f}s\n")
                f.write(f"  pdist time: {res['pdist_time']:.2f}s\n")
                f.write(f"  metric time: {res['metric_time']:.2f}s\n")
                f.write(f"  Avg per file: {res['avg_total']:.4f}s\n")
                f.write("\n")
            f.write(f"\nFASTEST: {fastest[0]} - {fastest[1]['total_time']:.2f}s total\n")

        print(f"Detailed results saved: {results_file}")

        return

    # Expand 'all' to all parameters
    if 'all' in args.params:
        params_to_test = ['scale_factor', 'r_min', 'r_max', 'n_radii', 'window_size']
    else:
        params_to_test = args.params

    print("="*80)
    print("Correlation Dimension Parameter Investigation")
    print("="*80)
    print(f"Input file: {args.input}")
    print(f"Parameters to investigate: {', '.join(params_to_test)}")

    # Load data
    print(f"\n1. Loading sequence data...")
    with open(args.input, "r") as file:
        contents = file.read().strip()

    repeat_len = 178
    n_repeats = int(len(contents) / repeat_len)
    repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]

    print(f"   Loaded {n_repeats} repeats of length {repeat_len}bp")

    # Define baseline parameters (from compare_fractal_vs_correlation.py)
    baseline_params = {
        'scale_factor': 30,
        'r_min': 0.01,
        'r_max': 0.5,
        'n_radii': 50,
        'window_size': 100
    }

    print(f"\nBaseline parameters:")
    for k, v in baseline_params.items():
        print(f"  {k}: {v}")

    # Define parameter variations to test
    param_variations = {
        'scale_factor': [10, 20, 30, 50, 100],  # Lower = more data points, slower
        'r_min': [0.001, 0.01, 0.05],  # Start of radius range
        'r_max': [0.3, 0.5, 0.7],  # End of radius range
        'n_radii': [20, 50, 100],  # Number of radii tested
        'window_size': [50, 100, 200],  # Window size in repeats
    }

    # Compute baseline once
    print(f"\n2. Computing baseline (scale_factor={baseline_params['scale_factor']})...")
    start = time.time()
    identity_matrix, subsample_every_baseline, _ = all_vs_all_identity_numba(
        repeats, max_exact_size=1000, scale_factor=baseline_params['scale_factor']
    )
    elapsed = time.time() - start
    print(f"   Identity matrix computed in {elapsed:.2f}s")
    print(f"   Matrix shape: {identity_matrix.shape}")
    print(f"   Subsampling interval: {subsample_every_baseline}")

    # Investigate each parameter
    step = 3
    for param_name in params_to_test:
        print(f"\n{step}. Investigating {param_name}...")
        param_values = param_variations[param_name]

        results = investigate_parameter(
            param_name,
            param_values,
            repeats,
            baseline_params,
            n_repeats,
            subsample_every_baseline
        )

        # Plot results
        print(f"   Creating comparison plot...")
        plot_parameter_investigation(
            param_name,
            param_values,
            results,
            baseline_params[param_name],
            n_repeats,
            args.output_prefix
        )

        step += 1

    print("\n" + "="*80)
    print("Investigation Complete!")
    print("="*80)
    print(f"Output files saved to: {args.output_prefix}_*.png")


if __name__ == "__main__":
    main()
