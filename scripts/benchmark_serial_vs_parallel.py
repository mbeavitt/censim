#!/usr/bin/env python3
"""
Benchmark scipy vs numba serial vs numba parallel for correlation dimension computation.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from censim.identity import all_vs_all_identity_scipy, all_vs_all_identity_numba
from censim.correlation_dimension import (
    hamming_distance_matrix,
    sliding_window_local_correlation,
    estimate_D2_from_C_r_batch
)


def generate_test_data(n_repeats=500, repeat_len=178):
    """Generate random test sequences."""
    bases = ['A', 'C', 'G', 'T']
    np.random.seed(42)
    repeats = []
    for _ in range(n_repeats):
        seq = ''.join(np.random.choice(bases, size=repeat_len))
        repeats.append(seq)
    return repeats


def benchmark_full_pipeline(repeats, method='scipy', use_parallel=False):
    """Benchmark full correlation dimension pipeline."""

    # 1. Compute identity matrix
    t0 = time.time()
    if method == 'scipy':
        identity_matrix = all_vs_all_identity_scipy(repeats, max_exact_size=1000, scale_factor=30)
    else:  # numba
        identity_matrix = all_vs_all_identity_numba(repeats, max_exact_size=1000, scale_factor=30, use_parallel=use_parallel)
    t1 = time.time()
    pdist_time = t1 - t0

    # 2. Convert to distance matrix
    D = hamming_distance_matrix(identity_matrix)

    # 3. Compute correlation dimension with sliding window
    r_values = np.linspace(0.01, 0.5, 50)
    t0 = time.time()
    positions, mean_corr = sliding_window_local_correlation(D, window_size=100, r_values=r_values, use_parallel=use_parallel)
    t1 = time.time()
    metric_time = t1 - t0

    # 4. Estimate D2 values
    t0 = time.time()
    log_r_values = np.log(r_values)
    d_values = estimate_D2_from_C_r_batch(r_values, mean_corr, log_r=log_r_values)
    t1 = time.time()
    d2_time = t1 - t0

    total_time = pdist_time + metric_time + d2_time

    return {
        'pdist_time': pdist_time,
        'metric_time': metric_time,
        'd2_time': d2_time,
        'total_time': total_time,
        'mean_d2': np.nanmean(d_values)
    }


if __name__ == "__main__":
    print("="*80)
    print("Benchmarking scipy vs numba serial vs numba parallel")
    print("="*80)

    # Generate test data
    print("\nGenerating test data (500 sequences of 178 bp)...")
    repeats = generate_test_data(n_repeats=500, repeat_len=178)
    print(f"✓ Generated {len(repeats)} sequences")

    # Warmup JIT compilation
    print("\nWarming up JIT compilation...")
    warmup_repeats = generate_test_data(n_repeats=50, repeat_len=178)
    benchmark_full_pipeline(warmup_repeats, method='numba', use_parallel=False)
    benchmark_full_pipeline(warmup_repeats, method='numba', use_parallel=True)
    print("✓ JIT warmup complete")

    # Run benchmarks
    print("\n" + "="*80)
    print("Running benchmarks...")
    print("="*80)

    methods = [
        ('scipy', False, 'SciPy (baseline)'),
        ('numba', False, 'Numba Serial'),
        ('numba', True, 'Numba Parallel'),
    ]

    results = {}

    for method, use_parallel, label in methods:
        print(f"\n{label}:")
        result = benchmark_full_pipeline(repeats, method=method, use_parallel=use_parallel)
        results[label] = result

        print(f"  pdist:      {result['pdist_time']:.3f}s")
        print(f"  metric:     {result['metric_time']:.3f}s")
        print(f"  D2 est:     {result['d2_time']:.3f}s")
        print(f"  TOTAL:      {result['total_time']:.3f}s")
        print(f"  Mean D2:    {result['mean_d2']:.4f}")

    # Summary comparison
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    scipy_time = results['SciPy (baseline)']['total_time']

    print(f"\n{'Method':<20} {'Total Time':<12} {'vs SciPy':<12} {'Speedup':<10}")
    print("-" * 60)

    for label in ['SciPy (baseline)', 'Numba Serial', 'Numba Parallel']:
        result = results[label]
        total = result['total_time']
        speedup = scipy_time / total
        vs_scipy = f"{speedup:.2f}x"

        print(f"{label:<20} {total:>8.3f}s    {vs_scipy:<12} ", end="")
        if speedup > 1.5:
            print("⚡ FASTER")
        elif speedup < 0.8:
            print("❌ SLOWER")
        else:
            print("≈ similar")

    # Recommendation
    print("\n" + "="*80)
    best = min(results.items(), key=lambda x: x[1]['total_time'])
    print(f"FASTEST: {best[0]} - {best[1]['total_time']:.3f}s")
    print("="*80)
