import numpy as np
import time
import os
import glob
import math
from numba import jit, prange
from src.censim.identity import all_vs_all_identity_numba
from src.censim.correlation_dimension import (
    hamming_distance_matrix,
    sliding_window_local_correlation,
    estimate_D2_from_C_r_batch,
    choose_meaningful_radii
)

def all_vs_all_identity_with_numba(repeats, max_exact_size=1000, scale_factor=30):
    """Compute all vs all identity matrix using numba with adaptive subsampling."""
    n_repeats = len(repeats)

    if n_repeats <= max_exact_size:
        subsample_every = 1
    else:
        subsample_every = max(1, int(math.sqrt(n_repeats / scale_factor)))

    if subsample_every == 1:
        identity_matrix = all_vs_all_identity_numba(repeats, max_exact_size=max_exact_size,
                                                     scale_factor=scale_factor, use_parallel=True)
        subset_repeats = repeats
    else:
        subset_indices = np.arange(0, n_repeats, subsample_every)
        subset_repeats = [repeats[i] for i in subset_indices]
        identity_matrix = all_vs_all_identity_numba(subset_repeats, max_exact_size=max_exact_size,
                                                     scale_factor=scale_factor, use_parallel=True)

    return identity_matrix.astype(np.float32), subsample_every, subset_repeats

@jit(nopython=True)
def seq_to_numeric(seq_bytes):
    """Convert sequence bytes to numeric array (A=0, C=1, G=2, T=3)."""
    result = np.zeros(len(seq_bytes), dtype=np.int8)
    for i in range(len(seq_bytes)):
        if seq_bytes[i] == 65:  # 'A'
            result[i] = 0
        elif seq_bytes[i] == 67:  # 'C'
            result[i] = 1
        elif seq_bytes[i] == 71:  # 'G'
            result[i] = 2
        elif seq_bytes[i] == 84:  # 'T'
            result[i] = 3
        else:
            result[i] = -1  # Invalid base
    return result

@jit(nopython=True)
def build_kmer_profiles(numeric_seqs, k, n_kmers_total):
    """Build k-mer profile matrix using numba JIT."""
    n_repeats = len(numeric_seqs)
    profile_matrix = np.zeros((n_repeats, n_kmers_total), dtype=np.float32)

    base_powers = np.array([4**(k-1-i) for i in range(k)], dtype=np.int32)

    for rep_idx in range(n_repeats):
        seq = numeric_seqs[rep_idx]
        seq_len = len(seq)

        if seq_len < k:
            continue

        n_kmers = seq_len // k

        for kmer_idx in range(n_kmers):
            start = kmer_idx * k

            # Convert k-mer to index
            kmer_value = 0
            valid = True
            for pos in range(k):
                base = seq[start + pos]
                if base < 0 or base > 3:
                    valid = False
                    break
                kmer_value += base * base_powers[pos]

            if valid:
                profile_matrix[rep_idx, kmer_value] += 1

        # Normalize to frequencies
        total = 0.0
        for j in range(n_kmers_total):
            total += profile_matrix[rep_idx, j]
        if total > 0:
            for j in range(n_kmers_total):
                profile_matrix[rep_idx, j] /= total

    return profile_matrix

@jit(nopython=True)
def compute_window_diversity(profile_matrix, window_size):
    """Compute diversity for all windows using numba JIT."""
    N = profile_matrix.shape[0]
    n_windows = N - window_size + 1
    n_features = profile_matrix.shape[1]
    diversity_values = np.zeros(n_windows, dtype=np.float32)

    for i in range(n_windows):
        # Compute consensus (mean profile) manually
        consensus = np.zeros(n_features, dtype=np.float32)
        for j in range(window_size):
            for k in range(n_features):
                consensus[k] += profile_matrix[i + j, k]
        for k in range(n_features):
            consensus[k] /= window_size

        # Compute Euclidean distances from consensus
        total_dist = 0.0
        for j in range(window_size):
            dist_sq = 0.0
            for k in range(n_features):
                diff = profile_matrix[i + j, k] - consensus[k]
                dist_sq += diff * diff
            total_dist += np.sqrt(dist_sq)

        diversity_values[i] = total_dist / window_size

    return diversity_values

def sliding_window_kmer_profile_distance(repeats, window_size=100, k=4):
    """Calculate average distance from k-mer consensus (numba-optimized, no parallel)."""
    n_kmers_total = 4 ** k  # 256 for k=4

    # Convert sequences to numeric arrays
    numeric_seqs = [seq_to_numeric(np.frombuffer(repeat.encode(), dtype=np.uint8))
                    for repeat in repeats]

    # Build k-mer profiles using JIT
    profile_matrix = build_kmer_profiles(numeric_seqs, k, n_kmers_total)

    # Compute diversity using JIT
    diversity_values = compute_window_diversity(profile_matrix, window_size)

    # Generate positions
    N = len(repeats)
    positions = [i + window_size // 2 for i in range(N - window_size + 1)]

    return positions, diversity_values.tolist()


# Configuration
repeat_len = 178
window_size = 100
max_exact_size = 1000
scale_factor = 30

# Get all fasta files
fasta_files = sorted(glob.glob("output/fasta/*.fa"))[:11]  # Test on 11 files (1 warmup + 10 benchmark)

print(f"Warming up numba JIT compilation...")
print("=" * 80)

# Warmup run to compile numba functions
warmup_file = fasta_files[0]
with open(warmup_file, "r") as f:
    lines = f.readlines()
    contents = ''.join(line.strip() for line in lines if not line.startswith('>'))
n_repeats = len(contents) // repeat_len
repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]
identity_matrix, subsample_every, subsampled_repeats = all_vs_all_identity_with_numba(
    repeats, max_exact_size=max_exact_size, scale_factor=scale_factor
)
# Warmup both metrics
kpd_positions, kpd_values = sliding_window_kmer_profile_distance(subsampled_repeats, window_size=window_size, k=4)
distance_matrix = hamming_distance_matrix(identity_matrix)
r_values = choose_meaningful_radii(distance_matrix, n_radii=50, r_min=0.01, r_max=0.5)
cd_positions, cd_correlations = sliding_window_local_correlation(distance_matrix, window_size, r_values, use_parallel=True)
cd_d2_values = estimate_D2_from_C_r_batch(r_values, cd_correlations, fit_window=(0.05, 0.3))
print(f"Warmup complete on {os.path.basename(warmup_file)}")

# Now benchmark on remaining files
fasta_files = fasta_files[1:]  # Skip warmup file
print(f"\nBenchmarking on {len(fasta_files)} files")
print("=" * 80)

kpd_times = []
cd_times = []

for fasta_file in fasta_files:
    print(f"\nProcessing: {os.path.basename(fasta_file)}")

    # Load sequence (skip header line)
    with open(fasta_file, "r") as f:
        lines = f.readlines()
        contents = ''.join(line.strip() for line in lines if not line.startswith('>'))

    n_repeats = len(contents) // repeat_len
    repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]
    print(f"  {n_repeats} repeats")

    # Compute identity matrix and subsample
    identity_matrix, subsample_every, subsampled_repeats = all_vs_all_identity_with_numba(
        repeats, max_exact_size=max_exact_size, scale_factor=scale_factor
    )
    print(f"  Subsampled: {len(subsampled_repeats)} repeats (every {subsample_every})")

    # Benchmark K-mer Profile Distance (NO identity matrix needed)
    start = time.time()
    kpd_positions, kpd_values = sliding_window_kmer_profile_distance(subsampled_repeats, window_size=window_size, k=4)
    kpd_time = time.time() - start
    kpd_times.append(kpd_time)
    print(f"  K-mer Profile Distance: {kpd_time:.4f}s")

    # Benchmark Correlation Dimension (uses identity matrix)
    start = time.time()
    distance_matrix = hamming_distance_matrix(identity_matrix)
    r_values = choose_meaningful_radii(distance_matrix, n_radii=50, r_min=0.01, r_max=0.5)
    cd_positions, cd_correlations = sliding_window_local_correlation(distance_matrix, window_size, r_values, use_parallel=True)
    cd_d2_values = estimate_D2_from_C_r_batch(r_values, cd_correlations, fit_window=(0.05, 0.3))
    cd_time = time.time() - start
    cd_times.append(cd_time)
    print(f"  Correlation Dimension: {cd_time:.4f}s")
    print(f"  Ratio: {cd_time/kpd_time:.2f}x")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"K-mer Profile Distance (NO identity matrix):")
print(f"  Mean: {np.mean(kpd_times):.4f}s")
print(f"  Std:  {np.std(kpd_times):.4f}s")
print(f"  Min:  {np.min(kpd_times):.4f}s")
print(f"  Max:  {np.max(kpd_times):.4f}s")
print()
print(f"Correlation Dimension (uses identity matrix):")
print(f"  Mean: {np.mean(cd_times):.4f}s")
print(f"  Std:  {np.std(cd_times):.4f}s")
print(f"  Min:  {np.min(cd_times):.4f}s")
print(f"  Max:  {np.max(cd_times):.4f}s")
print()
if np.mean(cd_times) > np.mean(kpd_times):
    print(f"Correlation Dimension is {np.mean(cd_times)/np.mean(kpd_times):.2f}x SLOWER")
else:
    print(f"K-mer Profile Distance is {np.mean(kpd_times)/np.mean(cd_times):.2f}x SLOWER")
