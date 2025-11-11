import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from src.censim.identity import all_vs_all_identity_numba
from src.censim.correlation_dimension import (
    hamming_distance_matrix,
    sliding_window_local_correlation,
    estimate_D2_from_C_r_batch,
    choose_meaningful_radii
)

def sliding_window_complexity_from_repeats(repeats, window_size=100, method='kmer'):
    """
    Calculate complexity using overlapping windows with non-overlapping k-mers.

    Args:
        repeats: list of repeat sequences (already subsampled)
        window_size: number of repeats per window
        method: 'kmer' or 'lzma'

    Returns:
        positions: center positions of windows (in repeat indices)
        complexity_values: complexity at each window position
    """
    start_time = time.time()

    N = len(repeats)

    positions = []
    complexity_values = []

    # Overlapping windows across repeats (step size = 1)
    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        # Extract window of repeats and concatenate
        window_repeats = repeats[i:i+window_size]
        window_seq = ''.join(window_repeats)

        # Calculate complexity for this window (uses non-overlapping k-mers internally)
        comp = calculate_kmer_entropy(window_seq, k=4)
        complexity_values.append(comp)

    elapsed_time = time.time() - start_time
    print(f"Total k-mer entropy calculation time: {elapsed_time:.6f} seconds ({len(complexity_values)} windows)")

    return positions, complexity_values

def all_vs_all_identity_with_numba(repeats, max_exact_size=1000, scale_factor=30):
    """
    Compute all vs all identity matrix using numba (fast) with adaptive subsampling.

    Args:
        repeats: list of sequences (strings)
        max_exact_size: array size threshold for exact computation
        scale_factor: controls subsampling rate for large arrays

    Returns:
        tuple of (identity_matrix, subsample_every, subset_repeats)
        - identity_matrix: numpy array of shape (n_repeats, n_repeats) with pairwise identities
        - subsample_every: subsampling interval (1 = no subsampling)
        - subset_repeats: list of repeats used (subsampled if applicable)
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
        # Exact computation using numba
        identity_matrix = all_vs_all_identity_numba(repeats, max_exact_size=max_exact_size,
                                                     scale_factor=scale_factor, use_parallel=True)
        subset_repeats = repeats  # No subsampling
    else:
        # Subsampled computation
        subset_indices = np.arange(0, n_repeats, subsample_every)
        subset_repeats = [repeats[i] for i in subset_indices]
        identity_matrix = all_vs_all_identity_numba(subset_repeats, max_exact_size=max_exact_size,
                                                     scale_factor=scale_factor, use_parallel=True)

    return identity_matrix.astype(np.float32), subsample_every, subset_repeats

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

    # Extract non-overlapping k-mers
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

def sliding_window_repeat_diversity(identity_matrix, window_size=100):
    """
    Calculate diversity based on average pairwise distance in sliding windows.

    For each window, calculates the mean pairwise distance (1 - identity) between
    all pairs of repeats in that window. This produces a smooth, continuous measure
    of local diversity.

    Args:
        identity_matrix: pairwise identity matrix for the repeats
        window_size: number of repeats per window

    Returns:
        positions: center positions of windows (in repeat indices)
        diversity_values: average pairwise distance at each window position
    """
    start_time = time.time()

    N = identity_matrix.shape[0]
    positions = []
    diversity_values = []

    # Sliding window across repeats
    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        # Extract submatrix for this window
        window_matrix = identity_matrix[i:i+window_size, i:i+window_size]

        # Calculate average pairwise distance (1 - identity)
        # Only use upper triangle to avoid counting pairs twice and diagonal
        upper_tri_indices = np.triu_indices(window_size, k=1)
        pairwise_identities = window_matrix[upper_tri_indices]
        pairwise_distances = 1.0 - pairwise_identities

        avg_distance = np.mean(pairwise_distances)
        diversity_values.append(avg_distance)

    elapsed_time = time.time() - start_time
    print(f"  Repeat diversity calculation time: {elapsed_time:.6f} seconds ({len(diversity_values)} windows)")

    return positions, diversity_values

def sliding_window_kmer_entropy_variance(repeats, window_size=100, k=4):
    """
    Calculate variance of per-repeat k-mer entropy within sliding windows.

    Precomputes k-mer entropy for each repeat, then measures variance within windows.
    High variance = diverse repeats, low variance = similar repeats.
    O(n) precompute + O(window_size) per window.

    Args:
        repeats: list of repeat sequences
        window_size: number of repeats per window
        k: k-mer size

    Returns:
        positions: center positions of windows
        variance_values: variance of k-mer entropy at each position
    """
    start_time = time.time()

    # Precompute k-mer entropy for each repeat
    per_repeat_entropy = []
    for repeat in repeats:
        entropy = calculate_kmer_entropy(repeat, k=k)
        per_repeat_entropy.append(entropy)

    per_repeat_entropy = np.array(per_repeat_entropy)

    N = len(repeats)
    positions = []
    variance_values = []

    # Sliding window
    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        window_entropies = per_repeat_entropy[i:i+window_size]
        variance = np.var(window_entropies)
        variance_values.append(variance)

    elapsed_time = time.time() - start_time
    print(f"  K-mer entropy variance calculation time: {elapsed_time:.6f} seconds ({len(variance_values)} windows)")

    return positions, variance_values

def sliding_window_hash_diversity(repeats, window_size=100, k=4):
    """
    Calculate all-pairwise Hamming distance between k-mer profiles within sliding windows.

    Similar to repeat diversity (plot 3) but uses k-mer composition instead of full sequence.
    Calculates mean pairwise Hamming distance between all k-mer profiles in each window.

    Args:
        repeats: list of repeat sequences
        window_size: number of repeats per window
        k: k-mer size

    Returns:
        positions: center positions of windows
        diversity_values: mean pairwise Hamming distance
    """
    start_time = time.time()

    # Generate all possible k-mers for DNA (4^k possibilities)
    bases = ['A', 'C', 'G', 'T']
    all_possible_kmers = []

    def generate_kmers(prefix, k):
        if k == 0:
            all_possible_kmers.append(prefix)
            return
        for base in bases:
            generate_kmers(prefix + base, k - 1)

    generate_kmers('', k)
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_possible_kmers)}
    n_kmers_total = len(all_possible_kmers)

    # Precompute all k-mer presence/absence as binary vectors
    print(f"    Precomputing k-mer presence/absence for {len(repeats)} repeats...")
    precompute_start = time.time()

    profile_matrix = np.zeros((len(repeats), n_kmers_total), dtype=np.uint8)

    for rep_idx, repeat in enumerate(repeats):
        if len(repeat) < k:
            continue
        n_kmers = len(repeat) - k + 1  # Overlapping kmers
        kmer_list = [repeat[i:i+k] for i in range(n_kmers)]

        for kmer in kmer_list:
            if kmer in kmer_to_idx:  # Only mark valid k-mers (all ACGT)
                profile_matrix[rep_idx, kmer_to_idx[kmer]] = 1  # Mark as present

    precompute_time = time.time() - precompute_start
    print(f"    Precomputation took {precompute_time:.4f}s")

    N = len(repeats)
    positions = []
    diversity_values = []

    # Sliding window
    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        # Get binary profile vectors for this window
        window_profiles = profile_matrix[i:i+window_size]

        # Calculate all pairwise Hamming distances
        # For each pair (j, k), Hamming distance = number of positions where bits differ
        pairwise_distances = []
        for j in range(window_size):
            for k in range(j + 1, window_size):
                hamming_dist = np.sum(window_profiles[j] != window_profiles[k])
                pairwise_distances.append(hamming_dist)

        # Average pairwise distance, normalized by total number of k-mers
        avg_distance = np.mean(pairwise_distances) / n_kmers_total if pairwise_distances else 0.0
        diversity_values.append(avg_distance)

    elapsed_time = time.time() - start_time
    print(f"  K-mer pairwise Hamming distance calculation time: {elapsed_time:.6f} seconds ({len(diversity_values)} windows)")

    return positions, diversity_values

def sliding_window_kmer_profile_distance(repeats, window_size=100, k=4):
    """
    Calculate average distance from k-mer consensus within sliding windows.

    Uses binary presence/absence vectors (bit vectors) instead of frequency counts.
    This is faster and measures k-mer diversity rather than frequency distribution.

    Args:
        repeats: list of repeat sequences
        window_size: number of repeats per window
        k: k-mer size

    Returns:
        positions: center positions of windows
        diversity_values: average distance from consensus
    """
    start_time = time.time()

    # Generate all possible k-mers for DNA (4^k possibilities)
    bases = ['A', 'C', 'G', 'T']
    all_possible_kmers = []

    def generate_kmers(prefix, k):
        if k == 0:
            all_possible_kmers.append(prefix)
            return
        for base in bases:
            generate_kmers(prefix + base, k - 1)

    generate_kmers('', k)
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_possible_kmers)}
    n_kmers_total = len(all_possible_kmers)

    # Precompute all k-mer presence/absence as binary vectors
    print(f"    Precomputing k-mer presence/absence for {len(repeats)} repeats...")
    precompute_start = time.time()

    # Use uint8 to store binary (0 or 1) - more memory efficient than float
    profile_matrix = np.zeros((len(repeats), n_kmers_total), dtype=np.uint8)

    for rep_idx, repeat in enumerate(repeats):
        if len(repeat) < k:
            continue
        n_kmers = len(repeat) // k
        kmer_list = [repeat[i*k:(i+1)*k] for i in range(n_kmers)]

        for kmer in kmer_list:
            if kmer in kmer_to_idx:  # Only mark valid k-mers (all ACGT)
                profile_matrix[rep_idx, kmer_to_idx[kmer]] = 1  # Mark as present

    precompute_time = time.time() - precompute_start
    print(f"    Precomputation took {precompute_time:.4f}s")

    N = len(repeats)
    positions = []
    diversity_values = []

    # Sliding window using vectorized operations
    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        # Get binary profile vectors for this window
        window_profiles = profile_matrix[i:i+window_size].astype(np.float32)

        # Consensus is proportion of repeats with each k-mer (ranges 0-1)
        consensus = window_profiles.mean(axis=0)

        # Calculate Hamming-like distances from consensus
        # Distance = how different each repeat is from the consensus
        distances = np.abs(window_profiles - consensus).sum(axis=1) / n_kmers_total
        avg_distance = distances.mean()
        diversity_values.append(avg_distance)

    elapsed_time = time.time() - start_time
    print(f"  K-mer profile distance calculation time: {elapsed_time:.6f} seconds ({len(diversity_values)} windows)")

    return positions, diversity_values

def sliding_window_kmer_profile_mode_distance(repeats, window_size=100, k=4):
    """
    Calculate sum of Hamming distances from the most common k-mer profile.

    Keeps the matrix as binary (uint8) and finds the most common exact profile
    in each window, then computes the sum of Hamming distances from this mode.

    Args:
        repeats: list of repeat sequences
        window_size: number of repeats per window
        k: k-mer size

    Returns:
        positions: center positions of windows
        diversity_values: sum of Hamming distances from mode
    """
    start_time = time.time()

    # Generate all possible k-mers for DNA (4^k possibilities)
    bases = ['A', 'C', 'G', 'T']
    all_possible_kmers = []

    def generate_kmers(prefix, k):
        if k == 0:
            all_possible_kmers.append(prefix)
            return
        for base in bases:
            generate_kmers(prefix + base, k - 1)

    generate_kmers('', k)
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_possible_kmers)}
    n_kmers_total = len(all_possible_kmers)

    # Precompute all k-mer presence/absence as binary vectors
    print(f"    Precomputing k-mer presence/absence for {len(repeats)} repeats...")
    precompute_start = time.time()

    # Use uint8 to store binary (0 or 1) - keep as binary throughout
    profile_matrix = np.zeros((len(repeats), n_kmers_total), dtype=np.uint8)

    for rep_idx, repeat in enumerate(repeats):
        if len(repeat) < k:
            continue
        n_kmers = len(repeat) - k + 1  # Overlapping kmers
        kmer_list = [repeat[i:i+k] for i in range(n_kmers)]

        for kmer in kmer_list:
            if kmer in kmer_to_idx:  # Only mark valid k-mers (all ACGT)
                profile_matrix[rep_idx, kmer_to_idx[kmer]] = 1  # Mark as present

    precompute_time = time.time() - precompute_start
    print(f"    Precomputation took {precompute_time:.4f}s")

    N = len(repeats)
    positions = []
    diversity_values = []

    # Sliding window - keep everything as binary
    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        # Get binary profile vectors for this window (keep as uint8)
        window_profiles = profile_matrix[i:i+window_size]

        # Find the most common profile (mode)
        # Convert each row to a tuple so we can count occurrences
        profile_tuples = [tuple(row) for row in window_profiles]

        # Count occurrences of each unique profile
        from collections import Counter
        profile_counts = Counter(profile_tuples)

        # Get the most common profile
        mode_profile_tuple = profile_counts.most_common(1)[0][0]
        mode_profile = np.array(mode_profile_tuple, dtype=np.uint8)

        # Calculate Hamming distance from mode for each profile
        # Hamming distance = number of positions where bits differ
        hamming_distances = np.sum(window_profiles != mode_profile, axis=1)

        # Sum of all Hamming distances (can normalize if desired)
        total_distance = np.sum(hamming_distances)

        # Normalize by window size and number of k-mers for comparability
        normalized_distance = float(total_distance) / (window_size * n_kmers_total)
        diversity_values.append(normalized_distance)

    elapsed_time = time.time() - start_time
    print(f"  K-mer profile mode distance calculation time: {elapsed_time:.6f} seconds ({len(diversity_values)} windows)")

    return positions, diversity_values

input_file = "data/2191000generation.out.fa"
repeat_len = 178
window_size = 100

print(f"Loading sequence data from {input_file}...")
with open(input_file, "r") as file:
    contents = file.read().strip()

n_repeats = int(len(contents) / repeat_len)
repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]
print(f"Loaded {n_repeats} repeats of length {repeat_len}bp")

print(f"\nComputing identity matrix and subsampling...")
identity_matrix, subsample_every, subsampled_repeats = all_vs_all_identity_with_numba(repeats, scale_factor=30, max_exact_size=1000)

print(f"\nCalculating k-mer entropy...")
kmer_positions, kmer_values = sliding_window_complexity_from_repeats(subsampled_repeats, window_size, method='kmer')

kmer_positions_scaled = [pos * subsample_every for pos in kmer_positions]
print(f"\nMean k-mer entropy: {np.mean(kmer_values):.4f}")
print(f"Std k-mer entropy: {np.std(kmer_values):.4f}")

# Calculate correlation dimension
print(f"\nCalculating correlation dimension...")
distance_matrix = hamming_distance_matrix(identity_matrix)
r_values = choose_meaningful_radii(distance_matrix, n_radii=50, r_min=0.01, r_max=0.5)
print(f"Using {len(r_values)} radii from {r_values[0]:.4f} to {r_values[-1]:.4f}")

cd_positions, cd_correlations = sliding_window_local_correlation(distance_matrix, window_size, r_values, use_parallel=True)
cd_d2_values = estimate_D2_from_C_r_batch(r_values, cd_correlations, fit_window=(0.05, 0.3))

cd_positions_scaled = [pos * subsample_every for pos in cd_positions]
print(f"\nMean correlation dimension (D2): {np.mean(cd_d2_values):.4f}")
print(f"Std correlation dimension (D2): {np.std(cd_d2_values):.4f}")

# Calculate repeat diversity (average pairwise distance)
print(f"\nCalculating repeat diversity (average pairwise distance)...")
rd_positions, rd_values = sliding_window_repeat_diversity(identity_matrix, window_size=100)
rd_positions_scaled = [pos * subsample_every for pos in rd_positions]
print(f"  Mean repeat diversity: {np.mean(rd_values):.4f}")
print(f"  Std repeat diversity: {np.std(rd_values):.4f}")

# Calculate k-mer profile mode distance
print(f"\nCalculating k-mer profile mode distance...")
kpmd_positions, kpmd_values = sliding_window_kmer_profile_mode_distance(subsampled_repeats, window_size=100, k=4)
kpmd_positions_scaled = [pos * subsample_every for pos in kpmd_positions]
print(f"  Mean k-mer profile mode distance: {np.mean(kpmd_values):.4f}")
print(f"  Std k-mer profile mode distance: {np.std(kpmd_values):.4f}")

# Calculate k-mer pairwise Hamming distance
print(f"\nCalculating k-mer pairwise Hamming distance...")
hd_positions, hd_values = sliding_window_hash_diversity(subsampled_repeats, window_size=100, k=4)
hd_positions_scaled = [pos * subsample_every for pos in hd_positions]
print(f"  Mean k-mer pairwise Hamming distance: {np.mean(hd_values):.4f}")
print(f"  Std k-mer pairwise Hamming distance: {np.std(hd_values):.4f}")

# Calculate k-mer profile distance
print(f"\nCalculating k-mer profile distance...")
kpd_positions, kpd_values = sliding_window_kmer_profile_distance(subsampled_repeats, window_size=100, k=4)
kpd_positions_scaled = [pos * subsample_every for pos in kpd_positions]
print(f"  Mean k-mer profile distance: {np.mean(kpd_values):.4f}")
print(f"  Std k-mer profile distance: {np.std(kpd_values):.4f}")

# Align arrays by matching positions
# k-mer positions start at window_size//2 due to centering
# CD positions start at 0 due to zero-padding
# We need to align them to the same x-axis
print(f"\nAligning arrays: k-mer has {len(kmer_values)} windows, CD has {len(cd_d2_values)} windows, RD has {len(rd_values)} windows")
print(f"  k-mer first position: {kmer_positions_scaled[0]}, last: {kmer_positions_scaled[-1]}")
print(f"  CD first position: {cd_positions_scaled[0]}, last: {cd_positions_scaled[-1]}")
print(f"  RD first position: {rd_positions_scaled[0]}, last: {rd_positions_scaled[-1]}")

# Use k-mer positions as reference, and extract CD and RD values at matching positions
kmer_values_aligned = np.array(kmer_values)
kmer_positions_aligned = np.array(kmer_positions_scaled)

# Find CD indices that match k-mer positions
cd_d2_values_list = list(cd_d2_values)
cd_positions_list = list(cd_positions_scaled)

# Extract CD values at positions matching k-mer
cd_d2_values_aligned = []
for kpos in kmer_positions_aligned:
    # Find closest CD position
    idx = min(range(len(cd_positions_list)), key=lambda i: abs(cd_positions_list[i] - kpos))
    cd_d2_values_aligned.append(cd_d2_values_list[idx])

cd_d2_values_aligned = np.array(cd_d2_values_aligned)

# Extract all new metrics at positions matching k-mer
rd_values_aligned = []
kpmd_values_aligned = []
hd_values_aligned = []
kpd_values_aligned = []

for kpos in kmer_positions_aligned:
    # Repeat Diversity
    idx = min(range(len(rd_positions_scaled)), key=lambda i: abs(rd_positions_scaled[i] - kpos))
    rd_values_aligned.append(rd_values[idx])

    # K-mer Profile Mode Distance
    idx = min(range(len(kpmd_positions_scaled)), key=lambda i: abs(kpmd_positions_scaled[i] - kpos))
    kpmd_values_aligned.append(kpmd_values[idx])

    # Hash Diversity
    idx = min(range(len(hd_positions_scaled)), key=lambda i: abs(hd_positions_scaled[i] - kpos))
    hd_values_aligned.append(hd_values[idx])

    # K-mer Profile Distance
    idx = min(range(len(kpd_positions_scaled)), key=lambda i: abs(kpd_positions_scaled[i] - kpos))
    kpd_values_aligned.append(kpd_values[idx])

rd_values_aligned = np.array(rd_values_aligned)
kpmd_values_aligned = np.array(kpmd_values_aligned)
hd_values_aligned = np.array(hd_values_aligned)
kpd_values_aligned = np.array(kpd_values_aligned)

print(f"  Aligned {len(kmer_values_aligned)} positions")

# Create comparison plot
print(f"\nCreating comparison plot...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

# Helper function to normalize
def normalize(values):
    vmin, vmax = np.min(values), np.max(values)
    if vmax - vmin > 0:
        return (values - vmin) / (vmax - vmin)
    return np.zeros_like(values)

# Plot 1: k-mer Entropy
ax1.plot(kmer_positions_aligned, normalize(kmer_values_aligned), color='g', linewidth=1.5)
ax1.set_xlabel('Repeat Index', fontsize=10)
ax1.set_ylabel('Normalized Value', fontsize=10)
ax1.set_title('k-mer Entropy', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, n_repeats)

# Plot 2: Correlation Dimension
ax2.plot(kmer_positions_aligned, normalize(cd_d2_values_aligned), color='b', linewidth=1.5)
ax2.set_xlabel('Repeat Index', fontsize=10)
ax2.set_ylabel('Normalized Value', fontsize=10)
ax2.set_title('Correlation Dimension (D2)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, n_repeats)

# Plot 3: Repeat Diversity (with identity matrix)
ax3.plot(kmer_positions_aligned, normalize(rd_values_aligned), color='r', linewidth=1.5)
ax3.set_xlabel('Repeat Index', fontsize=10)
ax3.set_ylabel('Normalized Value', fontsize=10)
ax3.set_title('Repeat Diversity (w/ identity matrix)', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, n_repeats)

# Plot 4: K-mer Profile Mode Distance
ax4.plot(kmer_positions_aligned, normalize(kpmd_values_aligned), color='purple', linewidth=1.5)
ax4.set_xlabel('Repeat Index', fontsize=10)
ax4.set_ylabel('Normalized Value', fontsize=10)
ax4.set_title('K-mer Profile Mode Distance', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, n_repeats)

# Plot 5: K-mer Pairwise Hamming Distance
ax5.plot(kmer_positions_aligned, normalize(hd_values_aligned), color='orange', linewidth=1.5)
ax5.set_xlabel('Repeat Index', fontsize=10)
ax5.set_ylabel('Normalized Value', fontsize=10)
ax5.set_title('K-mer Pairwise Hamming Distance', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, n_repeats)

# Plot 6: K-mer Profile Distance
ax6.plot(kmer_positions_aligned, normalize(kpd_values_aligned), color='cyan', linewidth=1.5)
ax6.set_xlabel('Repeat Index', fontsize=10)
ax6.set_ylabel('Normalized Value', fontsize=10)
ax6.set_title('K-mer Profile Distance', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, n_repeats)

plt.tight_layout()

# Save plot
output_file = 'output/kmer_vs_cd_comparison.png'
os.makedirs('output', exist_ok=True)
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Plot saved as {output_file}")

plt.show()
print(f"\nCompleted {input_file}")
print(f"\n=== Summary ===")
print(f"All 6 metrics plotted:")
print(f"  1. k-mer Entropy (baseline)")
print(f"  2. Correlation Dimension (D2) - uses identity matrix")
print(f"  3. Repeat Diversity - uses identity matrix")
print(f"  4. K-mer Profile Mode Distance - NO identity matrix, uses mode + Hamming")
print(f"  5. K-mer Pairwise Hamming Distance - NO identity matrix, all-pairwise comparison")
print(f"  6. K-mer Profile Distance - NO identity matrix")

#input_dir = os.listdir("output/fasta")
#for f in input_dir[:100]:
#    input_file = os.path.join("output", "fasta/", f)
#    repeat_len = 178
#    window_size = 100
#
#    with open(input_file, "r") as file:
#        contents = file.read().strip().split()[1:][0]
#
#    n_repeats = int(len(contents) / repeat_len)
#    repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]
#
#    _, subsample_every, subsampled_repeats = all_vs_all_identity_scipy(repeats, scale_factor=30, max_exact_size=1000)
#
#    kmer_positions, kmer_values = sliding_window_complexity_from_repeats(subsampled_repeats, window_size, method='kmer')
#
#    kmer_positions_scaled = [pos * subsample_every for pos in kmer_positions]
#    print(f"completed {input_file}")
