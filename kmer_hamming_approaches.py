import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

def all_vs_all_identity_with_numba(repeats, max_exact_size=1000, scale_factor=30):
    """
    Compute all vs all identity matrix using numba (fast) with adaptive subsampling.

    Args:
        repeats: list of sequences (strings)
        max_exact_size: array size threshold for exact computation
        scale_factor: controls subsampling rate for large arrays

    Returns:
        tuple of (identity_matrix, subsample_every, subset_repeats)
    """
    from src.censim.identity import all_vs_all_identity_numba

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
        identity_matrix = all_vs_all_identity_numba(repeats, max_exact_size=max_exact_size,
                                                     scale_factor=scale_factor, use_parallel=True)
        subset_repeats = repeats
    else:
        subset_indices = np.arange(0, n_repeats, subsample_every)
        subset_repeats = [repeats[i] for i in subset_indices]
        identity_matrix = all_vs_all_identity_numba(subset_repeats, max_exact_size=max_exact_size,
                                                     scale_factor=scale_factor, use_parallel=True)

    return identity_matrix.astype(np.float32), subsample_every, subset_repeats

def precompute_kmer_profiles(repeats, k=4):
    """
    Precompute k-mer presence/absence profiles for all repeats.

    Args:
        repeats: list of repeat sequences
        k: k-mer size

    Returns:
        profile_matrix: numpy array (n_repeats x n_possible_kmers) of binary profiles
        n_kmers_total: total number of possible k-mers
    """
    # Generate all possible k-mers
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

    print(f"    Precomputing k-mer presence/absence for {len(repeats)} repeats...")
    precompute_start = time.time()

    profile_matrix = np.zeros((len(repeats), n_kmers_total), dtype=np.uint8)

    for rep_idx, repeat in enumerate(repeats):
        if len(repeat) < k:
            continue
        n_kmers = len(repeat) - k + 1  # Overlapping kmers
        kmer_list = [repeat[i:i+k] for i in range(n_kmers)]

        for kmer in kmer_list:
            if kmer in kmer_to_idx:
                profile_matrix[rep_idx, kmer_to_idx[kmer]] = 1

    precompute_time = time.time() - precompute_start
    print(f"    Precomputation took {precompute_time:.4f}s")

    return profile_matrix, n_kmers_total

def approach1_consecutive_hamming(profile_matrix, n_kmers_total, window_size=100):
    """
    Approach 1: Sequential/Consecutive Hamming Distance
    Calculate Hamming distance between adjacent repeats, average within windows.
    """
    start_time = time.time()

    N = profile_matrix.shape[0]
    positions = []
    diversity_values = []

    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        window_profiles = profile_matrix[i:i+window_size]

        # Calculate consecutive distances
        consecutive_distances = []
        for j in range(window_size - 1):
            hamming_dist = np.sum(window_profiles[j] != window_profiles[j+1])
            consecutive_distances.append(hamming_dist)

        avg_distance = np.mean(consecutive_distances) / n_kmers_total if consecutive_distances else 0.0
        diversity_values.append(avg_distance)

    elapsed_time = time.time() - start_time
    print(f"  Consecutive Hamming calculation time: {elapsed_time:.6f} seconds")

    return positions, diversity_values

def approach2_allpairs_hamming(profile_matrix, n_kmers_total, window_size=100):
    """
    Approach 2: All-Pairwise Hamming Distance
    Calculate mean pairwise Hamming distance between all pairs in window.
    """
    start_time = time.time()

    N = profile_matrix.shape[0]
    positions = []
    diversity_values = []

    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        window_profiles = profile_matrix[i:i+window_size]

        # Calculate all pairwise distances
        pairwise_distances = []
        for j in range(window_size):
            for k in range(j + 1, window_size):
                hamming_dist = np.sum(window_profiles[j] != window_profiles[k])
                pairwise_distances.append(hamming_dist)

        avg_distance = np.mean(pairwise_distances) / n_kmers_total if pairwise_distances else 0.0
        diversity_values.append(avg_distance)

    elapsed_time = time.time() - start_time
    print(f"  All-pairwise Hamming calculation time: {elapsed_time:.6f} seconds")

    return positions, diversity_values

def approach3_distance_variance(profile_matrix, n_kmers_total, window_size=100):
    """
    Approach 3: Distance Variance/Heterogeneity
    Measure variance of all pairwise distances.
    """
    start_time = time.time()

    N = profile_matrix.shape[0]
    positions = []
    diversity_values = []

    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        window_profiles = profile_matrix[i:i+window_size]

        # Calculate all pairwise distances
        pairwise_distances = []
        for j in range(window_size):
            for k in range(j + 1, window_size):
                hamming_dist = np.sum(window_profiles[j] != window_profiles[k])
                pairwise_distances.append(hamming_dist / n_kmers_total)

        variance = np.var(pairwise_distances) if pairwise_distances else 0.0
        diversity_values.append(variance)

    elapsed_time = time.time() - start_time
    print(f"  Distance variance calculation time: {elapsed_time:.6f} seconds")

    return positions, diversity_values

def approach4_center_to_edges(profile_matrix, n_kmers_total, window_size=100):
    """
    Approach 4: Center-to-Edges Distance
    Compare center repeat to all others in window.
    """
    start_time = time.time()

    N = profile_matrix.shape[0]
    positions = []
    diversity_values = []

    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        window_profiles = profile_matrix[i:i+window_size]
        center_idx = window_size // 2
        center_profile = window_profiles[center_idx]

        # Calculate distances from center to all others
        distances = []
        for j in range(window_size):
            if j != center_idx:
                hamming_dist = np.sum(center_profile != window_profiles[j])
                distances.append(hamming_dist)

        avg_distance = np.mean(distances) / n_kmers_total if distances else 0.0
        diversity_values.append(avg_distance)

    elapsed_time = time.time() - start_time
    print(f"  Center-to-edges calculation time: {elapsed_time:.6f} seconds")

    return positions, diversity_values

def approach5_max_spanning(profile_matrix, n_kmers_total, window_size=100):
    """
    Approach 5: Maximum Spanning Distance
    Find the maximum pairwise distance in each window.
    """
    start_time = time.time()

    N = profile_matrix.shape[0]
    positions = []
    diversity_values = []

    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        window_profiles = profile_matrix[i:i+window_size]

        # Find maximum pairwise distance
        max_distance = 0
        for j in range(window_size):
            for k in range(j + 1, window_size):
                hamming_dist = np.sum(window_profiles[j] != window_profiles[k])
                max_distance = max(max_distance, hamming_dist)

        normalized_distance = max_distance / n_kmers_total
        diversity_values.append(normalized_distance)

    elapsed_time = time.time() - start_time
    print(f"  Max spanning distance calculation time: {elapsed_time:.6f} seconds")

    return positions, diversity_values

def approach6_distance_gradient(profile_matrix, n_kmers_total, window_size=100):
    """
    Approach 6: Distance Gradient/Slope
    Measure trend in consecutive distances (are repeats becoming more/less similar?).
    """
    start_time = time.time()

    N = profile_matrix.shape[0]
    positions = []
    diversity_values = []

    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        window_profiles = profile_matrix[i:i+window_size]

        # Calculate consecutive distances
        consecutive_distances = []
        for j in range(window_size - 1):
            hamming_dist = np.sum(window_profiles[j] != window_profiles[j+1])
            consecutive_distances.append(hamming_dist / n_kmers_total)

        # Calculate gradient (slope) - positive means increasing diversity
        if len(consecutive_distances) > 1:
            x = np.arange(len(consecutive_distances))
            slope = np.polyfit(x, consecutive_distances, 1)[0]
        else:
            slope = 0.0

        diversity_values.append(abs(slope))  # Use absolute value to measure change magnitude

    elapsed_time = time.time() - start_time
    print(f"  Distance gradient calculation time: {elapsed_time:.6f} seconds")

    return positions, diversity_values

# Main execution
input_file = "data/2191000generation.out.fa"
repeat_len = 178
window_size = 100

print(f"Loading sequence data from {input_file}...")
with open(input_file, "r") as file:
    contents = file.read().strip()

n_repeats = int(len(contents) / repeat_len)
repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]
print(f"Loaded {n_repeats} repeats of length {repeat_len}bp")

print(f"\nSubsampling repeats...")
_, subsample_every, subsampled_repeats = all_vs_all_identity_with_numba(repeats, scale_factor=30, max_exact_size=1000)

# Precompute k-mer profiles once
print(f"\nPrecomputing k-mer profiles...")
profile_matrix, n_kmers_total = precompute_kmer_profiles(subsampled_repeats, k=4)

# Calculate all 6 approaches
print(f"\nCalculating approach 1: Consecutive Hamming...")
pos1, vals1 = approach1_consecutive_hamming(profile_matrix, n_kmers_total, window_size)

print(f"\nCalculating approach 2: All-pairwise Hamming...")
pos2, vals2 = approach2_allpairs_hamming(profile_matrix, n_kmers_total, window_size)

print(f"\nCalculating approach 3: Distance Variance...")
pos3, vals3 = approach3_distance_variance(profile_matrix, n_kmers_total, window_size)

print(f"\nCalculating approach 4: Center-to-Edges...")
pos4, vals4 = approach4_center_to_edges(profile_matrix, n_kmers_total, window_size)

print(f"\nCalculating approach 5: Maximum Spanning Distance...")
pos5, vals5 = approach5_max_spanning(profile_matrix, n_kmers_total, window_size)

print(f"\nCalculating approach 6: Distance Gradient...")
pos6, vals6 = approach6_distance_gradient(profile_matrix, n_kmers_total, window_size)

# Scale positions back to original repeat indices
pos1_scaled = [pos * subsample_every for pos in pos1]
pos2_scaled = [pos * subsample_every for pos in pos2]
pos3_scaled = [pos * subsample_every for pos in pos3]
pos4_scaled = [pos * subsample_every for pos in pos4]
pos5_scaled = [pos * subsample_every for pos in pos5]
pos6_scaled = [pos * subsample_every for pos in pos6]

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

# Plot 1: Consecutive Hamming
ax1.plot(pos1_scaled, normalize(vals1), color='g', linewidth=1.5)
ax1.set_xlabel('Repeat Index', fontsize=10)
ax1.set_ylabel('Normalized Value', fontsize=10)
ax1.set_title('Consecutive Hamming Distance', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, n_repeats)

# Plot 2: All-pairwise Hamming
ax2.plot(pos2_scaled, normalize(vals2), color='b', linewidth=1.5)
ax2.set_xlabel('Repeat Index', fontsize=10)
ax2.set_ylabel('Normalized Value', fontsize=10)
ax2.set_title('All-Pairwise Hamming Distance', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, n_repeats)

# Plot 3: Distance Variance
ax3.plot(pos3_scaled, normalize(vals3), color='r', linewidth=1.5)
ax3.set_xlabel('Repeat Index', fontsize=10)
ax3.set_ylabel('Normalized Value', fontsize=10)
ax3.set_title('Distance Variance', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, n_repeats)

# Plot 4: Center-to-Edges
ax4.plot(pos4_scaled, normalize(vals4), color='purple', linewidth=1.5)
ax4.set_xlabel('Repeat Index', fontsize=10)
ax4.set_ylabel('Normalized Value', fontsize=10)
ax4.set_title('Center-to-Edges Distance', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, n_repeats)

# Plot 5: Maximum Spanning Distance
ax5.plot(pos5_scaled, normalize(vals5), color='orange', linewidth=1.5)
ax5.set_xlabel('Repeat Index', fontsize=10)
ax5.set_ylabel('Normalized Value', fontsize=10)
ax5.set_title('Maximum Spanning Distance', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, n_repeats)

# Plot 6: Distance Gradient
ax6.plot(pos6_scaled, normalize(vals6), color='cyan', linewidth=1.5)
ax6.set_xlabel('Repeat Index', fontsize=10)
ax6.set_ylabel('Normalized Value', fontsize=10)
ax6.set_title('Distance Gradient (abs slope)', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, n_repeats)

plt.tight_layout()

# Save plot
output_file = 'output/kmer_hamming_approaches.png'
os.makedirs('output', exist_ok=True)
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Plot saved as {output_file}")

plt.show()
print(f"\nCompleted {input_file}")
print(f"\n=== Summary ===")
print(f"All 6 k-mer Hamming distance approaches plotted:")
print(f"  1. Consecutive Hamming - adjacent repeat distances")
print(f"  2. All-Pairwise Hamming - mean of all pairs")
print(f"  3. Distance Variance - heterogeneity of pairwise distances")
print(f"  4. Center-to-Edges - center repeat vs others")
print(f"  5. Maximum Spanning - largest distance in window")
print(f"  6. Distance Gradient - trend in consecutive distances")
