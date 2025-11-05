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

# Align arrays by matching positions
# k-mer positions start at window_size//2 due to centering
# CD positions start at 0 due to zero-padding
# We need to align them to the same x-axis
print(f"\nAligning arrays: k-mer has {len(kmer_values)} windows, CD has {len(cd_d2_values)} windows")
print(f"  k-mer first position: {kmer_positions_scaled[0]}, last: {kmer_positions_scaled[-1]}")
print(f"  CD first position: {cd_positions_scaled[0]}, last: {cd_positions_scaled[-1]}")

# Use k-mer positions as reference, and extract CD values at matching positions
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
print(f"  Aligned {len(kmer_values_aligned)} positions")

# Create comparison plot
print(f"\nCreating comparison plot...")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: k-mer Entropy
ax1.plot(kmer_positions_scaled, kmer_values, color='g', linewidth=1.5, label='k-mer Entropy', alpha=0.8)
ax1.set_xlabel('Sequence Position (Repeat Index)', fontsize=12, fontweight='bold')
ax1.set_ylabel('k-mer Entropy (Normalized)', fontsize=12, fontweight='bold')
ax1.set_title(f'k-mer Entropy Analysis\nWindow size: {window_size} repeats | {n_repeats} total repeats',
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, n_repeats)

# Add statistics text box
stats_text = f'Mean: {np.mean(kmer_values):.4f}\nStd: {np.std(kmer_values):.4f}\nWindows: {len(kmer_values)}'
ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Correlation Dimension (D2)
ax2.plot(cd_positions_scaled, cd_d2_values, color='b', linewidth=1.5, label='Correlation Dimension (D2)', alpha=0.8)
ax2.set_xlabel('Sequence Position (Repeat Index)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Correlation Dimension (D2)', fontsize=12, fontweight='bold')
ax2.set_title(f'Correlation Dimension Analysis\nWindow size: {window_size} repeats | {n_repeats} total repeats',
             fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, n_repeats)

# Add statistics text box
cd_stats_text = f'Mean: {np.mean(cd_d2_values):.4f}\nStd: {np.std(cd_d2_values):.4f}\nWindows: {len(cd_d2_values)}'
ax2.text(0.98, 0.98, cd_stats_text, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Plot 3: Overlay comparison (normalized to 0-1 range) using aligned arrays
# Debug: print ranges
print(f"\nNormalization info:")
print(f"  k-mer: min={np.min(kmer_values_aligned):.4f}, max={np.max(kmer_values_aligned):.4f}, range={np.max(kmer_values_aligned) - np.min(kmer_values_aligned):.4f}")
print(f"  CD D2: min={np.min(cd_d2_values_aligned):.4f}, max={np.max(cd_d2_values_aligned):.4f}, range={np.max(cd_d2_values_aligned) - np.min(cd_d2_values_aligned):.4f}")

# Handle edge case where range might be zero
kmer_range = np.max(kmer_values_aligned) - np.min(kmer_values_aligned)
cd_range = np.max(cd_d2_values_aligned) - np.min(cd_d2_values_aligned)

if kmer_range > 0:
    kmer_normalized = (kmer_values_aligned - np.min(kmer_values_aligned)) / kmer_range
else:
    kmer_normalized = np.zeros_like(kmer_values_aligned)

if cd_range > 0:
    cd_normalized = (cd_d2_values_aligned - np.min(cd_d2_values_aligned)) / cd_range
else:
    cd_normalized = np.zeros_like(cd_d2_values_aligned)

ax3.plot(kmer_positions_aligned, kmer_normalized, color='g', linewidth=1.5, label='k-mer Entropy (normalized)', alpha=0.8)
ax3.plot(kmer_positions_aligned, cd_normalized, color='b', linewidth=1.5, label='Correlation Dimension (normalized)', alpha=0.8)
ax3.set_xlabel('Sequence Position (Repeat Index)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Normalized Complexity (0-1)', fontsize=12, fontweight='bold')
ax3.set_title(f'Metric Comparison (Normalized)\nWindow size: {window_size} repeats | {n_repeats} total repeats',
             fontsize=14, fontweight='bold')
ax3.legend(fontsize=10, loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, n_repeats)

# Calculate correlation between aligned metrics
correlation = np.corrcoef(kmer_values_aligned, cd_d2_values_aligned)[0, 1]
comp_stats_text = f'Correlation: {correlation:.4f}\nN windows: {len(kmer_values_aligned)}'
ax3.text(0.98, 0.98, comp_stats_text, transform=ax3.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

plt.tight_layout()

# Save plot
output_file = 'output/kmer_vs_cd_comparison.png'
os.makedirs('output', exist_ok=True)
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Plot saved as {output_file}")

plt.show()
print(f"\nCompleted {input_file}")
print(f"\nCorrelation between k-mer entropy and D2: {correlation:.4f}")

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
