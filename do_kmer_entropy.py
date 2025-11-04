import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import sys
import os

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
        comp = calculate_kmer_entropy(window_seq, k=4)
        complexity_values.append(comp)

    return positions, complexity_values

def all_vs_all_identity_scipy(repeats, max_exact_size=1000, scale_factor=30):
    """
    Compute all vs all identity matrix using scipy's pdist with adaptive subsampling.

    Args:
        repeats: list of sequences (strings)
        max_exact_size: array size threshold for exact computation
        scale_factor: controls subsampling rate for large arrays

    Returns:
        tuple of (identity_matrix, subsample_every)
        - identity_matrix: numpy array of shape (n_repeats, n_repeats) with pairwise identities
        - subsample_every: subsampling interval (1 = no subsampling)
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
        # Exact computation
        seq_array = np.array([[ord(c) for c in seq] for seq in repeats], dtype=np.uint8)
        distances = pdist(seq_array, metric='hamming')
        identity_matrix = 1 - squareform(distances)
        subset_repeats = repeats  # No subsampling
    else:
        # Subsampled computation
        subset_indices = np.arange(0, n_repeats, subsample_every)
        subset_repeats = [repeats[i] for i in subset_indices]

        seq_array = np.array([[ord(c) for c in seq] for seq in subset_repeats], dtype=np.uint8)
        distances = pdist(seq_array, metric='hamming')
        identity_matrix = 1 - squareform(distances)

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

    # Extract all k-mers as a view (no copying)
    n_kmers = len(sequence_string) - k + 1

    kmer_counts = {}

    for i in range(n_kmers):
        kmer = sequence_string[i:i+k]
        kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1

    unique_kmers = list(kmer_counts.keys())
    counts = np.array(list(kmer_counts.values()), dtype="float")

    # Calculate Shannon entropy using vectorized operations
    total = counts.sum()
    probs = counts / total
    entropy = -np.sum(probs * np.log2(probs))

    # Normalize by maximum possible entropy for k-mers
    # Max entropy = log2(4^k) for DNA with 4 bases
    max_entropy = 2 * k  # log2(4^k) = k * log2(4) = k * 2
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return normalized_entropy

input_dir = os.listdir("output/fasta")
for f in input_dir[:100]:
    input_file = os.path.join("output", "fasta/", f)
    repeat_len = 178
    window_size = 100

    with open(input_file, "r") as file:
        contents = file.read().strip().split()[1:][0]

    n_repeats = int(len(contents) / repeat_len)
    repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]

    _, subsample_every, subsampled_repeats = all_vs_all_identity_scipy(repeats, scale_factor=30, max_exact_size=1000)

    kmer_positions, kmer_values = sliding_window_complexity_from_repeats(subsampled_repeats, window_size, method='kmer')

    kmer_positions_scaled = [pos * subsample_every for pos in kmer_positions]
    print(f"completed {input_file}")
