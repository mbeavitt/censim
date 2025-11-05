#!/usr/bin/env python3
"""
Profile k-mer entropy calculation to understand performance bottleneck.

This script:
1. Loads intermediate data (subsampled repeats)
2. Saves it to a file for reproducibility
3. Tests multiple k-mer implementations
4. Plots timing comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter

# Load the data from the original script
def load_subsampled_repeats(input_file="./data/2191000generation.out.fa"):
    """Load and subsample repeats exactly as the comparison script does."""
    repeat_len = 178

    with open(input_file, "r") as file:
        contents = file.read().strip()

    n_repeats = int(len(contents) / repeat_len)
    repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]

    # Apply same subsampling as the comparison script
    import math
    if n_repeats <= 1000:
        subsample_every = 1
        subsampled_repeats = repeats
    else:
        subsample_every = max(1, int(math.sqrt(n_repeats / 30)))
        subset_indices = np.arange(0, n_repeats, subsample_every)
        subsampled_repeats = [repeats[i] for i in subset_indices]

    print(f"Loaded {n_repeats} repeats")
    print(f"Subsampled to {len(subsampled_repeats)} repeats (every {subsample_every}th)")

    return subsampled_repeats, subsample_every


# Original implementation (from the script)
def calculate_kmer_entropy_original(sequence_string, k=4):
    """Original implementation using np.unique on string list."""
    if len(sequence_string) < k:
        return 0.0

    n_kmers = len(sequence_string) - k + 1

    # BOTTLENECK: This is slow!
    kmer_list = [sequence_string[i:i+k] for i in range(n_kmers)]
    unique_kmers, counts = np.unique(kmer_list, return_counts=True)

    total = counts.sum()
    probs = counts / total
    entropy = -np.sum(probs * np.log2(probs))

    max_entropy = 2 * k
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return normalized_entropy


# Optimized implementation using Counter
def calculate_kmer_entropy_counter(sequence_string, k=4):
    """Optimized implementation using Counter."""
    if len(sequence_string) < k:
        return 0.0

    n_kmers = len(sequence_string) - k + 1

    # Use Counter - much faster for counting
    kmer_counts = Counter(sequence_string[i:i+k] for i in range(n_kmers))

    # Calculate entropy
    counts = np.array(list(kmer_counts.values()))
    total = counts.sum()
    probs = counts / total
    entropy = -np.sum(probs * np.log2(probs))

    max_entropy = 2 * k
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return normalized_entropy


# Even more optimized using dict
def calculate_kmer_entropy_dict(sequence_string, k=4):
    """Optimized implementation using plain dict."""
    if len(sequence_string) < k:
        return 0.0

    n_kmers = len(sequence_string) - k + 1

    # Use plain dict with manual counting
    kmer_counts = {}
    for i in range(n_kmers):
        kmer = sequence_string[i:i+k]
        kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1

    # Calculate entropy
    total = n_kmers
    entropy = 0.0
    for count in kmer_counts.values():
        prob = count / total
        entropy -= prob * np.log2(prob)

    max_entropy = 2 * k
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return normalized_entropy


def sliding_window_complexity(repeats, window_size, kmer_func, k=4):
    """Calculate complexity using sliding window."""
    N = len(repeats)

    if window_size >= N:
        positions = [N // 2]
        window_seq = ''.join(repeats)
        comp = kmer_func(window_seq, k)
        return positions, [comp]

    positions = []
    complexity_values = []

    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        window_repeats = repeats[i:i+window_size]
        window_seq = ''.join(window_repeats)

        comp = kmer_func(window_seq, k)
        complexity_values.append(comp)

    return positions, complexity_values


def main():
    print("="*80)
    print("K-mer Entropy Performance Profiling")
    print("="*80)

    # Load data
    print("\n1. Loading subsampled repeats...")
    subsampled_repeats, subsample_every = load_subsampled_repeats()

    # Save intermediate data for reproducibility
    print("\n2. Saving intermediate data...")
    output_file = "./output/kmer_test_data.txt"
    import os
    os.makedirs('./output', exist_ok=True)

    with open(output_file, 'w') as f:
        for repeat in subsampled_repeats:
            f.write(repeat + '\n')
    print(f"   Saved {len(subsampled_repeats)} repeats to {output_file}")
    print(f"   Each repeat is 178bp")

    # Test different parameter combinations
    window_sizes = [50, 100, 200]
    k_sizes = [3, 4, 5, 6]

    print(f"\n3. Running k-mer entropy analysis with different parameters...")
    print(f"   Window sizes: {window_sizes}")
    print(f"   K-mer sizes: {k_sizes}")

    results = {}

    for window_size in window_sizes:
        for k in k_sizes:
            print(f"\n   Computing: window={window_size}, k={k}...")
            start = time.time()
            positions, values = sliding_window_complexity(
                subsampled_repeats, window_size, calculate_kmer_entropy_original, k
            )
            elapsed = time.time() - start

            print(f"      Time: {elapsed:.3f}s | Windows: {len(positions)} | "
                  f"Mean: {np.mean(values):.4f} | Std: {np.std(values):.4f}")

            results[(window_size, k)] = {
                'positions': [pos * subsample_every for pos in positions],
                'values': values,
                'time': elapsed
            }

    # Plot comparison - all on same plot
    print(f"\n4. Creating comparison plot...")

    # Normalize all results to [0, 1] for comparison
    print(f"   Normalizing all traces to [0, 1] for comparison...")

    def normalize(values):
        """Normalize array to [0, 1] range."""
        values = np.array(values)
        vmin, vmax = np.min(values), np.max(values)
        if vmax - vmin == 0:
            return np.ones_like(values) * 0.5
        return (values - vmin) / (vmax - vmin)

    # Normalize each result
    for key in results:
        results[key]['normalized'] = normalize(results[key]['values'])

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Define color scheme
    import matplotlib.cm as cm
    colors = cm.get_cmap('tab20', len(results))

    # Track which is the original (window=100, k=4)
    original_params = (100, 4)

    for idx, ((window_size, k), data) in enumerate(sorted(results.items())):
        is_original = (window_size, k) == original_params

        # Use normalized values
        y_values = data['normalized']

        # Get original value range for label
        orig_min, orig_max = np.min(data['values']), np.max(data['values'])

        if is_original:
            # Highlight the original
            ax.plot(data['positions'], y_values,
                   color='red', linewidth=3.0, alpha=0.9,
                   label=f'W={window_size}, k={k} (ORIGINAL) [{orig_min:.3f}-{orig_max:.3f}]',
                   zorder=100)
        else:
            # Plot others with different colors
            ax.plot(data['positions'], y_values,
                   color=colors(idx), linewidth=1.5, alpha=0.7,
                   label=f'W={window_size}, k={k} [{orig_min:.3f}-{orig_max:.3f}]')

    ax.set_xlabel('Sequence Position (Repeat Index)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized K-mer Entropy [0-1]', fontsize=12, fontweight='bold')
    ax.set_title('K-mer Entropy: Parameter Sensitivity Analysis (Normalized)\n'
                 f'{len(subsampled_repeats)} subsampled repeats | Original (W=100, k=4) shown in RED\n'
                 f'Legend shows original value ranges',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    output_plot = './output/kmer_entropy_parameter_sweep.png'
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"   Plot saved as {output_plot}")

    # Also create a summary statistics table
    print(f"\n5. Parameter Sensitivity Summary:")
    print(f"   {'Window':<8} {'k':<4} {'Mean':<8} {'Std':<8} {'Time(s)':<8}")
    print(f"   {'-'*45}")
    for (window_size, k), data in sorted(results.items()):
        mean_val = np.mean(data['values'])
        std_val = np.std(data['values'])
        time_val = data['time']
        print(f"   {window_size:<8} {k:<4} {mean_val:<8.4f} {std_val:<8.4f} {time_val:<8.2f}")

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)

    plt.show()


if __name__ == "__main__":
    main()
