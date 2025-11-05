#!/usr/bin/env python3
"""
K-mer Entropy Analysis with sliding window.

This script:
1. Loads sequence data from a FASTA file
2. Slices into repeats
3. Performs sliding window analysis using k-mer entropy
4. Plots the entropy along the sequence
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from pathlib import Path


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
        normalized_entropy: k-mer entropy normalized by maximum possible entropy
    """
    if len(sequence_string) < k:
        return 0.0

    # Extract all k-mers
    n_kmers = len(sequence_string) - k + 1

    # Use numpy for fast counting
    kmer_list = [sequence_string[i:i+k] for i in range(n_kmers)]
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


def sliding_window_kmer_entropy(repeats, window_size=100, k=4):
    """
    Calculate k-mer entropy using sliding window on repeat array.

    Args:
        repeats: list of repeat sequences
        window_size: number of repeats per window
        k: k-mer size (default: 4)

    Returns:
        positions: center positions of windows (in repeat indices)
        entropy_values: k-mer entropy at each window position
    """
    N = len(repeats)
    positions = []
    entropy_values = []

    # Slide window across repeats
    for i in range(0, N - window_size + 1, 1):
        center = i + window_size // 2
        positions.append(center)

        # Extract window of repeats and concatenate
        window_repeats = repeats[i:i+window_size]
        window_seq = ''.join(window_repeats)

        # Calculate k-mer entropy for this window
        entropy = calculate_kmer_entropy(window_seq, k=k)
        entropy_values.append(entropy)

    return positions, entropy_values


def main():
    """Main function for k-mer entropy analysis."""
    parser = argparse.ArgumentParser(
        description='K-mer Entropy Analysis using sliding window'
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        default="./data/2191000generation.out.fa",
        help='Input FASTA file (default: ./data/2191000generation.out.fa)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./output/kmer_entropy.png',
        help='Output plot filename (default: ./output/kmer_entropy.png)'
    )

    parser.add_argument(
        '-w', '--window-size',
        type=int,
        default=100,
        help='Window size in repeats (default: 100)'
    )

    parser.add_argument(
        '-k', '--kmer-size',
        type=int,
        default=4,
        help='K-mer size (default: 4)'
    )

    args = parser.parse_args()

    # Configuration
    input_file = args.input
    output_file = args.output
    repeat_len = 178
    window_size = args.window_size
    k = args.kmer_size

    print("=" * 80)
    print("K-mer Entropy Analysis")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Window size: {window_size} repeats")
    print(f"K-mer size: {k}")

    # Load data
    print(f"\n1. Loading sequence data from {input_file}...")
    start_total = time.time()
    with open(input_file, "r") as file:
        contents = file.read().strip()

    n_repeats = int(len(contents) / repeat_len)
    repeats = [contents[i*repeat_len:(i+1)*repeat_len] for i in range(n_repeats)]

    print(f"   Loaded {n_repeats} repeats of length {repeat_len}bp")
    print(f"   Window size: {window_size} repeats")

    # Apply subsampling if needed
    print("\n2. Subsampling repeats...")
    import math
    if n_repeats <= 1000:
        subsample_every = 1
        subsampled_repeats = repeats
        print(f"   Using all repeats (no subsampling, n={n_repeats})")
    else:
        subsample_every = max(1, int(math.sqrt(n_repeats / 30)))
        subset_indices = np.arange(0, n_repeats, subsample_every)
        subsampled_repeats = [repeats[i] for i in subset_indices]
        print(f"   Subsampling every {subsample_every}th repeat: {len(subsampled_repeats)}/{n_repeats} total")
        print(f"   Subsampling interval: {subsample_every} (scale factor for positions)")

    # Calculate k-mer entropy with sliding window
    print(f"\n3. Computing k-mer entropy (k={k})...")
    start_metric = time.time()
    positions, entropy_values = sliding_window_kmer_entropy(
        subsampled_repeats, window_size=window_size, k=k
    )
    elapsed_metric = time.time() - start_metric

    # Scale positions back to original repeat indices
    positions_scaled = [int(p * subsample_every) for p in positions]

    print(f"   Computed {len(positions)} windows")
    print(f"   Time: {elapsed_metric:.2f}s")
    print(f"   k-mer entropy range: [{np.min(entropy_values):.4f}, {np.max(entropy_values):.4f}]")
    print(f"   k-mer entropy mean: {np.mean(entropy_values):.4f} ± {np.std(entropy_values):.4f}")

    # Create plot
    print("\n4. Creating plot...")
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(positions_scaled, entropy_values, 'g-', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Sequence Position (Repeat Index)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{k}-mer Entropy (Normalized)', fontsize=12, fontweight='bold')
    ax.set_title(f'K-mer Entropy Along Sequence\n'
                 f'Window size: {window_size} repeats | {n_repeats} total repeats | k={k}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_repeats)
    ax.set_ylim(-0.05, 1.05)

    # Add summary statistics text box
    stats_text = (
        f'k-mer: {np.mean(entropy_values):.3f} ± {np.std(entropy_values):.3f}'
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    os.makedirs('./output', exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Plot saved as {output_file}")

    # Save data to file
    data_file = './output/kmer_entropy_data.npz'
    np.savez(data_file,
             kmer_positions=np.array(positions_scaled),
             kmer_values=np.array(entropy_values),
             window_size=window_size,
             n_repeats=n_repeats,
             repeat_len=repeat_len,
             k=k,
             subsample_every=subsample_every)
    print(f"   Data saved as {data_file}")

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  K-mer Entropy ({k}-mer): {np.mean(entropy_values):.4f} ± {np.std(entropy_values):.4f}")
    print(f"  Processing time: {elapsed_metric:.2f}s")

    plt.show()


if __name__ == "__main__":
    main()
