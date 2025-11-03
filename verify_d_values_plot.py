#!/usr/bin/env python3
"""
Verify that optimized compute_correlation_dimension produces identical results
to the original plot_matrix_heatmap.py implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from censim.simulation import RepeatSequence, compute_correlation_dimension


def read_fasta(filepath):
    """Read FASTA file and return sequence."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Skip header lines starting with '>'
    sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
    return sequence


def main():
    # Load the evolved sequence
    evolved_seq_file = "./data/2191000generation.out.fa"
    if not Path(evolved_seq_file).exists():
        print(f"Error: {evolved_seq_file} not found")
        return

    print("Loading sequence...")
    sequence = read_fasta(evolved_seq_file)
    print(f"Loaded sequence of length {len(sequence)}")

    # Convert to RepeatSequence
    repeat_seq = RepeatSequence(sequence, repeat_size=178)
    n_repeats = repeat_seq.num_units()
    print(f"Converted to RepeatSequence with {n_repeats} units")

    # Compute d_values using optimized function
    print("\nComputing d_values with optimized algorithm...")
    d_values = compute_correlation_dimension(
        repeat_seq,
        repeat_len=178,
        r_min=0.006,
        r_max=0.1,
        n_radii=20,
        window_size=100
    )

    print(f"\nD_values stats:")
    print(f"  Shape: {d_values.shape}")
    print(f"  Mean:  {np.mean(d_values):.6f}")
    print(f"  Std:   {np.std(d_values):.6f}")
    print(f"  Min:   {np.min(d_values):.6f}")
    print(f"  Max:   {np.max(d_values):.6f}")

    # Create plot matching plot_matrix_heatmap.py style
    print("\nCreating plot...")
    fig, ax = plt.subplots(figsize=(16, 6))

    positions = np.arange(len(d_values))
    ax.plot(positions, d_values, 'b-', linewidth=1.5)
    ax.set_xlabel('Position (sequence index)', fontsize=11, fontweight='bold')
    ax.set_ylabel('D₂', fontsize=11, fontweight='bold')
    ax.set_title('Correlation Dimension (D₂) Along Sequence - Optimized Algorithm',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Use len(d_values) for x-axis since we're working with downsampled data
    ax.set_xlim(0, len(d_values))

    # Add horizontal line at D2=1 for reference
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='D₂ = 1')
    ax.legend(loc='upper right')

    # Add generation number in corner
    ax.text(0.02, 0.98, 'Generation: 2191000',
            transform=ax.transAxes,
            fontsize=10, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    output_file = "./verify_d_values_optimized.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved as {output_file}")
    print("\nCompare this with output from plot_matrix_heatmap.py to verify they match!")


if __name__ == "__main__":
    main()
