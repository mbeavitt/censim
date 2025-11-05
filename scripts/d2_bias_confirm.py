#!/usr/bin/env python3
"""
Test script to verify D2-biased insertion locations are working.

Compares insertion distributions with and without D2 bias.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from censim.simulation import (
    read_sequence,
    introduce_mutations,
    RepeatSequence,
    apply_indel_mutations,
    compute_correlation_dimension
)


def extract_insertion_positions(mutation_records):
    """Extract insertion positions from mutation records."""
    positions = []
    for record in mutation_records:
        gen, mut_type, pos, ref, mut, copy_num = record
        if mut_type == "DUP":
            positions.append(pos)
    return positions


def test_d2_bias():
    """Test D2-biased insertions."""

    # Load a sequence from the existing simulation
    sequence_file = Path("./cdtest_output/fasta/2000000generation.out.fa")

    if not sequence_file.exists():
        print(f"Error: {sequence_file} not found")
        print("Please ensure you have simulation data in ./cdtest_output/")
        return

    # Read sequence
    print("Reading sequence...")
    with open(sequence_file, 'r') as f:
        lines = f.readlines()
    sequence = ''.join(line.strip() for line in lines[1:])

    print(f"Sequence length: {len(sequence):,} bp")

    # Compute D2 values
    print("Computing correlation dimension...")
    d_values = compute_correlation_dimension(sequence, repeat_len=178)
    print(f"D2 values: min={np.min(d_values):.3f}, max={np.max(d_values):.3f}, mean={np.mean(d_values):.3f}")

    # Test 1: No bias (uniform)
    print("\nTest 1: Running simulation WITHOUT D2 bias (10,000 generations)...")
    _, records_unbiased, _, _, _, _ = introduce_mutations(
        sequence, 0, 10000,
        compute_correlation_dim=False,
        use_d2_bias=False
    )
    positions_unbiased = extract_insertion_positions(records_unbiased)
    print(f"  Insertions: {len(positions_unbiased)}")

    # Test 2: With bias
    print("\nTest 2: Running simulation WITH D2 bias (10,000 generations, strength=2.0)...")
    _, records_biased, _, _, _, _ = introduce_mutations(
        sequence, 0, 10000,
        compute_correlation_dim=True,
        use_d2_bias=True,
        d2_bias_strength=2.0
    )
    positions_biased = extract_insertion_positions(records_biased)
    print(f"  Insertions: {len(positions_biased)}")

    # Visualize
    print("\nCreating visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: D2 values
    ax = axes[0]
    window_positions = np.arange(len(d_values))
    ax.plot(window_positions, d_values, linewidth=2, color='blue')
    ax.set_xlabel('Window Position', fontsize=12)
    ax.set_ylabel('D2 Value', fontsize=12)
    ax.set_title('Correlation Dimension (D2) Profile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Unbiased insertion histogram
    ax = axes[1]
    # Convert positions to window indices
    seq_len = len(sequence)
    bins = np.linspace(0, seq_len, 100)
    ax.hist(positions_unbiased, bins=bins, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.set_xlabel('Sequence Position (bp)', fontsize=12)
    ax.set_ylabel('Insertion Count', fontsize=12)
    ax.set_title('Insertion Distribution WITHOUT D2 Bias (should be uniform)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Biased insertion histogram
    ax = axes[2]
    ax.hist(positions_biased, bins=bins, alpha=0.7, color='darkgreen', edgecolor='black')
    ax.set_xlabel('Sequence Position (bp)', fontsize=12)
    ax.set_ylabel('Insertion Count', fontsize=12)
    ax.set_title('Insertion Distribution WITH D2 Bias (should follow D2 profile)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = "d2_bias_test.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")

    # Statistical test
    print("\nStatistical Summary:")
    print(f"Unbiased insertions: {len(positions_unbiased)} total")
    print(f"Biased insertions: {len(positions_biased)} total")

    if len(positions_unbiased) > 0:
        print(f"\nUnbiased position range: {min(positions_unbiased):,} - {max(positions_unbiased):,}")
        print(f"Unbiased position mean: {np.mean(positions_unbiased):,.0f}")
        print(f"Unbiased position std: {np.std(positions_unbiased):,.0f}")

    if len(positions_biased) > 0:
        print(f"\nBiased position range: {min(positions_biased):,} - {max(positions_biased):,}")
        print(f"Biased position mean: {np.mean(positions_biased):,.0f}")
        print(f"Biased position std: {np.std(positions_biased):,.0f}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_d2_bias()
