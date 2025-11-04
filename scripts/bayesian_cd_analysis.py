#!/usr/bin/env python3
"""
Analyze existing simulation results with EMA smoothing of correlation dimension.

Reads FASTA files from 2M to 3M generations, computes correlation dimension
at each step, applies EMA smoothing, and visualizes the results.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from censim.simulation import compute_correlation_dimension, smooth_d_values_ema


def read_fasta_sequence(fasta_file):
    """Read sequence from FASTA file."""
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
    # Skip header line, concatenate sequence lines
    sequence = ''.join(line.strip() for line in lines[1:])
    return sequence


def main():
    # Configuration
    data_dir = Path("./cdtest_output/fasta")
    start_gen = 2_000_000
    end_gen = 3_000_000
    step = 1000  # Files are available every 1000 generations

    # EMA smoothing parameter (higher = less smoothing, lower = more smoothing)
    alpha = 0.3

    print(f"EMA CD Analysis: {start_gen:,} to {end_gen:,} generations")
    print(f"EMA alpha: {alpha} (higher = less smoothing)")
    print(f"Reading from: {data_dir}")
    print("=" * 70)

    # Storage for results
    generations = []
    d_values_raw = []
    d_values_smoothed = []

    # Initialize smoothed as None (will be set on first iteration)
    smoothed = None

    # Process each generation
    for gen in range(start_gen, end_gen + step, step):
        fasta_file = data_dir / f"{gen}generation.out.fa"

        if not fasta_file.exists():
            print(f"Warning: {fasta_file} not found, skipping...")
            continue

        print(f"[{gen:>9,}] Loading and computing CD...", end=" ", flush=True)

        # Read sequence
        sequence = read_fasta_sequence(fasta_file)

        # Compute correlation dimension
        d_values = compute_correlation_dimension(sequence, repeat_len=178)

        # Store raw values
        generations.append(gen)
        d_values_raw.append(d_values)

        # Apply EMA smoothing
        smoothed = smooth_d_values_ema(smoothed, d_values, alpha=alpha)
        d_values_smoothed.append(smoothed)

        print(f"✓ (mean raw D2={np.mean(d_values):.3f}, smoothed={np.mean(smoothed):.3f})")

    print("=" * 70)
    print(f"Processed {len(generations)} generations")

    # Convert to arrays for plotting
    generations = np.array(generations)

    # Create visualizations
    print("\nGenerating plots...")

    # Figure 1: Mean D2 over time (raw vs smoothed)
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    raw_means = [np.mean(d) for d in d_values_raw]
    smoothed_means = [np.mean(d) for d in d_values_smoothed]
    raw_stds = [np.std(d) for d in d_values_raw]
    smoothed_stds = [np.std(d) for d in d_values_smoothed]

    ax1.plot(generations, raw_means, 'o-', alpha=0.5, label='Raw D2 (noisy)', color='lightblue')
    ax1.fill_between(generations,
                      np.array(raw_means) - np.array(raw_stds),
                      np.array(raw_means) + np.array(raw_stds),
                      alpha=0.2, color='lightblue')

    ax1.plot(generations, smoothed_means, 'o-', label=f'EMA Smoothed D2 (α={alpha})', color='darkblue', linewidth=2)
    ax1.fill_between(generations,
                      np.array(smoothed_means) - np.array(smoothed_stds),
                      np.array(smoothed_means) + np.array(smoothed_stds),
                      alpha=0.3, color='darkblue')

    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Mean Correlation Dimension (D2)', fontsize=12)
    ax1.set_title('EMA Smoothing of Correlation Dimension Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file1 = "ema_cd_timeseries.png"
    plt.savefig(output_file1, dpi=150)
    print(f"Saved: {output_file1}")

    # Figure 2: Distribution at different time points
    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Select 6 evenly spaced generations to show distribution evolution
    indices = np.linspace(0, len(generations) - 1, 6, dtype=int)

    for i, idx in enumerate(indices):
        ax = axes[i]
        gen = generations[idx]

        # Plot raw and smoothed distributions
        ax.hist(d_values_raw[idx], bins=50, alpha=0.5, label='Raw', color='lightcoral', density=True)
        ax.hist(d_values_smoothed[idx], bins=50, alpha=0.7, label='EMA Smoothed',
                color='darkgreen', density=True)

        ax.set_xlabel('D2 Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Generation {gen:,}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig2.suptitle(f'Evolution of D2 Distribution with EMA Smoothing (α={alpha})',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file2 = "ema_cd_distribution_evolution.png"
    plt.savefig(output_file2, dpi=150)
    print(f"Saved: {output_file2}")

    # Figure 3: Spatial distribution at final generation
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(14, 10))

    final_idx = -1
    positions = np.arange(len(d_values_raw[final_idx]))

    # Raw values
    ax3a.plot(positions, d_values_raw[final_idx], alpha=0.5, color='lightblue', label='Raw D2')
    ax3a.set_xlabel('Window Position', fontsize=11)
    ax3a.set_ylabel('D2 Value', fontsize=11)
    ax3a.set_title(f'Raw Correlation Dimension at Generation {generations[final_idx]:,}',
                   fontsize=12, fontweight='bold')
    ax3a.grid(True, alpha=0.3)
    ax3a.legend()

    # Smoothed values
    positions_smooth = np.arange(len(d_values_smoothed[final_idx]))
    ax3b.plot(positions_smooth, d_values_smoothed[final_idx], color='darkblue',
              linewidth=2, label=f'EMA Smoothed D2 (α={alpha})')
    ax3b.set_xlabel('Window Position', fontsize=11)
    ax3b.set_ylabel('D2 Value', fontsize=11)
    ax3b.set_title(f'EMA Smoothed Correlation Dimension at Generation {generations[final_idx]:,}',
                   fontsize=12, fontweight='bold')
    ax3b.grid(True, alpha=0.3)
    ax3b.legend()

    plt.tight_layout()
    output_file3 = "ema_cd_spatial_final.png"
    plt.savefig(output_file3, dpi=150)
    print(f"Saved: {output_file3}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"\nFinal statistics (generation {generations[-1]:,}):")
    print(f"  Raw D2:      mean={np.mean(d_values_raw[-1]):.4f}, std={np.std(d_values_raw[-1]):.4f}")
    print(f"  Smoothed D2: mean={np.mean(d_values_smoothed[-1]):.4f}, std={np.std(d_values_smoothed[-1]):.4f}")
    print(f"  Noise reduction: {(1 - np.std(d_values_smoothed[-1])/np.std(d_values_raw[-1]))*100:.1f}%")


if __name__ == "__main__":
    main()
