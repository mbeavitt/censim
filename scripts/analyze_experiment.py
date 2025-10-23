#!/usr/bin/env python3
"""
Analyze experiment results and generate edit distance plots for each generation level.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import Levenshtein
from collections import defaultdict


def load_consensus_repeat(seq_file):
    """Extract the first 178bp repeat as consensus from the .seq file."""
    with open(seq_file) as f:
        seq = f.read().strip()
    consensus = seq[:178]
    return consensus


def load_evolved_repeats(fa_file, repeat_length=178):
    """Load all repeats from the evolved fa file."""
    with open(fa_file) as f:
        sequence = f.read().strip()

    repeats = []
    for i in range(0, len(sequence), repeat_length):
        if i + repeat_length <= len(sequence):
            repeats.append(sequence[i:i+repeat_length])

    return repeats


def calculate_edit_distances(consensus, repeats):
    """Calculate edit distance for each repeat against consensus."""
    distances = []
    for repeat in repeats:
        dist = Levenshtein.distance(consensus, repeat)
        distances.append(dist)
    return np.array(distances)


def calculate_windowed_average(distances, window_size=100):
    """Calculate windowed average of edit distances."""
    n_repeats = len(distances)
    windowed_distances = []
    windowed_positions = []

    for i in range(0, n_repeats, window_size):
        window_end = min(i + window_size, n_repeats)
        window_avg = np.mean(distances[i:window_end])
        window_center = (i + window_end) / 2
        windowed_distances.append(window_avg)
        windowed_positions.append(window_center / n_repeats)

    return np.array(windowed_positions), np.array(windowed_distances)


def identify_successful_runs(output_dir):
    """Identify runs that completed all 6M generations without collapsing."""
    output_path = Path(output_dir)

    # Find all runs with 6M generation file
    completed_files = list(output_path.glob("run*_6000000gen.fa"))

    successful_runs = []
    for f in completed_files:
        run_id = int(f.name.split('_')[0].replace('run', ''))
        # Check if there's a collapse marker
        collapse_marker = output_path / f"run{run_id:04d}_COLLAPSED.txt"
        if not collapse_marker.exists():
            successful_runs.append(run_id)

    return sorted(successful_runs)


def plot_generation_level(generation, all_positions, all_distances, output_file):
    """Plot edit distance for a specific generation level across all runs."""

    # Calculate mean and std across all runs
    # First, ensure all runs have the same number of windows
    min_length = min(len(pos) for pos in all_positions)

    # Truncate all to the same length
    positions_aligned = [pos[:min_length] for pos in all_positions]
    distances_aligned = [dist[:min_length] for dist in all_distances]

    # Stack into arrays
    positions = np.array(positions_aligned[0])  # All should be the same
    distances_matrix = np.array(distances_aligned)  # Shape: (n_runs, n_windows)

    # Calculate statistics
    mean_distances = np.mean(distances_matrix, axis=0)
    std_distances = np.std(distances_matrix, axis=0)

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot mean line
    plt.plot(positions, mean_distances, linewidth=2, color='#2E86AB', label='Mean')

    # Plot confidence band (mean ± std)
    plt.fill_between(positions,
                     mean_distances - std_distances,
                     mean_distances + std_distances,
                     alpha=0.3, color='#2E86AB', label='±1 SD')

    plt.xlabel('Scaled Array Position', fontsize=14)
    plt.ylabel('Average Edit Distance', fontsize=14)
    plt.title(f'Edit Distance at Generation {generation:,} (n={len(all_distances)} runs)',
              fontsize=16, pad=20)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.xlim(0, 1)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"  Saved plot: {output_file}")


def main():
    # Configuration
    seq_file = "data/15000copy_cen178.seq"
    output_dir = "experiment_output"
    plot_dir = "experiment_plots"
    generations = [1000000, 2000000, 3000000, 4000000, 5000000, 6000000]
    window_size = 100

    # Create plot directory
    Path(plot_dir).mkdir(exist_ok=True)

    print("Loading consensus repeat...")
    consensus = load_consensus_repeat(seq_file)
    print(f"Consensus length: {len(consensus)} bp")

    print("\nIdentifying successful runs...")
    successful_runs = identify_successful_runs(output_dir)
    print(f"Found {len(successful_runs)} successful runs")

    if len(successful_runs) == 0:
        print("No successful runs found. Exiting.")
        return

    # Process each generation level
    for generation in generations:
        print(f"\nProcessing generation {generation:,}...")

        all_positions = []
        all_distances = []

        # Load data from all successful runs
        for run_id in successful_runs:
            fa_file = Path(output_dir) / f"run{run_id:04d}_{generation}gen.fa"

            if not fa_file.exists():
                continue

            # Load repeats and calculate edit distances
            repeats = load_evolved_repeats(fa_file)
            distances = calculate_edit_distances(consensus, repeats)

            # Calculate windowed average
            positions, windowed_dist = calculate_windowed_average(distances, window_size)

            all_positions.append(positions)
            all_distances.append(windowed_dist)

        print(f"  Loaded {len(all_distances)} runs for generation {generation:,}")

        # Generate plot
        output_file = Path(plot_dir) / f"edit_distance_{generation}gen.png"
        plot_generation_level(generation, all_positions, all_distances, output_file)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Plots saved to: {plot_dir}/")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
