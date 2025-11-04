#!/usr/bin/env python3
"""
Create an animated movie of correlation dimension evolution over time.

Shows raw vs EMA-smoothed CD values evolving from 2M to 3M generations.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

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


def compute_all_data(data_dir, start_gen, end_gen, step, alpha=0.3):
    """Compute raw and smoothed CD for all generations."""
    print("Computing correlation dimensions for all generations...")

    generations = []
    d_values_raw = []
    d_values_smoothed = []

    smoothed = None

    for gen in tqdm(range(start_gen, end_gen + step, step), desc="Processing"):
        fasta_file = data_dir / f"{gen}generation.out.fa"

        if not fasta_file.exists():
            continue

        # Read sequence and compute CD
        sequence = read_fasta_sequence(fasta_file)
        d_values = compute_correlation_dimension(sequence, repeat_len=178)

        # Apply EMA smoothing
        smoothed = smooth_d_values_ema(smoothed, d_values, alpha=alpha)

        generations.append(gen)
        d_values_raw.append(d_values)
        d_values_smoothed.append(smoothed)

    return generations, d_values_raw, d_values_smoothed


def create_animation(generations, d_values_raw, d_values_smoothed, alpha, output_file, fps=10, skip_frames=5):
    """Create animation of CD evolution over time."""

    # Subsample frames for faster rendering
    indices = list(range(0, len(generations), skip_frames))

    print(f"Creating animation with {len(indices)} frames...")

    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Determine global y-axis limits for consistency
    all_raw = np.concatenate([d_values_raw[i] for i in indices])
    all_smoothed = np.concatenate([d_values_smoothed[i] for i in indices])
    y_min = min(all_raw.min(), all_smoothed.min()) - 0.1
    y_max = max(all_raw.max(), all_smoothed.max()) + 0.1

    def init():
        """Initialize animation."""
        ax1.clear()
        ax2.clear()
        return []

    def update(frame_idx):
        """Update function for animation."""
        idx = indices[frame_idx]
        gen = generations[idx]

        ax1.clear()
        ax2.clear()

        # Plot raw CD
        positions_raw = np.arange(len(d_values_raw[idx]))
        ax1.plot(positions_raw, d_values_raw[idx], color='lightcoral', linewidth=1, alpha=0.7)
        ax1.set_ylabel('D2 Value', fontsize=12, fontweight='bold')
        ax1.set_title(f'Raw Correlation Dimension - Generation {gen:,}',
                     fontsize=13, fontweight='bold')
        ax1.set_ylim(y_min, y_max)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, len(d_values_raw[idx]))

        # Plot smoothed CD
        positions_smooth = np.arange(len(d_values_smoothed[idx]))
        ax2.plot(positions_smooth, d_values_smoothed[idx], color='darkblue', linewidth=2)
        ax2.set_xlabel('Window Position', fontsize=12, fontweight='bold')
        ax2.set_ylabel('D2 Value', fontsize=12, fontweight='bold')
        ax2.set_title(f'EMA Smoothed Correlation Dimension (α={alpha}) - Generation {gen:,}',
                     fontsize=13, fontweight='bold')
        ax2.set_ylim(y_min, y_max)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, len(d_values_smoothed[idx]))

        # Add progress indicator
        progress = (idx / len(generations)) * 100
        fig.text(0.5, 0.02, f'Progress: {progress:.1f}% ({gen:,} / {generations[-1]:,} generations)',
                ha='center', fontsize=10, style='italic')

        plt.tight_layout(rect=[0, 0.03, 1, 1])

        return []

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(indices), interval=1000/fps, blit=True
    )

    # Save animation
    print(f"Saving animation to {output_file}...")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='censim'), bitrate=1800)
    anim.save(output_file, writer=writer)
    plt.close()

    print(f"Animation saved successfully!")


def main():
    # Configuration
    data_dir = Path("./cdtest_output/fasta")
    start_gen = 2_000_000
    end_gen = 3_000_000
    step = 1000
    alpha = 0.3
    fps = 10  # Frames per second
    skip_frames = 5  # Only render every Nth frame for speed

    output_file = "cd_evolution.mp4"

    print("=" * 70)
    print("Correlation Dimension Evolution Animation")
    print("=" * 70)
    print(f"Generations: {start_gen:,} to {end_gen:,}")
    print(f"EMA alpha: {alpha}")
    print(f"Output: {output_file}")
    print(f"FPS: {fps}, Skip: {skip_frames} (every {skip_frames}th frame)")
    print("=" * 70)

    # Compute all data
    generations, d_values_raw, d_values_smoothed = compute_all_data(
        data_dir, start_gen, end_gen, step, alpha
    )

    # Create animation
    create_animation(
        generations, d_values_raw, d_values_smoothed,
        alpha, output_file, fps, skip_frames
    )

    print("\n" + "=" * 70)
    print("Done!")
    print(f"View with: ffplay {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
