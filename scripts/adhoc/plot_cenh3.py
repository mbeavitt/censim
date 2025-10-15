#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np

def read_cenh3_file(file_path):
    """Read CENH3 occupancy file and return unit indices and occupancy values."""
    units = []
    occupancy = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                units.append(int(parts[0]))
                occupancy.append(int(parts[1]))

    return np.array(units), np.array(occupancy)

def plot_cenh3_histogram(file_path, output_path=None, bins=50):
    """Plot histogram of CENH3 occupancy across units."""
    units, occupancy = read_cenh3_file(file_path)

    # Get only the units with CENH3 (occupancy = 1)
    occupied_units = units[occupancy == 1]

    # Calculate statistics
    total_units = len(units)
    num_occupied = len(occupied_units)
    occupancy_rate = num_occupied / total_units if total_units > 0 else 0

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top plot: Histogram of CENH3-occupied units
    ax1.hist(occupied_units, bins=bins, color='darkblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Unit Index', fontsize=12)
    ax1.set_ylabel('Number of CENH3 Nucleosomes', fontsize=12)
    ax1.set_title(f'CENH3 Distribution Across Units\n({num_occupied}/{total_units} units occupied, {occupancy_rate:.1%} occupancy)',
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add vertical line at center
    center = total_units // 2
    ax1.axvline(center, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Center (unit {center})')
    ax1.legend()

    # Bottom plot: Full occupancy map as binary track
    ax2.scatter(units[occupancy == 1], np.ones(num_occupied), color='darkblue',
                marker='|', s=100, alpha=0.6, label='CENH3 present')
    ax2.scatter(units[occupancy == 0], np.zeros(np.sum(occupancy == 0)), color='lightgray',
                marker='|', s=50, alpha=0.3, label='No CENH3')
    ax2.set_xlabel('Unit Index', fontsize=12)
    ax2.set_ylabel('Occupancy', fontsize=12)
    ax2.set_title('CENH3 Occupancy Map', fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Absent', 'Present'])
    ax2.axvline(center, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.legend(loc='upper right')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    # Print statistics
    print(f"\nCENH3 Statistics:")
    print(f"  Total units: {total_units}")
    print(f"  Occupied units: {num_occupied}")
    print(f"  Occupancy rate: {occupancy_rate:.2%}")
    if num_occupied > 0:
        print(f"  Mean position: {np.mean(occupied_units):.1f}")
        print(f"  Std deviation: {np.std(occupied_units):.1f}")
        print(f"  Center position: {center}")

def main():
    parser = argparse.ArgumentParser(description='Plot CENH3 occupancy histogram')
    parser.add_argument('cenh3_file', help='Path to CENH3 occupancy file')
    parser.add_argument('-o', '--output', help='Output file path (e.g., cenh3_plot.png)')
    parser.add_argument('-b', '--bins', type=int, default=50,
                        help='Number of bins for histogram (default: 50)')

    args = parser.parse_args()

    plot_cenh3_histogram(args.cenh3_file, args.output, args.bins)

if __name__ == "__main__":
    main()
