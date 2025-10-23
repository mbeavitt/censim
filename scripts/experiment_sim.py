#!/usr/bin/env python3
"""
Modified simulation script that only outputs FASTA files at specific generations.
Used for the edit distance experiment.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to Python path to import censim modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from censim.simulation import read_sequence, introduce_mutations

def main():
    parser = argparse.ArgumentParser(description="Run centromere simulation with selective output")
    parser.add_argument("--sequence-file", "-s", required=True,
                        help="Input sequence file (.seq)")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory")
    parser.add_argument("--run-id", "-r", type=int, required=True,
                        help="Run ID number")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Generations to output FASTA files
    output_generations = [1000000, 2000000, 3000000, 4000000, 5000000, 6000000]

    # Read initial sequence
    current_sequence = read_sequence(args.sequence_file)

    collapsed = False

    # Run simulation for 6 million generations in 1000-generation chunks
    for generation in range(1000, 6000001, 1000):
        if generation % 100000 == 0:
            print(f"Run {args.run_id}: [{generation:>7}/6000000]", flush=True)

        # Run 1000 generations of mutation
        mutated_sequence, mutation_records, cenh3_occupancy, collapsed = introduce_mutations(
            current_sequence, generation - 1000, 1000
        )

        # Only write FASTA files at specific generations
        if generation in output_generations:
            fasta_output = output_dir / f"run{args.run_id:04d}_{generation}gen.fa"
            with open(fasta_output, "w") as f:
                f.write(mutated_sequence + '\n')
            print(f"Run {args.run_id}: Saved generation {generation}", flush=True)

        # If array collapsed, stop the simulation
        if collapsed:
            print(f"Run {args.run_id}: Collapsed at generation {generation}", flush=True)
            # Create a marker file to indicate collapse
            collapse_marker = output_dir / f"run{args.run_id:04d}_COLLAPSED.txt"
            with open(collapse_marker, "w") as f:
                f.write(f"{generation}\n")
            break

        # Update current sequence for next iteration
        current_sequence = mutated_sequence

    # If we completed all 6 million generations successfully
    if not collapsed:
        print(f"Run {args.run_id}: Completed all 6 million generations", flush=True)

if __name__ == "__main__":
    main()
