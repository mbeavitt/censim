#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

# Add src to Python path to import censim modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from censim.simulation import read_sequence, introduce_mutations
import random
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Run test simulation")
    parser.add_argument("generations", type=int, nargs='?', default=1000,
                        help="Number of generations to simulate (default: 1000)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible results")

    args = parser.parse_args()

    # Create output directories
    output_base = Path("./output")
    os.makedirs(output_base / "fasta", exist_ok=True)
    os.makedirs(output_base / "records", exist_ok=True)
    os.makedirs(output_base / "cenh3", exist_ok=True)

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    generations = args.generations

    # Input files
    input_sequence_file = "./data/15000copy_cen178.seq"

    # Output files
    fasta_output = output_base / "fasta" / f"{generations}generation.out.fa"
    record_output = output_base / "records" / f"{generations}generation.record.txt"
    cenh3_output = output_base / "cenh3" / f"{generations}generation.cenh3.txt"

    # Read initial sequence
    sequence = read_sequence(input_sequence_file)

    # Run simulation
    print(f"Running simulation for {generations} generations...")
    mutated_sequence, mutation_records, cenh3_occupancy, collapsed = introduce_mutations(
        sequence, 0, generations
    )

    # Write FASTA file (no header to match old CLI behavior)
    with open(fasta_output, "w") as f:
        f.write(mutated_sequence + '\n')

    # Write mutation records
    with open(record_output, "w") as f:
        for record in mutation_records:
            gen, mut_type, idx, ref, mut, copy_num = record
            f.write(f"{gen}, {mut_type}, {idx}, {ref}, {mut}, {copy_num}\n")

    # Write CENH3 occupancy
    with open(cenh3_output, "w") as f:
        for idx, occupied in enumerate(cenh3_occupancy):
            if occupied:
                f.write(f"{idx}\t1\n")
            else:
                f.write(f"{idx}\t0\n")

    print(f"Mutated sequence written to {fasta_output}")
    print(f"Mutation records written to {record_output}")
    print(f"CENH3 occupancy written to {cenh3_output}")

    if collapsed:
        print("Simulation ended early due to array collapse")
        sys.exit(1)

if __name__ == "__main__":
    main()
