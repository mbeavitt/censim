#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add src to Python path to import censim modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from censim.simulation import read_sequence, introduce_mutations
import random
import numpy as np

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create output directory
    output_base = Path("./output")
    os.makedirs(output_base, exist_ok=True)

    # Input files
    input_sequence_file = "./data/15000copy_cen178.seq"

    # Output files - using profile_test naming for compatibility
    fasta_output = output_base / "profile_test.fa"
    record_output = output_base / "profile_test.record.txt"
    cenh3_output = output_base / "profile_test.cenh3.txt"

    # Read initial sequence
    sequence = read_sequence(input_sequence_file)

    # Run simulation for 10,000 generations
    generations = 10000
    print(f"Running optimized simulation for {generations} generations with seed 42...")
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
