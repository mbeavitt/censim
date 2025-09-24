#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add src to Python path to import censim modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from censim.simulation import read_sequence, read_pos_file, introduce_mutations

def main():
    # Create output directories
    os.makedirs("output/fasta", exist_ok=True)
    os.makedirs("output/records", exist_ok=True)
    os.makedirs("output/generation_unit_pos", exist_ok=True)

    # Initial parameters
    input_sequence_file = "./data/15000copy_cen178.seq"
    initial_pos_file = "./data/15000copy_cen178.178bp.bed.pos"

    # Read initial sequence and unit positions
    current_sequence = read_sequence(input_sequence_file)
    current_unit_data = read_pos_file(initial_pos_file)

    # Run simulation for 6 million generations in 1000-generation chunks
    for generation in range(1000, 6000001, 1000):
        print(f"[{generation:>7}/6000000] Running simulation...", end=" ")

        # Run 1000 generations of mutation
        mutated_sequence, mutation_records, adjusted_pos = introduce_mutations(
            current_sequence, generation - 1000, 1000, current_unit_data
        )

        # Write output files
        fasta_output = f"./output/fasta/{generation}generation.out.fa"
        record_output = f"./output/records/{generation}generation.record.txt"
        pos_output = f"./output/generation_unit_pos/{generation}generation.unit.pos"

        # Write FASTA file with header
        with open(fasta_output, "w") as f:
            f.write(f">centro_{generation}gen\n")
            f.write(mutated_sequence + '\n')

        # Write mutation records
        with open(record_output, "w") as f:
            for record in mutation_records:
                gen, mut_type, idx, ref, mut, copy_num = record
                f.write(f"{gen}, {mut_type}, {idx}, {ref}, {mut}, {copy_num}\n")

        # Write adjusted positions
        with open(pos_output, "w") as f:
            for pos in sorted(adjusted_pos):
                f.write(f"centro_{generation}gen\t{pos}\n")

        print("done")

        # Update current sequence and unit data for next iteration
        current_sequence = mutated_sequence
        # Convert adjusted_pos list back to DataFrame format for next iteration
        import pandas as pd
        current_unit_data = pd.DataFrame({"start": adjusted_pos})

if __name__ == "__main__":
    main()