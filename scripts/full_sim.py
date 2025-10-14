#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

# Add src to Python path to import censim modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from censim.simulation import read_sequence, read_pos_file, introduce_mutations

def main():
    parser = argparse.ArgumentParser(description="Run full centromere simulation")
    parser.add_argument("--sequence-file", "-s", required=True,
                        help="Input sequence file (.seq)")
    parser.add_argument("--pos-file", "-p", required=True,
                        help="Input position file (.pos)")
    parser.add_argument("--output-dir", "-o", default="./output",
                        help="Output directory (default: ./output)")

    args = parser.parse_args()

    # Create output directories
    output_base = Path(args.output_dir)
    os.makedirs(output_base / "fasta", exist_ok=True)
    os.makedirs(output_base / "records", exist_ok=True)
    os.makedirs(output_base / "generation_unit_pos", exist_ok=True)
    os.makedirs(output_base / "cenh3", exist_ok=True)

    # Initial parameters
    input_sequence_file = args.sequence_file
    initial_pos_file = args.pos_file

    # Read initial sequence and unit positions
    current_sequence = read_sequence(input_sequence_file)
    current_unit_data = read_pos_file(initial_pos_file)

    # Run simulation for 6 million generations in 1000-generation chunks
    for generation in range(1000, 1001, 1000):
        print(f"[{generation:>7}/6000000] Running simulation...", end=" ")

        # Run 1000 generations of mutation
        mutated_sequence, mutation_records, adjusted_pos, cenh3_occupancy = introduce_mutations(
            current_sequence, generation - 1000, 1000, current_unit_data
        )

        # Write output files
        fasta_output = output_base / "fasta" / f"{generation}generation.out.fa"
        record_output = output_base / "records" / f"{generation}generation.record.txt"
        pos_output = output_base / "generation_unit_pos" / f"{generation}generation.unit.pos"
        cenh3_output = output_base / "cenh3" / f"{generation}generation.cenh3.txt"

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

        # Write CENH3 occupancy
        with open(cenh3_output, "w") as f:
            for idx, occupied in enumerate(cenh3_occupancy):
                if occupied:
                    f.write(f"{idx}\t1\n")
                else:
                    f.write(f"{idx}\t0\n")

        print("done")

        # Update current sequence and unit data for next iteration
        current_sequence = mutated_sequence
        # Convert adjusted_pos list back to DataFrame format for next iteration
        import pandas as pd
        current_unit_data = pd.DataFrame({"start": adjusted_pos})

if __name__ == "__main__":
    main()
