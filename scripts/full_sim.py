#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

# Add src to Python path to import censim modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from censim.simulation import read_sequence, introduce_mutations

def main():
    parser = argparse.ArgumentParser(description="Run full centromere simulation")
    parser.add_argument("--sequence-file", "-s", required=True,
                        help="Input sequence file (.seq)")
    parser.add_argument("--output-dir", "-o", default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--no-correlation-dimension", action="store_true",
                        help="Disable correlation dimension calculations (faster)")
    parser.add_argument("--d2-bias", action="store_true",
                        help="Bias insertion locations toward high D2 regions")
    parser.add_argument("--d2-bias-strength", type=float, default=1.0,
                        help="Strength of D2 bias (0=uniform, 1=linear, >1=stronger, default: 1.0)")
    parser.add_argument("--max-generations", type=int, default=6000000,
                        help="Maximum generations to run (default: 6000000)")

    args = parser.parse_args()

    # Validation: D2 bias requires CD computation
    if args.d2_bias:
        if args.no_correlation_dimension:
            parser.error("--d2-bias requires correlation dimension calculation (don't use --no-correlation-dimension)")

    # Create output directories
    output_base = Path(args.output_dir)
    os.makedirs(output_base / "fasta", exist_ok=True)
    os.makedirs(output_base / "records", exist_ok=True)
    os.makedirs(output_base / "cenh3", exist_ok=True)

    # Initial parameters
    input_sequence_file = args.sequence_file

    # Read initial sequence
    current_sequence = read_sequence(input_sequence_file)

    # Run simulation for specified generations in 1000-generation chunks
    for generation in range(1000, args.max_generations + 1, 1000):
        print(f"[{generation:>7}/{args.max_generations}] Running simulation...", end=" ")

        # Run 1000 generations of mutation
        mutated_sequence, mutation_records, cenh3_occupancy, collapsed, d_values_history, d_values_latest = introduce_mutations(
            current_sequence, generation - 1000, 1000,
            compute_correlation_dim=not args.no_correlation_dimension,
            use_d2_bias=args.d2_bias,
            d2_bias_strength=args.d2_bias_strength
        )

        # Write output files
        fasta_output = output_base / "fasta" / f"{generation}generation.out.fa"
        record_output = output_base / "records" / f"{generation}generation.record.txt"
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

        # Write CENH3 occupancy
        with open(cenh3_output, "w") as f:
            for idx, occupied in enumerate(cenh3_occupancy):
                if occupied:
                    f.write(f"{idx}\t1\n")
                else:
                    f.write(f"{idx}\t0\n")

        print("done")

        # If array collapsed, stop the simulation
        if collapsed:
            print(f"\nSimulation ended at generation {generation} due to array collapse")
            break

        # Update current sequence for next iteration
        current_sequence = mutated_sequence

if __name__ == "__main__":
    main()
