#!/usr/bin/env python3
"""
Simple CLI for running centromere evolution simulations.
"""

import argparse
import os
from pathlib import Path

from censim.simulation import read_sequence, introduce_mutations


def main():
    parser = argparse.ArgumentParser(
        description="Run centromere evolution simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--sequence-file", "-s", required=True,
        help="Input sequence file (.seq)"
    )
    parser.add_argument(
        "--output-dir", "-o", default="./output",
        help="Output directory"
    )
    parser.add_argument(
        "--max-generations", "-g", type=int, default=6000000,
        help="Maximum generations to run"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=1000,
        help="Save output every N generations"
    )
    parser.add_argument(
        "--d2-bias", action="store_true",
        help="Bias insertion locations toward high D2 regions (automatically enables correlation dimension calculations)"
    )
    parser.add_argument(
        "--d2-bias-strength", type=float, default=1.0,
        help="Strength of D2 bias (0=uniform, 1=linear, >1=stronger)"
    )

    args = parser.parse_args()

    # Create output directories
    output_base = Path(args.output_dir)
    os.makedirs(output_base / "fasta", exist_ok=True)
    os.makedirs(output_base / "records", exist_ok=True)
    os.makedirs(output_base / "cenh3", exist_ok=True)

    # Read initial sequence
    print(f"Reading sequence from {args.sequence_file}")
    current_sequence = read_sequence(args.sequence_file)
    print(f"Sequence loaded: {len(current_sequence):,} bp")

    # Run simulation
    collapsed = False
    for generation in range(
        args.checkpoint_interval,
        args.max_generations + 1,
        args.checkpoint_interval
    ):
        print(f"[{generation:>7}/{args.max_generations}] Running simulation...", end=" ", flush=True)

        # Run checkpoint_interval generations of mutation
        mutated_sequence, mutation_records, cenh3_occupancy, collapsed, d_values_history, d_values_latest = introduce_mutations(
            current_sequence,
            generation - args.checkpoint_interval,
            args.checkpoint_interval,
            use_d2_bias=args.d2_bias,
            d2_bias_strength=args.d2_bias_strength
        )

        # Write output files
        fasta_output = output_base / "fasta" / f"{generation:07d}generation.out.fa"
        record_output = output_base / "records" / f"{generation:07d}generation.record.txt"
        cenh3_output = output_base / "cenh3" / f"{generation:07d}generation.cenh3.txt"

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
                f.write(f"{idx}\t{1 if occupied else 0}\n")

        print("done")

        # Check for collapse
        if collapsed:
            print(f"\nSimulation ended at generation {generation} due to array collapse")
            break

        # Update current sequence for next iteration
        current_sequence = mutated_sequence

    if not collapsed:
        print(f"\nSimulation completed: {generation:,} generations")


if __name__ == "__main__":
    main()
