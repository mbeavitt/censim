#!/usr/bin/env python3
"""
Simple CLI for running centromere evolution simulations.
"""

import argparse
import os
from pathlib import Path

from censim.simulation import read_sequence, introduce_mutations
from censim.plotting import plot_similarity_and_kmer

# Default 178bp centromeric monomer sequence
DEFAULT_MONOMER = "AGTATAAGAACTTAAACCGCAACCCGATCTTAAAAGCCTAAGTAGTGTTTCCTTGTTAGAAGACACAAAGCCAAAGACTCATATGGACTTTGGCTACACCATGAAAGCTTTGAGAAGCAAGAAGAAGGTTGGTTAGTGTTTTGGAGTCGAATATGACTTGATGTCATGTGTATGATTG"


def main():
    parser = argparse.ArgumentParser(
        description="Run centromere evolution simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--resume-from",
        help="Resume from a previous checkpoint FASTA file (e.g., ./data/2191000generation.out.fa)"
    )
    parser.add_argument(
        "--monomer", "-m",
        help="Custom monomer sequence to use (overrides default 178bp sequence, ignored if --resume-from is set)"
    )
    parser.add_argument(
        "--copies", "-c", type=int, default=15000,
        help="Number of monomer copies to start with (ignored if --resume-from is set)"
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
        "--link-dups", action="store_true",
        help="Link duplications to D2 values (automatically enables correlation dimension calculations)"
    )
    parser.add_argument(
        "--link-dels", action="store_true",
        help="Link deletions to D2 values (automatically enables correlation dimension calculations)"
    )
    parser.add_argument(
        "--link-strength", type=float, default=1.0,
        help="Strength of D2 bias for linked events (0=uniform, 1=linear, >1=stronger)"
    )
    parser.add_argument(
        "--inverted-d", action="store_true",
        help="Use raw diversity values instead of inverted D2 (default: False, uses 1-diversity)"
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Generate plots at each checkpoint (default: False)"
    )

    args = parser.parse_args()

    # Create output directories
    output_base = Path(args.output_dir)
    os.makedirs(output_base / "fasta", exist_ok=True)
    os.makedirs(output_base / "records", exist_ok=True)
    os.makedirs(output_base / "cenh3", exist_ok=True)
    if args.plots:
        os.makedirs(output_base / "plots", exist_ok=True)

    # Generate or load initial sequence
    repeat_size = 178
    start_generation = 0

    if args.resume_from:
        # Load sequence from FASTA file
        import re
        print(f"Resuming from {args.resume_from}")
        with open(args.resume_from, 'r') as f:
            lines = f.readlines()
            # First line is header, extract generation if present
            header = lines[0].strip()
            gen_match = re.search(r'(\d+)gen', header)
            if gen_match:
                start_generation = int(gen_match.group(1))
            # Rest is sequence
            current_sequence = ''.join(line.strip() for line in lines[1:])

        n_copies = len(current_sequence) // repeat_size
        print(f"Loaded sequence: {len(current_sequence):,} bp ({n_copies:,} copies)")
        print(f"Starting from generation {start_generation:,}")
    else:
        # Generate sequence from monomer
        monomer = args.monomer if args.monomer else DEFAULT_MONOMER

        # Repeat or truncate monomer to fill 178bp
        if len(monomer) < repeat_size:
            # Repeat short monomer to fill 178bp
            repeats_needed = (repeat_size + len(monomer) - 1) // len(monomer)
            full_unit = (monomer * repeats_needed)[:repeat_size]
        else:
            # Truncate long monomer to 178bp
            full_unit = monomer[:repeat_size]

        current_sequence = full_unit * args.copies
        print(f"Starting with {args.copies:,} copies of 178bp monomer")
        if len(monomer) != repeat_size:
            print(f"(Base monomer: {len(monomer)}bp -> repeated/truncated to {repeat_size}bp)")
        print(f"Total sequence length: {len(current_sequence):,} bp")

    # Run simulation
    collapsed = False
    for generation in range(
        start_generation + args.checkpoint_interval,
        args.max_generations + 1,
        args.checkpoint_interval
    ):
        print(f"[{generation:>7}/{args.max_generations}] Running simulation...", end=" ", flush=True)

        # Run checkpoint_interval generations of mutation
        mutated_sequence, mutation_records, cenh3_occupancy, collapsed, d_values_latest = introduce_mutations(
            current_sequence,
            generation - args.checkpoint_interval,
            args.checkpoint_interval,
            link_dups=args.link_dups,
            link_dels=args.link_dels,
            link_strength=args.link_strength,
            invert_d2=not args.inverted_d  # Flag set = use raw diversity (invert=False)
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
                f.write(f"{idx}\t{1 if occupied else 0}\n")

        # Generate plot if enabled
        if args.plots:
            from censim.simulation import compute_correlation_dimension
            import numpy as np

            # Compute d_values if not already available (with same subsampling as will be used for matrix)
            # d_values are only computed automatically if link_dups or link_dels is enabled
            if d_values_latest is None:
                d_values_latest = compute_correlation_dimension(
                    mutated_sequence,
                    repeat_len=repeat_size,
                    scale_factor=30,  # Match the subsampling in plotting
                    invert=not args.inverted_d
                )

            # Save individual plot image
            plot_output = output_base / "plots" / f"{generation}generation.jpg"
            plot_similarity_and_kmer(
                mutated_sequence,
                repeat_size,
                d_values_latest,
                generation,
                str(plot_output)
            )

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
