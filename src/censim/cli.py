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
        "--monomer", "-m",
        help="Custom monomer sequence to use (overrides default 178bp sequence)"
    )
    parser.add_argument(
        "--copies", "-c", type=int, default=15000,
        help="Number of monomer copies to start with"
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
    parser.add_argument(
        "--inverted-d", action="store_true",
        help="Use raw diversity values instead of inverted D2 (default: False, uses 1-diversity)"
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Generate plots at each checkpoint (default: False)"
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Create video from plots (saves space vs individual images)"
    )
    parser.add_argument(
        "--video-fps", type=int, default=10,
        help="Frames per second for video output (default: 10)"
    )

    args = parser.parse_args()

    # Create output directories
    output_base = Path(args.output_dir)
    os.makedirs(output_base / "fasta", exist_ok=True)
    os.makedirs(output_base / "records", exist_ok=True)
    os.makedirs(output_base / "cenh3", exist_ok=True)
    if args.plots and not args.video:
        os.makedirs(output_base / "plots", exist_ok=True)

    # Initialize video writer if requested
    video_writer = None
    if args.video:
        import imageio
        video_path = output_base / "simulation.mp4"
        video_writer = imageio.get_writer(
            str(video_path),
            fps=args.video_fps,
            codec='libx264',
            quality=8,  # 1-10, higher is better
            pixelformat='yuv420p'
        )
        print(f"Video output: {video_path} ({args.video_fps} fps)")

    # Generate initial sequence
    monomer = args.monomer if args.monomer else DEFAULT_MONOMER
    repeat_size = 178

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
            d2_bias_strength=args.d2_bias_strength,
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
        if args.plots or args.video:
            from censim.simulation import compute_correlation_dimension
            import numpy as np

            # Compute d_values if not already available (with same subsampling as will be used for matrix)
            if d_values_latest is None:
                d_values_latest = compute_correlation_dimension(
                    mutated_sequence,
                    repeat_len=repeat_size,
                    scale_factor=30,  # Match the subsampling in plotting
                    invert=not args.inverted_d
                )

            if args.video:
                # Generate frame and add to video
                frame = plot_similarity_and_kmer(
                    mutated_sequence,
                    repeat_size,
                    d_values_latest,
                    generation,
                    return_frame=True
                )
                video_writer.append_data(frame)
            else:
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

    # Close video writer if it was used
    if video_writer is not None:
        video_writer.close()
        print(f"\nVideo saved: {video_path}")


if __name__ == "__main__":
    main()
