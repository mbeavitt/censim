#!/usr/bin/env python3
"""HPC worker script for checkpoint-based simulation.

This script runs exactly 1000 generations, saves a checkpoint, and exits.
Another worker will pick up from the checkpoint to continue the simulation.

Usage:
    python hpc_worker.py --run-id <id> --sequence-file <file> [options]
    python hpc_worker.py --run-id <id> --resume [options]
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to Python path to import censim modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from censim.simulation import introduce_mutations
from censim.checkpoint import SimulationCheckpoint


def main():
    parser = argparse.ArgumentParser(
        description="Run 1000 generations of centromere simulation with checkpointing"
    )
    parser.add_argument(
        "--run-id", "-r", required=True,
        help="Unique run identifier (e.g., 1-400 for job array)"
    )
    parser.add_argument(
        "--sequence-file", "-s",
        help="Initial sequence file (.seq) - required if not resuming"
    )
    parser.add_argument(
        "--checkpoint-dir", "-c", default="./checkpoints",
        help="Checkpoint directory (default: ./checkpoints)"
    )
    parser.add_argument(
        "--output-dir", "-o", default="./output",
        help="Output directory for FASTA and records (default: ./output)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint (ignore --sequence-file)"
    )
    parser.add_argument(
        "--no-correlation-dimension", action="store_true",
        help="Disable correlation dimension calculations (faster)"
    )
    parser.add_argument(
        "--d2-bias", action="store_true",
        help="Bias insertion locations toward high D2 regions"
    )
    parser.add_argument(
        "--d2-bias-strength", type=float, default=1.0,
        help="Strength of D2 bias (0=uniform, 1=linear, >1=stronger, default: 1.0)"
    )
    parser.add_argument(
        "--max-generations", type=int, default=6000000,
        help="Maximum generation to simulate to (default: 6000000)"
    )

    args = parser.parse_args()

    # Validation
    if args.d2_bias and args.no_correlation_dimension:
        parser.error("--d2-bias requires correlation dimension calculation")

    if not args.resume and not args.sequence_file:
        parser.error("--sequence-file is required when not resuming")

    # Initialize checkpoint manager
    checkpoint_mgr = SimulationCheckpoint(args.run_id, args.checkpoint_dir)

    # Create output directories
    output_base = Path(args.output_dir)
    os.makedirs(output_base / "fasta", exist_ok=True)
    os.makedirs(output_base / "records", exist_ok=True)

    # Determine starting state
    if args.resume:
        print(f"[Run {args.run_id}] Resuming from checkpoint...")
        try:
            ckpt = checkpoint_mgr.load_checkpoint()
            current_sequence = ckpt["sequence"]
            start_generation = ckpt["generation"]
            all_mutation_records = ckpt["mutation_records"]
            print(f"[Run {args.run_id}] Loaded checkpoint at generation {start_generation}")
        except FileNotFoundError:
            print(f"[Run {args.run_id}] ERROR: No checkpoint found for run_id={args.run_id}")
            sys.exit(1)
    else:
        print(f"[Run {args.run_id}] Starting new simulation...")
        with open(args.sequence_file, "r") as f:
            current_sequence = f.read().strip()
        start_generation = 0
        all_mutation_records = []

    # Check if we've already reached the target
    if start_generation >= args.max_generations:
        print(f"[Run {args.run_id}] Simulation already complete at generation {start_generation}")
        sys.exit(0)

    # Calculate end generation for this chunk (1000 generations or until max)
    end_generation = min(start_generation + 1000, args.max_generations)
    num_generations = end_generation - start_generation

    print(f"[Run {args.run_id}] Simulating generations {start_generation} -> {end_generation}")

    # Run simulation for this chunk
    mutated_sequence, mutation_records, cenh3_occupancy, collapsed, d_values_history, d_values_latest = (
        introduce_mutations(
            current_sequence,
            start_generation,
            num_generations,
            compute_correlation_dim=not args.no_correlation_dimension,
            use_d2_bias=args.d2_bias,
            d2_bias_strength=args.d2_bias_strength
        )
    )

    # Append new mutation records to the complete history
    all_mutation_records.extend(mutation_records)

    # Write output files for this generation
    fasta_output = output_base / "fasta" / f"run_{args.run_id}_gen_{end_generation}.fa"
    record_output = output_base / "records" / f"run_{args.run_id}_gen_{end_generation}.txt"

    # Write FASTA file
    with open(fasta_output, "w") as f:
        f.write(f">run_{args.run_id}_gen_{end_generation}\n")
        f.write(mutated_sequence + '\n')

    # Write mutation records (only the new ones from this chunk)
    with open(record_output, "w") as f:
        for record in mutation_records:
            gen, mut_type, idx, ref, mut, copy_num = record
            f.write(f"{gen}, {mut_type}, {idx}, {ref}, {mut}, {copy_num}\n")

    # Save checkpoint for next worker
    metadata = {
        "collapsed": collapsed,
        "sequence_length": len(mutated_sequence),
        "total_mutations": len(all_mutation_records),
        "d2_bias": args.d2_bias,
        "d2_bias_strength": args.d2_bias_strength
    }

    checkpoint_path = checkpoint_mgr.save_checkpoint(
        end_generation,
        mutated_sequence,
        all_mutation_records,
        metadata
    )

    print(f"[Run {args.run_id}] Checkpoint saved: {checkpoint_path}")

    # Check for completion or collapse
    if collapsed:
        print(f"[Run {args.run_id}] Simulation ended due to array collapse")
        sys.exit(0)

    if end_generation >= args.max_generations:
        print(f"[Run {args.run_id}] Simulation complete at generation {end_generation}")
        sys.exit(0)

    print(f"[Run {args.run_id}] Worker complete. Next worker will resume from generation {end_generation}")


if __name__ == "__main__":
    main()
