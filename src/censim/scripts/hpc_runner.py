#!/usr/bin/env python3
"""
HPC-ready runner with checkpointing support for long evolutionary simulations.
Designed to run in chunks that can be resumed across job submissions.
"""

import argparse
import os
import pickle
import json
from pathlib import Path
from datetime import datetime

from censim.simulation import read_sequence, introduce_mutations


def save_checkpoint(checkpoint_file, generation, sequence, run_id):
    """Save checkpoint with current state"""
    checkpoint_data = {
        'generation': generation,
        'sequence': sequence,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat()
    }

    # Write to temporary file first, then rename (atomic operation)
    temp_file = str(checkpoint_file) + '.tmp'
    with open(temp_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    os.rename(temp_file, checkpoint_file)


def load_checkpoint(checkpoint_file):
    """Load checkpoint if it exists"""
    if not os.path.exists(checkpoint_file):
        return None

    try:
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run HPC evolutionary simulation with checkpointing"
    )
    parser.add_argument("--sequence-file", "-s", required=True,
                        help="Input sequence file (.seq)")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory for this run")
    parser.add_argument("--run-id", "-r", type=int, required=True,
                        help="Run ID (0-299 for 300 individuals)")
    parser.add_argument("--chunk-size", type=int, default=60000,
                        help="Generations per chunk (default: 60000)")
    parser.add_argument("--max-generations", type=int, default=6000000,
                        help="Maximum generations total (default: 6000000)")
    parser.add_argument("--start-generation", type=int, default=0,
                        help="Starting generation (for manual override)")
    parser.add_argument("--no-correlation-dimension", action="store_true",
                        help="Disable correlation dimension calculations")
    parser.add_argument("--d2-bias", action="store_true",
                        help="Bias insertion locations toward high D2 regions")
    parser.add_argument("--d2-bias-strength", type=float, default=1.0,
                        help="Strength of D2 bias (default: 1.0)")

    args = parser.parse_args()

    # Validation
    if args.d2_bias and args.no_correlation_dimension:
        parser.error("--d2-bias requires correlation dimension calculation")

    # Setup paths
    output_base = Path(args.output_dir)
    run_dir = output_base / f"run_{args.run_id:03d}"

    # Create directory structure
    for subdir in ["fasta", "records", "cenh3", "checkpoints"]:
        os.makedirs(run_dir / subdir, exist_ok=True)

    checkpoint_file = run_dir / "checkpoints" / "state.pkl"

    # Try to load checkpoint
    checkpoint = load_checkpoint(checkpoint_file)

    if checkpoint and checkpoint['run_id'] == args.run_id:
        print(f"[RESUME] Found checkpoint at generation {checkpoint['generation']}")
        start_gen = checkpoint['generation']
        current_sequence = checkpoint['sequence']
    else:
        print(f"[START] Starting fresh from generation 0")
        start_gen = args.start_generation
        current_sequence = read_sequence(args.sequence_file)

    # Calculate which chunk we're starting from
    chunk_start = (start_gen // args.chunk_size) * args.chunk_size
    if chunk_start == start_gen:
        target_generation = min(start_gen + args.chunk_size, args.max_generations)
    else:
        # Resume within a chunk
        target_generation = min(chunk_start + args.chunk_size, args.max_generations)

    print(f"[INFO] Run ID: {args.run_id}")
    print(f"[INFO] Starting from generation: {start_gen}")
    print(f"[INFO] Target generation for this job: {target_generation}")
    print(f"[INFO] Final target: {args.max_generations}")

    # Save metadata
    metadata_file = run_dir / "metadata.json"
    if not metadata_file.exists():
        metadata = {
            'run_id': args.run_id,
            'sequence_file': args.sequence_file,
            'max_generations': args.max_generations,
            'chunk_size': args.chunk_size,
            'd2_bias': args.d2_bias,
            'd2_bias_strength': args.d2_bias_strength,
            'correlation_dimension': not args.no_correlation_dimension,
            'start_time': datetime.now().isoformat()
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    # Run simulation in 1000-generation increments
    collapsed = False

    for generation in range(start_gen + 1000, target_generation + 1, 1000):
        if generation <= start_gen:
            continue

        print(f"[{generation:>7}/{args.max_generations}] Running simulation...", end=" ", flush=True)

        try:
            # Run 1000 generations of mutation
            mutated_sequence, mutation_records, cenh3_occupancy, collapsed, d_values_history, d_values_latest = introduce_mutations(
                current_sequence,
                generation - 1000,
                1000,
                compute_correlation_dim=not args.no_correlation_dimension,
                use_d2_bias=args.d2_bias,
                d2_bias_strength=args.d2_bias_strength
            )

            # Write output files
            fasta_output = run_dir / "fasta" / f"{generation:07d}generation.out.fa"
            record_output = run_dir / "records" / f"{generation:07d}generation.record.txt"
            cenh3_output = run_dir / "cenh3" / f"{generation:07d}generation.cenh3.txt"

            # Write FASTA file
            with open(fasta_output, "w") as f:
                f.write(f">centro_run{args.run_id:03d}_gen{generation}\n")
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

            # Save checkpoint after successful iteration
            save_checkpoint(checkpoint_file, generation, mutated_sequence, args.run_id)

            print("done")

            # Check for collapse
            if collapsed:
                print(f"\n[COLLAPSE] Array collapsed at generation {generation}")
                # Mark as complete
                complete_file = run_dir / "COLLAPSED"
                with open(complete_file, 'w') as f:
                    f.write(f"Collapsed at generation {generation}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                break

            # Update sequence
            current_sequence = mutated_sequence

        except Exception as e:
            print(f"\n[ERROR] Failed at generation {generation}: {e}")
            import traceback
            traceback.print_exc()
            # Checkpoint is still at previous successful generation
            raise

    # Check if we've completed the full run
    if generation >= args.max_generations and not collapsed:
        complete_file = run_dir / "COMPLETE"
        with open(complete_file, 'w') as f:
            f.write(f"Completed at generation {generation}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        print(f"\n[COMPLETE] Run {args.run_id} finished!")
    elif generation >= target_generation and not collapsed:
        print(f"\n[CHUNK_DONE] Completed chunk to generation {generation}")
        print(f"[NEXT] Submit next job to continue to {min(generation + args.chunk_size, args.max_generations)}")


if __name__ == "__main__":
    main()
