#!/usr/bin/env python3
"""
Helper script to manage HPC job submissions and check run status
"""

import argparse
import subprocess
import json
from pathlib import Path
import sys


def get_run_status(output_dir, run_id):
    """Check the status of a specific run"""
    run_dir = Path(output_dir) / f"run_{run_id:03d}"

    if not run_dir.exists():
        return "NOT_STARTED"

    if (run_dir / "COMPLETE").exists():
        return "COMPLETE"

    if (run_dir / "COLLAPSED").exists():
        return "COLLAPSED"

    checkpoint = run_dir / "checkpoints" / "state.pkl"
    if checkpoint.exists():
        import pickle
        with open(checkpoint, 'rb') as f:
            data = pickle.load(f)
            return f"IN_PROGRESS ({data['generation']:,} gen)"

    return "UNKNOWN"


def status_command(args):
    """Show status of all runs"""
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        return

    print(f"Status of runs in {output_dir}:\n")

    status_counts = {
        "NOT_STARTED": 0,
        "IN_PROGRESS": 0,
        "COMPLETE": 0,
        "COLLAPSED": 0,
        "UNKNOWN": 0
    }

    for run_id in range(300):
        status = get_run_status(output_dir, run_id)

        if status.startswith("IN_PROGRESS"):
            status_counts["IN_PROGRESS"] += 1
        else:
            status_counts[status] = status_counts.get(status, 0) + 1

        if args.verbose or status not in ["NOT_STARTED", "COMPLETE"]:
            print(f"Run {run_id:03d}: {status}")

    print("\n" + "="*50)
    print("Summary:")
    for status, count in sorted(status_counts.items()):
        if count > 0:
            print(f"  {status}: {count}")

    total_complete = status_counts["COMPLETE"] + status_counts["COLLAPSED"]
    print(f"\nTotal completed: {total_complete}/300 ({100*total_complete/300:.1f}%)")


def submit_command(args):
    """Submit initial job array"""
    script_dir = Path(__file__).parent
    submit_script = script_dir / "submit_hpc.sh"

    if not submit_script.exists():
        print(f"ERROR: Submit script not found: {submit_script}")
        sys.exit(1)

    # Check for sequence file
    if not Path(args.sequence_file).exists():
        print(f"ERROR: Sequence file not found: {args.sequence_file}")
        sys.exit(1)

    # Create logs directory
    logs_dir = script_dir.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Build environment variables
    env_vars = [
        f"SEQUENCE_FILE={args.sequence_file}",
        f"OUTPUT_DIR={args.output_dir}",
        f"CHUNK_SIZE={args.chunk_size}",
        f"MAX_GENERATIONS={args.max_generations}"
    ]

    if args.d2_bias:
        env_vars.append("D2_BIAS=--d2-bias")
        env_vars.append(f"D2_BIAS_STRENGTH={args.d2_bias_strength}")

    if args.no_correlation_dimension:
        env_vars.append("NO_CD=--no-correlation-dimension")

    # Build sbatch command
    cmd = ["sbatch"]

    # Add environment variable exports
    for var in env_vars:
        cmd.extend(["--export", f"ALL,{var}"])

    cmd.append(str(submit_script))

    print("Submitting job array...")
    print(f"Command: {' '.join(cmd)}")

    if not args.dry_run:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            sys.exit(1)
    else:
        print("DRY RUN - not actually submitting")


def resubmit_command(args):
    """Resubmit incomplete runs"""
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        return

    incomplete_runs = []

    for run_id in range(300):
        status = get_run_status(output_dir, run_id)
        if status not in ["COMPLETE", "COLLAPSED"]:
            incomplete_runs.append(run_id)

    if not incomplete_runs:
        print("All runs are complete or collapsed!")
        return

    print(f"Found {len(incomplete_runs)} incomplete runs")

    if args.list:
        print("Incomplete runs:", ", ".join(str(r) for r in incomplete_runs[:20]))
        if len(incomplete_runs) > 20:
            print(f"... and {len(incomplete_runs) - 20} more")
        return

    # Resubmit as array
    script_dir = Path(__file__).parent
    submit_script = script_dir / "submit_hpc.sh"

    # Create array specification
    array_spec = ",".join(str(r) for r in incomplete_runs)

    env_vars = [
        f"SEQUENCE_FILE={args.sequence_file}",
        f"OUTPUT_DIR={args.output_dir}",
        f"CHUNK_SIZE={args.chunk_size}",
        f"MAX_GENERATIONS={args.max_generations}"
    ]

    cmd = ["sbatch", f"--array={array_spec}"]

    for var in env_vars:
        cmd.extend(["--export", f"ALL,{var}"])

    cmd.append(str(submit_script))

    print(f"Resubmitting {len(incomplete_runs)} jobs...")

    if not args.dry_run:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            sys.exit(1)
    else:
        print("DRY RUN - not actually submitting")
        print(f"Would run: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage HPC evolutionary simulation runs"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Status command
    status_parser = subparsers.add_parser('status', help='Check status of runs')
    status_parser.add_argument('--output-dir', '-o', default='./output',
                              help='Output directory')
    status_parser.add_argument('--verbose', '-v', action='store_true',
                              help='Show all runs including completed')

    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit initial job array')
    submit_parser.add_argument('--sequence-file', '-s', required=True,
                              help='Input sequence file')
    submit_parser.add_argument('--output-dir', '-o', default='./output',
                              help='Output directory')
    submit_parser.add_argument('--chunk-size', type=int, default=60000,
                              help='Generations per chunk (default: 60000)')
    submit_parser.add_argument('--max-generations', type=int, default=6000000,
                              help='Max generations (default: 6000000)')
    submit_parser.add_argument('--d2-bias', action='store_true',
                              help='Enable D2 bias')
    submit_parser.add_argument('--d2-bias-strength', type=float, default=1.0,
                              help='D2 bias strength')
    submit_parser.add_argument('--no-correlation-dimension', action='store_true',
                              help='Disable correlation dimension')
    submit_parser.add_argument('--dry-run', action='store_true',
                              help='Show what would be submitted without submitting')

    # Resubmit command
    resubmit_parser = subparsers.add_parser('resubmit',
                                           help='Resubmit incomplete runs')
    resubmit_parser.add_argument('--sequence-file', '-s', required=True,
                                help='Input sequence file')
    resubmit_parser.add_argument('--output-dir', '-o', default='./output',
                                help='Output directory')
    resubmit_parser.add_argument('--chunk-size', type=int, default=60000,
                                help='Generations per chunk')
    resubmit_parser.add_argument('--max-generations', type=int, default=6000000,
                                help='Max generations')
    resubmit_parser.add_argument('--list', action='store_true',
                                help='Just list incomplete runs')
    resubmit_parser.add_argument('--dry-run', action='store_true',
                                help='Show what would be submitted')

    args = parser.parse_args()

    if args.command == 'status':
        status_command(args)
    elif args.command == 'submit':
        submit_command(args)
    elif args.command == 'resubmit':
        resubmit_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
