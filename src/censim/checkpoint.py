"""Checkpoint management for HPC simulations.

This module handles saving and loading simulation state to disk,
enabling restart-based workflows for SLURM job arrays.
"""

import json
import gzip
from pathlib import Path


class SimulationCheckpoint:
    """Manage simulation checkpoints for HPC restart workflows."""

    def __init__(self, run_id, checkpoint_dir="checkpoints"):
        """Initialize checkpoint manager.

        Args:
            run_id: Unique identifier for this simulation run
            checkpoint_dir: Directory to store checkpoint files
        """
        self.run_id = run_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, generation):
        """Get path to checkpoint file for a specific generation.

        Args:
            generation: Generation number

        Returns:
            Path object for checkpoint file
        """
        return self.checkpoint_dir / f"run_{self.run_id}_gen_{generation}.ckpt.gz"

    def get_latest_checkpoint(self):
        """Find the most recent checkpoint for this run.

        Returns:
            tuple: (generation, checkpoint_path) or (0, None) if no checkpoint exists
        """
        checkpoints = sorted(self.checkpoint_dir.glob(f"run_{self.run_id}_gen_*.ckpt.gz"))
        if not checkpoints:
            return 0, None

        # Extract generation number from filename
        latest = checkpoints[-1]
        gen_str = latest.stem.split("_gen_")[1].split(".")[0]
        generation = int(gen_str)

        return generation, latest

    def save_checkpoint(self, generation, sequence, mutation_records, metadata=None):
        """Save simulation state to checkpoint file.

        Args:
            generation: Current generation number
            sequence: DNA sequence string
            mutation_records: List of mutation records [(gen, type, idx, ref, mut, copy_num), ...]
            metadata: Optional dictionary of additional metadata to save
        """
        checkpoint_path = self.get_checkpoint_path(generation)

        # Prepare checkpoint data
        checkpoint = {
            "run_id": self.run_id,
            "generation": generation,
            "sequence": sequence,
            "mutation_records": mutation_records,
            "metadata": metadata or {}
        }

        # Save as compressed JSON for efficient storage
        with gzip.open(checkpoint_path, "wt", encoding="utf-8") as f:
            json.dump(checkpoint, f)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path=None):
        """Load simulation state from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file. If None, loads latest checkpoint.

        Returns:
            dict: Checkpoint data with keys: run_id, generation, sequence,
                  mutation_records, metadata

        Raises:
            FileNotFoundError: If no checkpoint exists
        """
        if checkpoint_path is None:
            generation, checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint found for run_id={self.run_id}")

        with gzip.open(checkpoint_path, "rt", encoding="utf-8") as f:
            checkpoint = json.load(f)

        return checkpoint

    def cleanup_old_checkpoints(self, keep_last_n=3):
        """Remove old checkpoint files, keeping only the most recent N.

        Args:
            keep_last_n: Number of most recent checkpoints to keep (default: 3)
        """
        checkpoints = sorted(self.checkpoint_dir.glob(f"run_{self.run_id}_gen_*.ckpt.gz"))

        # Remove all but the last N checkpoints
        for ckpt in checkpoints[:-keep_last_n]:
            ckpt.unlink()
