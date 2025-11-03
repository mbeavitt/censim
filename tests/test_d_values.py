#!/usr/bin/env python3
"""
Test suite for d_values (correlation dimension) regression testing.
Ensures that d_values computation remains consistent across code changes.
"""

import pytest
import hashlib
import pickle
from pathlib import Path
import random
import numpy as np

from censim.simulation import (
    RepeatSequence,
    compute_correlation_dimension,
    read_sequence,
    introduce_mutations
)


def read_fasta(filepath):
    """Read FASTA file and return sequence."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Skip header lines starting with '>'
    sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
    return sequence


@pytest.fixture
def evolved_sequence():
    """Fixture for evolved sequence data (for regression test only)."""
    evolved_seq_file = "./data/2191000generation.out.fa"

    # Verify required files exist
    if not Path(evolved_seq_file).exists():
        pytest.skip(f"Required file not found: {evolved_seq_file}")

    # Read the evolved sequence
    sequence = read_fasta(evolved_seq_file)

    # Convert to RepeatSequence
    repeat_seq = RepeatSequence(sequence, repeat_size=178)

    return repeat_seq


@pytest.fixture
def test_setup():
    """Set up test environment for simulation-based tests."""
    params = {
        'input_seq': "./data/15000copy_cen178.seq",
    }

    # Verify required files exist
    if not Path(params['input_seq']).exists():
        pytest.skip(f"Required file not found: {params['input_seq']}")

    return params


def d_values_hash(d_values):
    """
    Calculate hash of d_values for regression testing.

    Args:
        d_values: numpy array

    Returns:
        str: SHA256 hash of the serialized d_values
    """
    # Serialize the d_values to bytes
    serialized = pickle.dumps(d_values)
    return hashlib.sha256(serialized).hexdigest()


def test_d_values_regression(evolved_sequence):
    """
    Test that d_values computation produces consistent results on evolved sequence data.

    This test computes d_values on a pre-evolved sequence and checks that
    the resulting d_values match a known reference hash. If this test fails,
    it indicates that the correlation dimension calculation has changed.
    """
    # Reference hash computed from a known good run
    # This hash was generated on: 2025-11-03 (with vectorized batch estimate_D2_from_C_r_batch)
    # Update this hash if you intentionally change the d_values computation
    REFERENCE_HASH = "c808b5a2fd45f8ef83a47d05b0198fe5c401ddc93131fc308344f4514b8d5916"

    # Compute d_values on the evolved sequence
    d_values = compute_correlation_dimension(evolved_sequence, repeat_len=178)

    # Compute hash of d_values
    computed_hash = d_values_hash(d_values)

    # Check against reference
    assert computed_hash == REFERENCE_HASH, (
        f"d_values hash mismatch!\n"
        f"Expected: {REFERENCE_HASH}\n"
        f"Got:      {computed_hash}\n"
        f"This indicates that the correlation dimension calculation has changed.\n"
        f"If this change is intentional, update REFERENCE_HASH in the test."
    )

    # Additional sanity checks
    assert isinstance(d_values, np.ndarray), "d_values should be a numpy array"
    assert len(d_values) > 0, "d_values array is empty"
    assert np.all(np.isfinite(d_values)), "d_values contains non-finite values"


def test_d_values_structure(test_setup):
    """Test that d_values_history has the expected structure."""
    seed = 42
    generations = 5  # Quick test

    random.seed(seed)
    np.random.seed(seed)

    sequence = read_sequence(test_setup['input_seq'])
    mutated_sequence, mutation_records, cenh3_occupancy, collapsed, d_values_history = introduce_mutations(
        sequence, 0, generations
    )

    # Check that we got d_values for each generation
    assert len(d_values_history) == generations

    # Check that each entry is a numpy array
    for i, d_values in enumerate(d_values_history):
        assert isinstance(d_values, np.ndarray), f"Generation {i}: not a numpy array"
        assert d_values.dtype == np.float64 or d_values.dtype == np.float32, (
            f"Generation {i}: unexpected dtype {d_values.dtype}"
        )


def test_d_values_deterministic(test_setup):
    """Test that d_values computation is deterministic given same seed."""
    seed = 42
    generations = 10

    # Run 1
    random.seed(seed)
    np.random.seed(seed)
    sequence = read_sequence(test_setup['input_seq'])
    _, _, _, _, d_values_history1 = introduce_mutations(sequence, 0, generations)

    # Run 2
    random.seed(seed)
    np.random.seed(seed)
    sequence = read_sequence(test_setup['input_seq'])
    _, _, _, _, d_values_history2 = introduce_mutations(sequence, 0, generations)

    # Should be identical
    assert len(d_values_history1) == len(d_values_history2)
    for i, (d1, d2) in enumerate(zip(d_values_history1, d_values_history2)):
        np.testing.assert_array_equal(d1, d2,
            err_msg=f"Generation {i}: d_values differ with same seed")


@pytest.mark.performance
def test_d_values_computation_performance(test_setup):
    """Test performance of simulation with d_values computation (30 generations)."""
    import time
    from censim.performance import tracker

    generations = 30
    seed = 42

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Read inputs
    sequence = read_sequence(test_setup['input_seq'])

    # Time the simulation with d_values computation
    start = time.time()
    mutated_sequence, mutation_records, cenh3_occupancy, collapsed, d_values_history = introduce_mutations(
        sequence, 0, generations
    )
    duration = time.time() - start

    # Record performance
    tracker.record_benchmark("d_values_simulation_30gen", duration, generations, seed)

    # Verify d_values were computed
    assert len(d_values_history) == generations, "d_values not computed for all generations"

    # Performance assertion - should complete within reasonable time
    # 30 generations with d_values computation is more expensive than without
    assert duration < 120, f"D_values simulation took too long: {duration:.2f}s"


if __name__ == "__main__":
    # Helper script to generate the reference hash
    print("Generating reference hash for d_values computation on evolved sequence...")

    # Load the evolved sequence
    evolved_seq_file = "./data/2191000generation.out.fa"
    if not Path(evolved_seq_file).exists():
        print(f"Error: {evolved_seq_file} not found")
        exit(1)

    sequence = read_fasta(evolved_seq_file)
    print(f"Loaded sequence of length {len(sequence)}")

    # Convert to RepeatSequence
    repeat_seq = RepeatSequence(sequence, repeat_size=178)
    print(f"Converted to RepeatSequence with {repeat_seq.num_units()} units")

    # Compute d_values
    print("Computing d_values...")
    d_values = compute_correlation_dimension(repeat_seq, repeat_len=178)

    reference_hash = d_values_hash(d_values)
    print(f"\nReference hash:")
    print(f"  {reference_hash}")
    print(f"\nUpdate REFERENCE_HASH in test_d_values_regression() with this value.")
    print(f"\nD_values stats:")
    print(f"  Shape: {d_values.shape}")
    print(f"  Mean:  {np.mean(d_values):.6f}")
    print(f"  Std:   {np.std(d_values):.6f}")
    print(f"  Min:   {np.min(d_values):.6f}")
    print(f"  Max:   {np.max(d_values):.6f}")
