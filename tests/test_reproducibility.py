#!/usr/bin/env python3
"""
Test suite for simulation reproducibility and correctness using pytest.
Ensures that identical seeds produce identical results and different seeds produce different results.
"""

import pytest
import hashlib
import shutil
from pathlib import Path
import random
import numpy as np

from censim.simulation import read_sequence, introduce_mutations


@pytest.fixture
def test_setup():
    """Set up test environment and clean up after."""
    test_dir = Path("./test_output")
    test_dir.mkdir(exist_ok=True)

    # Test parameters
    params = {
        'test_dir': test_dir,
        'input_seq': "./data/15000copy_cen178.seq",
    }

    # Verify required files exist
    if not Path(params['input_seq']).exists():
        pytest.fail(f"Required file not found: {params['input_seq']}")

    yield params

    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)


def run_simulation(test_setup, generations, seed, suffix=""):
    """Run simulation with given parameters and return output file paths."""
    params = test_setup
    base_name = f"{generations}gen_seed{seed}{suffix}"

    fasta_out = params['test_dir'] / f"{base_name}.fa"
    record_out = params['test_dir'] / f"{base_name}.record.txt"
    cenh3_out = params['test_dir'] / f"{base_name}.cenh3.txt"

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)

    # Read input files
    sequence = read_sequence(params['input_seq'])

    # Run simulation
    mutated_sequence, mutation_records, cenh3_occupancy, collapsed = introduce_mutations(
        sequence, 0, generations
    )

    # Write output files
    with open(fasta_out, "w") as f:
        f.write(mutated_sequence + '\n')

    with open(record_out, "w") as f:
        for record in mutation_records:
            gen, mut_type, idx, ref, mut, copy_num = record
            f.write(f"{gen}, {mut_type}, {idx}, {ref}, {mut}, {copy_num}\n")

    with open(cenh3_out, "w") as f:
        for idx, occupied in enumerate(cenh3_occupancy):
            if occupied:
                f.write(f"{idx}\t1\n")
            else:
                f.write(f"{idx}\t0\n")

    return fasta_out, record_out


def file_hash(file_path):
    """Calculate SHA256 hash of file content."""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_same_seed_same_result_100gen(test_setup):
    """Test that same seed produces identical results (100 generations)."""
    seed = 42
    generations = 100

    # Run simulation twice with same seed
    fasta1, record1 = run_simulation(test_setup, generations, seed, "_run1")
    fasta2, record2 = run_simulation(test_setup, generations, seed, "_run2")

    # Compare file contents using hashes
    assert file_hash(fasta1) == file_hash(fasta2), "FASTA outputs differ with same seed"
    assert file_hash(record1) == file_hash(record2), "Record outputs differ with same seed"


def test_same_seed_same_result_1000gen(test_setup):
    """Test that same seed produces identical results (1000 generations)."""
    seed = 123
    generations = 1000

    # Run simulation twice with same seed
    fasta1, record1 = run_simulation(test_setup, generations, seed, "_run1")
    fasta2, record2 = run_simulation(test_setup, generations, seed, "_run2")

    # Compare file contents using hashes
    assert file_hash(fasta1) == file_hash(fasta2), "FASTA outputs differ with same seed"
    assert file_hash(record1) == file_hash(record2), "Record outputs differ with same seed"


def test_different_seeds_different_results(test_setup):
    """Test that different seeds produce different results."""
    generations = 100
    seed1, seed2 = 42, 123

    # Run simulations with different seeds
    fasta1, record1 = run_simulation(test_setup, generations, seed1, "_seed1")
    fasta2, record2 = run_simulation(test_setup, generations, seed2, "_seed2")

    # Results should be different
    assert file_hash(fasta1) != file_hash(fasta2), "FASTA outputs identical with different seeds"
    assert file_hash(record1) != file_hash(record2), "Record outputs identical with different seeds"


def test_output_file_structure(test_setup):
    """Test that output files have expected structure and content."""
    seed = 42
    generations = 50

    fasta_out, record_out = run_simulation(test_setup, generations, seed)

    # Check that output files exist and are non-empty
    assert fasta_out.exists(), "FASTA output file not created"
    assert record_out.exists(), "Record output file not created"

    assert fasta_out.stat().st_size > 0, "FASTA file is empty"
    assert record_out.stat().st_size > 0, "Record file is empty"

@pytest.mark.performance
def test_small_simulation_performance(test_setup):
    """Test that small simulations complete within reasonable time."""
    import time
    from censim.performance import tracker

    generations = 10
    seed = 42

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Read inputs
    sequence = read_sequence(test_setup['input_seq'])

    # Time the simulation
    start = time.time()
    mutated_sequence, mutation_records, cenh3_occupancy, collapsed = introduce_mutations(
        sequence, 0, generations
    )
    duration = time.time() - start

    # Record performance
    tracker.record_benchmark("small_simulation_10gen", duration, generations, seed)

    # Write outputs (not timed, but needed for test completeness)
    with open(test_setup['test_dir'] / "perf_test.fa", "w") as f:
        f.write(f">centro_{generations}gen\n{mutated_sequence}\n")

    assert duration < 10, f"Small simulation took too long: {duration:.2f}s"


@pytest.mark.performance
def test_medium_simulation_performance(test_setup):
    """Test performance for 100 generation simulation."""
    import time
    from censim.performance import tracker

    generations = 100
    seed = 42

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Read inputs
    sequence = read_sequence(test_setup['input_seq'])

    # Time the simulation
    start = time.time()
    mutated_sequence, mutation_records, cenh3_occupancy, collapsed = introduce_mutations(
        sequence, 0, generations
    )
    duration = time.time() - start

    # Record performance
    tracker.record_benchmark("medium_simulation_100gen", duration, generations, seed)

    # Write outputs
    with open(test_setup['test_dir'] / "perf_test_100.fa", "w") as f:
        f.write(f">centro_{generations}gen\n{mutated_sequence}\n")

    assert duration < 60, f"Medium simulation took too long: {duration:.2f}s"


@pytest.mark.performance
def test_large_simulation_performance(test_setup):
    """Test performance for 1000 generation simulation."""
    import time
    from censim.performance import tracker

    generations = 1000
    seed = 42

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Read inputs
    sequence = read_sequence(test_setup['input_seq'])

    # Time the simulation
    start = time.time()
    mutated_sequence, mutation_records, cenh3_occupancy, collapsed = introduce_mutations(
        sequence, 0, generations
    )
    duration = time.time() - start

    # Record performance
    tracker.record_benchmark("large_simulation_1000gen", duration, generations, seed)

    # Write outputs
    with open(test_setup['test_dir'] / "perf_test_1000.fa", "w") as f:
        f.write(f">centro_{generations}gen\n{mutated_sequence}\n")

    assert duration < 600, f"Large simulation took too long: {duration:.2f}s"


@pytest.mark.performance
def test_extra_large_simulation_performance(test_setup):
    """Test performance for 10,000 generation simulation."""
    import time
    from censim.performance import tracker

    generations = 10000
    seed = 42

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Read inputs
    sequence = read_sequence(test_setup['input_seq'])

    # Time the simulation
    start = time.time()
    mutated_sequence, mutation_records, cenh3_occupancy, collapsed = introduce_mutations(
        sequence, 0, generations
    )
    duration = time.time() - start

    # Record performance
    tracker.record_benchmark("extra_large_simulation_10000gen", duration, generations, seed)

    # Write outputs
    with open(test_setup['test_dir'] / "perf_test_10000.fa", "w") as f:
        f.write(f">centro_{generations}gen\n{mutated_sequence}\n")

    assert duration < 3600, f"Extra large simulation took too long: {duration:.2f}s"


@pytest.mark.performance
def test_massive_simulation_performance(test_setup):
    """Test performance for 100,000 generation simulation."""
    import time
    from censim.performance import tracker

    generations = 100000
    seed = 42

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Read inputs
    sequence = read_sequence(test_setup['input_seq'])

    # Time the simulation
    start = time.time()
    mutated_sequence, mutation_records, cenh3_occupancy, collapsed = introduce_mutations(
        sequence, 0, generations
    )
    duration = time.time() - start

    # Record performance
    tracker.record_benchmark("massive_simulation_100000gen", duration, generations, seed)

    # Write outputs
    with open(test_setup['test_dir'] / "perf_test_100000.fa", "w") as f:
        f.write(f">centro_{generations}gen\n{mutated_sequence}\n")

    assert duration < 7200, f"Massive simulation took too long: {duration:.2f}s (max: 2 hours)"


@pytest.mark.parametrize("seed,generations", [
    (42, 50),
    (123, 100),
    (999, 75),
])
def test_multiple_seeds_reproducibility(test_setup, seed, generations):
    """Test reproducibility across multiple seed/generation combinations."""
    # Run twice with same parameters
    fasta1, record1 = run_simulation(test_setup, generations, seed, "_run1")
    fasta2, record2 = run_simulation(test_setup, generations, seed, "_run2")

    # Should be identical
    assert file_hash(fasta1) == file_hash(fasta2), f"Results differ for seed={seed}, gen={generations}"
    assert file_hash(record1) == file_hash(record2), f"Records differ for seed={seed}, gen={generations}"


def test_seed_none_is_random(test_setup):
    """Test that not providing a seed produces different results each time."""
    generations = 50

    # Read input files once
    sequence = read_sequence(test_setup['input_seq'])

    # Run without explicit seed twice (library will use random state)
    fasta_out1 = test_setup['test_dir'] / "random1.fa"
    record_out1 = test_setup['test_dir'] / "random1.record.txt"
    cenh3_out1 = test_setup['test_dir'] / "random1.cenh3.txt"

    # First run - reset to a random state
    random.seed(None)
    np.random.seed(None)
    mutated_sequence1, mutation_records1, cenh3_occupancy1, _ = introduce_mutations(
        sequence, 0, generations
    )

    with open(fasta_out1, "w") as f:
        f.write(mutated_sequence1 + '\n')
    with open(record_out1, "w") as f:
        for record in mutation_records1:
            gen, mut_type, idx, ref, mut, copy_num = record
            f.write(f"{gen}, {mut_type}, {idx}, {ref}, {mut}, {copy_num}\n")
    with open(cenh3_out1, "w") as f:
        for idx, occupied in enumerate(cenh3_occupancy1):
            f.write(f"{idx}\t{'1' if occupied else '0'}\n")

    # Second run - reset to a different random state
    fasta_out2 = test_setup['test_dir'] / "random2.fa"
    record_out2 = test_setup['test_dir'] / "random2.record.txt"
    cenh3_out2 = test_setup['test_dir'] / "random2.cenh3.txt"

    random.seed(None)
    np.random.seed(None)
    mutated_sequence2, mutation_records2, cenh3_occupancy2, _ = introduce_mutations(
        sequence, 0, generations
    )

    with open(fasta_out2, "w") as f:
        f.write(mutated_sequence2 + '\n')
    with open(record_out2, "w") as f:
        for record in mutation_records2:
            gen, mut_type, idx, ref, mut, copy_num = record
            f.write(f"{gen}, {mut_type}, {idx}, {ref}, {mut}, {copy_num}\n")
    with open(cenh3_out2, "w") as f:
        for idx, occupied in enumerate(cenh3_occupancy2):
            f.write(f"{idx}\t{'1' if occupied else '0'}\n")

    # Results should likely be different (though theoretically could be same)
    record1_hash = file_hash(record_out1)
    record2_hash = file_hash(record_out2)

    # This test might occasionally fail due to random chance, but very unlikely
    assert record1_hash != record2_hash, "Two unseeded runs produced identical results (very unlikely but possible)"


# ============================================================================
# REGRESSION TESTS AGAINST REFERENCE DATA
# ============================================================================

@pytest.fixture
def reference_data_dir():
    """Path to reference data directory."""
    ref_dir = Path("./test_reference_data")
    if not ref_dir.exists():
        pytest.skip("Reference data not found. Run generate_reference_data.py first.")
    return ref_dir


@pytest.mark.regression
def test_regression_against_reference_small(test_setup, reference_data_dir):
    """Test that simulation produces identical results to reference data (small case)."""
    seed = 42
    generations = 50
    description = "small_test"

    # Reference files
    fasta_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.fa"
    record_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.record.txt"

    # Check reference files exist
    assert fasta_ref.exists(), f"Reference FASTA not found: {fasta_ref}"
    assert record_ref.exists(), f"Reference record not found: {record_ref}"

    # Run new simulation
    fasta_new, record_new = run_simulation(test_setup, generations, seed, "_regression")

    # Compare against reference
    assert file_hash(fasta_new) == file_hash(fasta_ref), \
        f"FASTA output differs from reference for {description}"
    assert file_hash(record_new) == file_hash(record_ref), \
        f"Record output differs from reference for {description}"


@pytest.mark.regression
def test_regression_against_reference_medium(test_setup, reference_data_dir):
    """Test that simulation produces identical results to reference data (medium case)."""
    seed = 123
    generations = 100
    description = "medium_test"

    # Reference files
    fasta_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.fa"
    record_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.record.txt"

    # Check reference files exist
    assert fasta_ref.exists(), f"Reference FASTA not found: {fasta_ref}"
    assert record_ref.exists(), f"Reference record not found: {record_ref}"

    # Run new simulation
    fasta_new, record_new, = run_simulation(test_setup, generations, seed, "_regression")

    # Compare against reference
    assert file_hash(fasta_new) == file_hash(fasta_ref), \
        f"FASTA output differs from reference for {description}"
    assert file_hash(record_new) == file_hash(record_ref), \
        f"Record output differs from reference for {description}"


@pytest.mark.regression
def test_regression_against_reference_large(test_setup, reference_data_dir):
    """Test that simulation produces identical results to reference data (large case)."""
    seed = 999
    generations = 200
    description = "large_test"

    # Reference files
    fasta_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.fa"
    record_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.record.txt"

    # Check reference files exist
    assert fasta_ref.exists(), f"Reference FASTA not found: {fasta_ref}"
    assert record_ref.exists(), f"Reference record not found: {record_ref}"

    # Run new simulation
    fasta_new, record_new = run_simulation(test_setup, generations, seed, "_regression")

    # Compare against reference
    assert file_hash(fasta_new) == file_hash(fasta_ref), \
        f"FASTA output differs from reference for {description}"
    assert file_hash(record_new) == file_hash(record_ref), \
        f"Record output differs from reference for {description}"


@pytest.mark.regression
@pytest.mark.parametrize("test_case", [
    ("small_test", 42, 50),
    ("medium_test", 123, 100),
    ("large_test", 999, 200),
])
def test_all_regression_cases(test_setup, reference_data_dir, test_case):
    """Parametrized test for all regression cases."""
    description, seed, generations = test_case

    # Reference files
    fasta_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.fa"
    record_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.record.txt"

    # Skip if reference files don't exist
    if not all([fasta_ref.exists(), record_ref.exists()]):
        pytest.skip(f"Reference files missing for {description}")

    # Run new simulation
    fasta_new, record_new = run_simulation(test_setup, generations, seed, f"_{description}_regression")

    # Compare against reference
    assert file_hash(fasta_new) == file_hash(fasta_ref), \
        f"FASTA output differs from reference for {description}"
    assert file_hash(record_new) == file_hash(record_ref), \
        f"Record output differs from reference for {description}"
