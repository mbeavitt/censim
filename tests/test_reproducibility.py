#!/usr/bin/env python3
"""
Test suite for simulation reproducibility and correctness using pytest.
Ensures that identical seeds produce identical results and different seeds produce different results.
"""

import pytest
import subprocess
import hashlib
import shutil
from pathlib import Path


@pytest.fixture
def test_setup():
    """Set up test environment and clean up after."""
    test_dir = Path("./test_output")
    test_dir.mkdir(exist_ok=True)

    # Test parameters
    params = {
        'test_dir': test_dir,
        'input_seq': "./data/15000copy_cen178.seq",
        'input_pos': "./data/15000copy_cen178.178bp.bed.pos",
    }

    # Verify required files exist
    required_files = [params['input_seq'], params['input_pos']]
    for file_path in required_files:
        if not Path(file_path).exists():
            pytest.fail(f"Required file not found: {file_path}")

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
    pos_out = params['test_dir'] / f"{base_name}.pos"
    cenh3_out = params['test_dir'] / f"{base_name}.cenh3.txt"

    cmd = [
        "python", "-m", "censim.simulation",
        params['input_seq'], str(generations), params['input_pos'],
        str(fasta_out), str(record_out), str(pos_out), str(cenh3_out),
        "--seed", str(seed)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        pytest.fail(f"Simulation failed: {result.stderr}")

    return fasta_out, record_out, pos_out


def file_hash(file_path):
    """Calculate SHA256 hash of file content."""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_same_seed_same_result_100gen(test_setup):
    """Test that same seed produces identical results (100 generations)."""
    seed = 42
    generations = 100

    # Run simulation twice with same seed
    fasta1, record1, pos1 = run_simulation(test_setup, generations, seed, "_run1")
    fasta2, record2, pos2 = run_simulation(test_setup, generations, seed, "_run2")

    # Compare file contents using hashes
    assert file_hash(fasta1) == file_hash(fasta2), "FASTA outputs differ with same seed"
    assert file_hash(record1) == file_hash(record2), "Record outputs differ with same seed"
    assert file_hash(pos1) == file_hash(pos2), "Position outputs differ with same seed"


def test_same_seed_same_result_1000gen(test_setup):
    """Test that same seed produces identical results (1000 generations)."""
    seed = 123
    generations = 1000

    # Run simulation twice with same seed
    fasta1, record1, pos1 = run_simulation(test_setup, generations, seed, "_run1")
    fasta2, record2, pos2 = run_simulation(test_setup, generations, seed, "_run2")

    # Compare file contents using hashes
    assert file_hash(fasta1) == file_hash(fasta2), "FASTA outputs differ with same seed"
    assert file_hash(record1) == file_hash(record2), "Record outputs differ with same seed"
    assert file_hash(pos1) == file_hash(pos2), "Position outputs differ with same seed"


def test_different_seeds_different_results(test_setup):
    """Test that different seeds produce different results."""
    generations = 100
    seed1, seed2 = 42, 123

    # Run simulations with different seeds
    fasta1, record1, pos1 = run_simulation(test_setup, generations, seed1, "_seed1")
    fasta2, record2, pos2 = run_simulation(test_setup, generations, seed2, "_seed2")

    # Results should be different
    assert file_hash(fasta1) != file_hash(fasta2), "FASTA outputs identical with different seeds"
    assert file_hash(record1) != file_hash(record2), "Record outputs identical with different seeds"
    assert file_hash(pos1) != file_hash(pos2), "Position outputs identical with different seeds"


def test_output_file_structure(test_setup):
    """Test that output files have expected structure and content."""
    seed = 42
    generations = 50

    fasta_out, record_out, pos_out = run_simulation(test_setup, generations, seed)

    # Check that output files exist and are non-empty
    assert fasta_out.exists(), "FASTA output file not created"
    assert record_out.exists(), "Record output file not created"
    assert pos_out.exists(), "Position output file not created"

    assert fasta_out.stat().st_size > 0, "FASTA file is empty"
    assert record_out.stat().st_size > 0, "Record file is empty"
    assert pos_out.stat().st_size > 0, "Position file is empty"

    # Check position file format (should be tab-separated with 2 columns)
    with open(pos_out, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 0, "Position file has no content"

        # Check first few lines have correct format
        for i, line in enumerate(lines[:5]):
            parts = line.strip().split('\t')
            assert len(parts) == 2, f"Line {i+1} doesn't have 2 columns: {line.strip()}"
            assert parts[1].isdigit(), f"Line {i+1} second column not numeric: {parts[1]}"


def test_position_consistency(test_setup):
    """Test that position files maintain 178bp spacing."""
    seed = 42
    generations = 100

    _, _, pos_out = run_simulation(test_setup, generations, seed)

    # Read positions
    positions = []
    with open(pos_out, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            positions.append(int(parts[1]))

    # Check that positions are sorted
    assert positions == sorted(positions), "Positions are not sorted"

    # Check that all gaps are 178bp (allowing for indels that might change this)
    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]

    # Most gaps should be 178, but some variation is expected due to mutations
    bp_178_count = sum(1 for gap in gaps if gap == 178)
    total_gaps = len(gaps)

    # At least 80% should still be 178bp (adjust threshold as needed)
    ratio = bp_178_count / total_gaps if total_gaps > 0 else 0
    assert ratio > 0.8, f"Too few 178bp gaps: {bp_178_count}/{total_gaps} = {ratio:.2%}"


@pytest.mark.performance
def test_small_simulation_performance(test_setup):
    """Test that small simulations complete within reasonable time."""
    from censim.performance import benchmark_simulation

    cmd = [
        "python", "-m", "censim.simulation",
        test_setup['input_seq'], "10", test_setup['input_pos'],
        str(test_setup['test_dir'] / "perf_test.fa"),
        str(test_setup['test_dir'] / "perf_test.record.txt"),
        str(test_setup['test_dir'] / "perf_test.pos"),
        str(test_setup['test_dir'] / "perf_test.cenh3.txt"),
        "--seed", "42"
    ]

    duration, result = benchmark_simulation("small_simulation_10gen", cmd, 10, 42)

    assert result.returncode == 0, f"Simulation failed: {result.stderr}"
    assert duration < 10, f"Small simulation took too long: {duration:.2f}s"


@pytest.mark.performance
def test_medium_simulation_performance(test_setup):
    """Test performance for 100 generation simulation."""
    from censim.performance import benchmark_simulation

    cmd = [
        "python", "-m", "censim.simulation",
        test_setup['input_seq'], "100", test_setup['input_pos'],
        str(test_setup['test_dir'] / "perf_test_100.fa"),
        str(test_setup['test_dir'] / "perf_test_100.record.txt"),
        str(test_setup['test_dir'] / "perf_test_100.pos"),
        str(test_setup['test_dir'] / "perf_test_100.cenh3.txt"),
        "--seed", "42"
    ]

    duration, result = benchmark_simulation("medium_simulation_100gen", cmd, 100, 42)

    assert result.returncode == 0, f"Simulation failed: {result.stderr}"
    assert duration < 60, f"Medium simulation took too long: {duration:.2f}s"


@pytest.mark.performance
def test_large_simulation_performance(test_setup):
    """Test performance for 1000 generation simulation."""
    from censim.performance import benchmark_simulation

    cmd = [
        "python", "-m", "censim.simulation",
        test_setup['input_seq'], "1000", test_setup['input_pos'],
        str(test_setup['test_dir'] / "perf_test_1000.fa"),
        str(test_setup['test_dir'] / "perf_test_1000.record.txt"),
        str(test_setup['test_dir'] / "perf_test_1000.pos"),
        str(test_setup['test_dir'] / "perf_test_1000.cenh3.txt"),
        "--seed", "42"
    ]

    duration, result = benchmark_simulation("large_simulation_1000gen", cmd, 1000, 42)

    assert result.returncode == 0, f"Simulation failed: {result.stderr}"
    assert duration < 600, f"Large simulation took too long: {duration:.2f}s"


@pytest.mark.performance
def test_extra_large_simulation_performance(test_setup):
    """Test performance for 10,000 generation simulation."""
    from censim.performance import benchmark_simulation

    cmd = [
        "python", "-m", "censim.simulation",
        test_setup['input_seq'], "10000", test_setup['input_pos'],
        str(test_setup['test_dir'] / "perf_test_10000.fa"),
        str(test_setup['test_dir'] / "perf_test_10000.record.txt"),
        str(test_setup['test_dir'] / "perf_test_10000.pos"),
        str(test_setup['test_dir'] / "perf_test_10000.cenh3.txt"),
        "--seed", "42"
    ]

    duration, result = benchmark_simulation("extra_large_simulation_10000gen", cmd, 10000, 42)

    assert result.returncode == 0, f"Simulation failed: {result.stderr}"
    assert duration < 3600, f"Extra large simulation took too long: {duration:.2f}s"


@pytest.mark.parametrize("seed,generations", [
    (42, 50),
    (123, 100),
    (999, 75),
])
def test_multiple_seeds_reproducibility(test_setup, seed, generations):
    """Test reproducibility across multiple seed/generation combinations."""
    # Run twice with same parameters
    fasta1, record1, pos1 = run_simulation(test_setup, generations, seed, "_run1")
    fasta2, record2, pos2 = run_simulation(test_setup, generations, seed, "_run2")

    # Should be identical
    assert file_hash(fasta1) == file_hash(fasta2), f"Results differ for seed={seed}, gen={generations}"
    assert file_hash(record1) == file_hash(record2), f"Records differ for seed={seed}, gen={generations}"
    assert file_hash(pos1) == file_hash(pos2), f"Positions differ for seed={seed}, gen={generations}"


def test_seed_none_is_random(test_setup):
    """Test that not providing a seed produces different results each time."""
    generations = 50

    # Run without seed twice
    cmd1 = [
        "python", "-m", "censim.simulation",
        test_setup['input_seq'], str(generations), test_setup['input_pos'],
        str(test_setup['test_dir'] / "random1.fa"),
        str(test_setup['test_dir'] / "random1.record.txt"),
        str(test_setup['test_dir'] / "random1.pos"),
        str(test_setup['test_dir'] / "random1.cenh3.txt")
    ]

    cmd2 = [
        "python", "-m", "censim.simulation",
        test_setup['input_seq'], str(generations), test_setup['input_pos'],
        str(test_setup['test_dir'] / "random2.fa"),
        str(test_setup['test_dir'] / "random2.record.txt"),
        str(test_setup['test_dir'] / "random2.pos"),
        str(test_setup['test_dir'] / "random2.cenh3.txt")
    ]

    result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=60)
    result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)

    assert result1.returncode == 0, f"First random simulation failed: {result1.stderr}"
    assert result2.returncode == 0, f"Second random simulation failed: {result2.stderr}"

    # Results should likely be different (though theoretically could be same)
    record1_hash = file_hash(test_setup['test_dir'] / "random1.record.txt")
    record2_hash = file_hash(test_setup['test_dir'] / "random2.record.txt")

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
    pos_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.pos"

    # Check reference files exist
    assert fasta_ref.exists(), f"Reference FASTA not found: {fasta_ref}"
    assert record_ref.exists(), f"Reference record not found: {record_ref}"
    assert pos_ref.exists(), f"Reference position not found: {pos_ref}"

    # Run new simulation
    fasta_new, record_new, pos_new = run_simulation(test_setup, generations, seed, "_regression")

    # Compare against reference
    assert file_hash(fasta_new) == file_hash(fasta_ref), \
        f"FASTA output differs from reference for {description}"
    assert file_hash(record_new) == file_hash(record_ref), \
        f"Record output differs from reference for {description}"
    assert file_hash(pos_new) == file_hash(pos_ref), \
        f"Position output differs from reference for {description}"


@pytest.mark.regression
def test_regression_against_reference_medium(test_setup, reference_data_dir):
    """Test that simulation produces identical results to reference data (medium case)."""
    seed = 123
    generations = 100
    description = "medium_test"

    # Reference files
    fasta_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.fa"
    record_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.record.txt"
    pos_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.pos"

    # Check reference files exist
    assert fasta_ref.exists(), f"Reference FASTA not found: {fasta_ref}"
    assert record_ref.exists(), f"Reference record not found: {record_ref}"
    assert pos_ref.exists(), f"Reference position not found: {pos_ref}"

    # Run new simulation
    fasta_new, record_new, pos_new = run_simulation(test_setup, generations, seed, "_regression")

    # Compare against reference
    assert file_hash(fasta_new) == file_hash(fasta_ref), \
        f"FASTA output differs from reference for {description}"
    assert file_hash(record_new) == file_hash(record_ref), \
        f"Record output differs from reference for {description}"
    assert file_hash(pos_new) == file_hash(pos_ref), \
        f"Position output differs from reference for {description}"


@pytest.mark.regression
def test_regression_against_reference_large(test_setup, reference_data_dir):
    """Test that simulation produces identical results to reference data (large case)."""
    seed = 999
    generations = 200
    description = "large_test"

    # Reference files
    fasta_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.fa"
    record_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.record.txt"
    pos_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.pos"

    # Check reference files exist
    assert fasta_ref.exists(), f"Reference FASTA not found: {fasta_ref}"
    assert record_ref.exists(), f"Reference record not found: {record_ref}"
    assert pos_ref.exists(), f"Reference position not found: {pos_ref}"

    # Run new simulation
    fasta_new, record_new, pos_new = run_simulation(test_setup, generations, seed, "_regression")

    # Compare against reference
    assert file_hash(fasta_new) == file_hash(fasta_ref), \
        f"FASTA output differs from reference for {description}"
    assert file_hash(record_new) == file_hash(record_ref), \
        f"Record output differs from reference for {description}"
    assert file_hash(pos_new) == file_hash(pos_ref), \
        f"Position output differs from reference for {description}"


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
    pos_ref = reference_data_dir / f"{description}_seed{seed}_{generations}gen.pos"

    # Skip if reference files don't exist
    if not all([fasta_ref.exists(), record_ref.exists(), pos_ref.exists()]):
        pytest.skip(f"Reference files missing for {description}")

    # Run new simulation
    fasta_new, record_new, pos_new = run_simulation(test_setup, generations, seed, f"_{description}_regression")

    # Compare against reference
    assert file_hash(fasta_new) == file_hash(fasta_ref), \
        f"FASTA output differs from reference for {description}"
    assert file_hash(record_new) == file_hash(record_ref), \
        f"Record output differs from reference for {description}"
    assert file_hash(pos_new) == file_hash(pos_ref), \
        f"Position output differs from reference for {description}"