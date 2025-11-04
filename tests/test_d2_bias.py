#!/usr/bin/env python3
"""
Test suite for D2-biased insertion functionality.
Ensures that D2 bias changes mutation patterns as expected.
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


def file_hash(file_path):
    """Calculate SHA256 hash of file content."""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_d2_bias_changes_results(test_setup):
    """Test that enabling D2 bias produces different mutation patterns than unbiased."""
    seed = 42
    generations = 50

    # Use pre-evolved sequence so CD data is available from the start
    evolved_seq_file = "./data/2191000generation.out.fa"
    if not Path(evolved_seq_file).exists():
        pytest.skip(f"Pre-evolved sequence not found: {evolved_seq_file}")

    # Read FASTA format
    with open(evolved_seq_file, 'r') as f:
        lines = f.readlines()
    sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))

    # Run 1: No D2 bias (default behavior)
    random.seed(seed)
    np.random.seed(seed)
    seq_unbiased, records_unbiased, cenh3_unbiased, _, _, _ = introduce_mutations(
        sequence, 0, generations,
        compute_correlation_dim=False,
        use_ema_smoothing=False,
        use_d2_bias=False
    )

    # Run 2: With D2 bias enabled
    random.seed(seed)
    np.random.seed(seed)
    seq_biased, records_biased, cenh3_biased, _, _, _ = introduce_mutations(
        sequence, 0, generations,
        compute_correlation_dim=True,
        use_ema_smoothing=True,
        ema_alpha=0.3,
        use_d2_bias=True,
        d2_bias_strength=2.0
    )

    # The sequences and records should be different
    assert seq_unbiased != seq_biased, "D2 bias should produce different sequences"
    assert len(records_unbiased) != len(records_biased) or records_unbiased != records_biased, \
        "D2 bias should produce different mutation records"


def test_ema_smoothing_changes_results(test_setup):
    """Test that enabling EMA smoothing produces different results (due to different CD calculations)."""
    seed = 42
    generations = 100

    sequence = read_sequence(test_setup['input_seq'])

    # Run 1: No EMA smoothing
    random.seed(seed)
    np.random.seed(seed)
    seq_no_ema, records_no_ema, _, _, history_no_ema, smoothed_no_ema = introduce_mutations(
        sequence, 0, generations,
        compute_correlation_dim=True,
        use_ema_smoothing=False,
        use_d2_bias=False
    )

    # Run 2: With EMA smoothing (but no bias, so shouldn't affect mutations)
    random.seed(seed)
    np.random.seed(seed)
    seq_with_ema, records_with_ema, _, _, history_with_ema, smoothed_with_ema = introduce_mutations(
        sequence, 0, generations,
        compute_correlation_dim=True,
        use_ema_smoothing=True,
        ema_alpha=0.3,
        use_d2_bias=False
    )

    # Without D2 bias, sequences should be identical (EMA doesn't affect mutations)
    assert seq_no_ema == seq_with_ema, \
        "EMA smoothing without D2 bias should not change mutation results"
    assert records_no_ema == records_with_ema, \
        "EMA smoothing without D2 bias should not change mutation records"

    # But smoothed values should exist when enabled
    assert smoothed_no_ema is None, "Smoothed values should be None when EMA disabled"
    assert smoothed_with_ema is not None, "Smoothed values should exist when EMA enabled"
    assert len(smoothed_with_ema) > 0, "Smoothed values array should not be empty"


def test_d2_bias_strength_affects_results(test_setup):
    """Test that different D2 bias strengths produce different results."""
    seed = 42
    generations = 50  # Shorter test

    # Use pre-evolved sequence so CD data is available from the start
    evolved_seq_file = "./data/2191000generation.out.fa"
    if not Path(evolved_seq_file).exists():
        pytest.skip(f"Pre-evolved sequence not found: {evolved_seq_file}")

    # Read FASTA format
    with open(evolved_seq_file, 'r') as f:
        lines = f.readlines()
    sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))

    # Run with bias strength 1.0
    random.seed(seed)
    np.random.seed(seed)
    seq_strength1, records_strength1, _, _, _, _ = introduce_mutations(
        sequence, 0, generations,
        compute_correlation_dim=True,
        use_ema_smoothing=True,
        ema_alpha=0.3,
        use_d2_bias=True,
        d2_bias_strength=1.0
    )

    # Run with bias strength 3.0 (stronger bias)
    random.seed(seed)
    np.random.seed(seed)
    seq_strength3, records_strength3, _, _, _, _ = introduce_mutations(
        sequence, 0, generations,
        compute_correlation_dim=True,
        use_ema_smoothing=True,
        ema_alpha=0.3,
        use_d2_bias=True,
        d2_bias_strength=3.0
    )

    # Different bias strengths should produce different results
    assert seq_strength1 != seq_strength3, \
        "Different D2 bias strengths should produce different sequences"
    assert records_strength1 != records_strength3, \
        "Different D2 bias strengths should produce different mutation records"


def test_d2_bias_requires_ema_smoothing(test_setup):
    """Test that D2 bias uses smoothed values when available."""
    seed = 42
    generations = 50  # Shorter for faster test

    sequence = read_sequence(test_setup['input_seq'])

    # This should work but won't use bias (no smoothed values initially)
    random.seed(seed)
    np.random.seed(seed)
    seq, records, _, _, _, _ = introduce_mutations(
        sequence, 0, generations,
        compute_correlation_dim=True,
        use_ema_smoothing=False,  # No EMA smoothing
        use_d2_bias=True,          # But bias enabled (won't work without smoothed values)
        d2_bias_strength=2.0
    )

    # Should complete without error, but bias won't be applied
    assert seq is not None
    assert len(records) > 0


def test_d2_bias_deterministic_with_seed(test_setup):
    """Test that D2-biased simulations are deterministic with same seed."""
    seed = 42
    generations = 50

    sequence = read_sequence(test_setup['input_seq'])

    # Run 1
    random.seed(seed)
    np.random.seed(seed)
    seq1, records1, _, _, _, _ = introduce_mutations(
        sequence, 0, generations,
        compute_correlation_dim=True,
        use_ema_smoothing=True,
        ema_alpha=0.3,
        use_d2_bias=True,
        d2_bias_strength=2.0
    )

    # Run 2 with same seed
    random.seed(seed)
    np.random.seed(seed)
    seq2, records2, _, _, _, _ = introduce_mutations(
        sequence, 0, generations,
        compute_correlation_dim=True,
        use_ema_smoothing=True,
        ema_alpha=0.3,
        use_d2_bias=True,
        d2_bias_strength=2.0
    )

    # Should be identical
    assert seq1 == seq2, "Same seed should produce identical D2-biased sequences"
    assert records1 == records2, "Same seed should produce identical D2-biased mutation records"


def test_d2_bias_different_seeds_different_results(test_setup):
    """Test that different seeds produce different D2-biased results."""
    generations = 50

    sequence = read_sequence(test_setup['input_seq'])

    # Run with seed 42
    random.seed(42)
    np.random.seed(42)
    seq1, records1, _, _, _, _ = introduce_mutations(
        sequence, 0, generations,
        compute_correlation_dim=True,
        use_ema_smoothing=True,
        ema_alpha=0.3,
        use_d2_bias=True,
        d2_bias_strength=2.0
    )

    # Run with seed 123
    random.seed(123)
    np.random.seed(123)
    seq2, records2, _, _, _, _ = introduce_mutations(
        sequence, 0, generations,
        compute_correlation_dim=True,
        use_ema_smoothing=True,
        ema_alpha=0.3,
        use_d2_bias=True,
        d2_bias_strength=2.0
    )

    # Should be different
    assert seq1 != seq2, "Different seeds should produce different D2-biased sequences"
    assert records1 != records2, "Different seeds should produce different D2-biased mutation records"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
