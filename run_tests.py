#!/usr/bin/env python3
"""
Test runner script for simulation reproducibility tests.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run tests with appropriate options."""
    print("Running Simulation Reproducibility Tests")
    print("=" * 50)

    # Check if pytest is available
    try:
        subprocess.run(["python", "-m", "pytest", "--version"],
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: pytest not found. Install with: pip install pytest")
        return 1

    # Basic test run
    print("\n1. Running basic reproducibility tests...")
    cmd = ["python", "-m", "pytest", "test_reproducibility.py", "-v"]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("❌ Basic tests failed!")
        return result.returncode

    print("✅ Basic tests passed!")

    # Performance tests (optional)
    print("\n2. Running performance tests...")
    perf_cmd = ["python", "-m", "pytest", "test_reproducibility.py",
                "-v", "-m", "performance"]
    perf_result = subprocess.run(perf_cmd)

    if perf_result.returncode == 0:
        print("✅ Performance tests passed!")
    else:
        print("⚠️  Performance tests had issues (non-critical)")

    # Regression tests (if reference data exists)
    ref_dir = Path("./test_reference_data")
    if ref_dir.exists():
        print("\n3. Running regression tests against reference data...")
        regression_cmd = ["python", "-m", "pytest", "test_reproducibility.py",
                         "-v", "-m", "regression"]
        regression_result = subprocess.run(regression_cmd)

        if regression_result.returncode == 0:
            print("✅ Regression tests passed!")
        else:
            print("❌ Regression tests failed - simulation behavior has changed!")
            print("  If this is intentional, regenerate reference data:")
            print("  python generate_reference_data.py")
            return regression_result.returncode
    else:
        print("\n3. Skipping regression tests (no reference data)")
        print("  Generate reference data with: python generate_reference_data.py")

    # Quick test run (for CI/development)
    print("\n4. Quick test summary:")
    summary_cmd = ["python", "-m", "pytest", "test_reproducibility.py",
                   "--tb=line", "-q", "-m", "not regression"]
    subprocess.run(summary_cmd)

    print("\n" + "=" * 50)
    print("✅ All critical tests completed successfully!")
    print("\nUsage tips:")
    print("  - Run specific test: python -m pytest test_reproducibility.py::test_same_seed_same_result_100gen -v")
    print("  - Skip performance: python -m pytest test_reproducibility.py -m 'not performance' -v")
    print("  - Run regression only: python -m pytest test_reproducibility.py -m regression -v")
    print("  - Generate reference: python generate_reference_data.py")
    print("  - Run with coverage: python -m pytest test_reproducibility.py --cov=bin --cov-report=html")

    return 0


if __name__ == "__main__":
    sys.exit(main())