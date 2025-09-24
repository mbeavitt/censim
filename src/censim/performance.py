"""
Performance tracking and benchmarking utilities.
"""

import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import subprocess
import sys


class PerformanceTracker:
    """Track performance metrics in individual log files."""

    def __init__(self, logs_dir: str = "logs/performance"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _get_code_hash(self) -> str:
        """Generate a hash of the simulation code to detect changes."""
        sim_file = Path("src/censim/simulation.py")
        if sim_file.exists():
            with open(sim_file, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()[:8]
        return "unknown"

    def _get_git_commit(self) -> str:
        """Get current git commit hash (short form)."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return "unknown"

    def record_benchmark(self, test_name: str, duration: float,
                        generations: int, seed: int = None) -> Dict:
        """Record a benchmark result to individual log file."""
        timestamp = datetime.now()

        git_commit = self._get_git_commit()

        entry = {
            "test_name": test_name,
            "timestamp": timestamp.isoformat(),
            "duration": duration,
            "generations": generations,
            "seed": seed,
            "code_hash": self._get_code_hash(),
            "git_commit": git_commit,
            "python_version": sys.version.split()[0]
        }

        # Create filename with timestamp and git commit
        filename = f"{test_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{git_commit}.json"
        log_file = self.logs_dir / filename

        with open(log_file, 'w') as f:
            json.dump(entry, f, indent=2)

        # Also look for previous runs and show comparison
        self._print_comparison(test_name, entry)

        return entry

    def _print_comparison(self, test_name: str, current: Dict):
        """Print comparison with previous runs."""
        # Find previous runs for this test
        pattern = f"{test_name}_*.json"
        previous_files = sorted(self.logs_dir.glob(pattern))

        if len(previous_files) <= 1:
            print(f"PERF {test_name}: {current['duration']:.2f}s (first run)")
            return

        # Load the previous run (second to last file)
        try:
            with open(previous_files[-2], 'r') as f:
                previous = json.load(f)

            if (previous["generations"] == current["generations"] and
                previous.get("seed") == current.get("seed")):

                duration_change = current["duration"] - previous["duration"]
                percent_change = (duration_change / previous["duration"]) * 100

                if duration_change < 0:
                    print(f"PERF {test_name}: {current['duration']:.2f}s "
                          f"({percent_change:+.1f}% faster)")
                else:
                    print(f"PERF {test_name}: {current['duration']:.2f}s "
                          f"({percent_change:+.1f}% slower)")
            else:
                print(f"PERF {test_name}: {current['duration']:.2f}s (different parameters)")

        except (json.JSONDecodeError, IOError):
            print(f"PERF {test_name}: {current['duration']:.2f}s (comparison failed)")


# Global tracker instance
tracker = PerformanceTracker()


def benchmark_simulation(test_name: str, cmd: List[str],
                        generations: int, seed: int = None) -> Tuple[float, subprocess.CompletedProcess]:
    """Benchmark a simulation command and record results."""
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    duration = time.time() - start_time

    tracker.record_benchmark(test_name, duration, generations, seed)

    return duration, result