#!/usr/bin/env python3
"""
Pytest configuration for simulation tests.
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "regression: mark test as a regression test against reference data"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add performance marker to performance tests
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)


@pytest.fixture(scope="session")
def simulation_timeout():
    """Default timeout for simulation runs."""
    return 120  # 2 minutes