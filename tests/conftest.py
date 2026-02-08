"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
from wikipedia_health.config import Config


@pytest.fixture
def default_config():
    """Provide default configuration for tests."""
    return Config()


@pytest.fixture
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "data"
