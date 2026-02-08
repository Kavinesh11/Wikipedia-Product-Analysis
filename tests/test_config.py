"""Tests for configuration management."""

import pytest
from pathlib import Path
from wikipedia_health.config import Config, load_config


def test_default_config_creation():
    """Test that default configuration can be created."""
    config = Config()
    
    assert config.api.pageviews_endpoint == "https://wikimedia.org/api/rest_v1/metrics/pageviews"
    assert config.statistical.significance_level == 0.05
    assert config.statistical.confidence_level == 0.95
    assert config.time_series.seasonal_period == 7
    assert config.causal.pre_period_length == 90
    assert config.validation.max_missing_percentage == 0.10


def test_config_from_yaml():
    """Test loading configuration from YAML file."""
    config_path = Path("config.yaml")
    if config_path.exists():
        config = Config.from_yaml(config_path)
        
        assert config.api.timeout == 30
        assert config.api.max_retries == 5
        assert config.statistical.bootstrap_samples == 10000
        assert len(config.time_series.forecast_methods) == 3
        assert len(config.validation.platforms) == 3


def test_load_config_with_defaults():
    """Test load_config function with defaults."""
    config = load_config()
    
    assert isinstance(config, Config)
    assert config.api.backoff_factor == 2.0
    assert config.causal.baseline_window == 90


def test_config_to_dict():
    """Test converting config to dictionary."""
    config = Config()
    config_dict = config.to_dict()
    
    assert 'api' in config_dict
    assert 'statistical' in config_dict
    assert 'time_series' in config_dict
    assert 'causal' in config_dict
    assert 'validation' in config_dict
    assert config_dict['statistical']['significance_level'] == 0.05
