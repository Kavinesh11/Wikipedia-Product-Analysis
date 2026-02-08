"""Tests for configuration validation."""

import pytest
from pathlib import Path
import tempfile
import yaml
import json

from wikipedia_health.config import Config, load_config, validate_config, ValidationResult
from wikipedia_health.config.config import (
    APIConfig,
    StatisticalConfig,
    TimeSeriesConfig,
    CausalConfig,
    ValidationConfig
)


class TestConfigLoading:
    """Tests for configuration loading."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = Config()
        assert config.api.timeout == 30
        assert config.statistical.significance_level == 0.05
        assert config.time_series.seasonal_period == 7
    
    def test_load_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'api': {'timeout': 60},
                'statistical': {'significance_level': 0.01}
            }, f)
            temp_path = Path(f.name)
        
        try:
            config = load_config(temp_path)
            assert config.api.timeout == 60
            assert config.statistical.significance_level == 0.01
        finally:
            temp_path.unlink()
    
    def test_load_config_from_json(self):
        """Test loading configuration from JSON file."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'api': {'timeout': 45},
                'statistical': {'bootstrap_samples': 5000}
            }, f)
            temp_path = Path(f.name)
        
        try:
            config = load_config(temp_path)
            assert config.api.timeout == 45
            assert config.statistical.bootstrap_samples == 5000
        finally:
            temp_path.unlink()
    
    def test_load_config_unsupported_format(self):
        """Test loading configuration from unsupported format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Unsupported configuration file format"):
                load_config(temp_path)
        finally:
            temp_path.unlink()
    
    def test_load_config_nonexistent_file(self):
        """Test loading configuration from nonexistent file."""
        config = load_config(Path('nonexistent.yaml'))
        # Should return default config
        assert config.api.timeout == 30


class TestConfigSaving:
    """Tests for configuration saving."""
    
    def test_save_config_to_yaml(self):
        """Test saving configuration to YAML file."""
        config = Config()
        config.api.timeout = 75
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            config.to_yaml(temp_path)
            
            # Load and verify
            loaded_config = Config.from_yaml(temp_path)
            assert loaded_config.api.timeout == 75
        finally:
            temp_path.unlink()
    
    def test_save_config_to_json(self):
        """Test saving configuration to JSON file."""
        config = Config()
        config.statistical.significance_level = 0.01
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            config.to_json(temp_path)
            
            # Load and verify
            loaded_config = Config.from_json(temp_path)
            assert loaded_config.statistical.significance_level == 0.01
        finally:
            temp_path.unlink()


class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_validate_default_config(self):
        """Test validating default configuration."""
        config = Config()
        result = validate_config(config)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_invalid_api_timeout(self):
        """Test validation with invalid API timeout."""
        config = Config()
        config.api.timeout = -10
        result = validate_config(config)
        assert not result.is_valid
        assert any('timeout' in str(e).lower() for e in result.errors)
    
    def test_validate_invalid_significance_level(self):
        """Test validation with invalid significance level."""
        config = Config()
        config.statistical.significance_level = 1.5
        result = validate_config(config)
        assert not result.is_valid
        assert any('significance_level' in str(e).lower() for e in result.errors)
    
    def test_validate_invalid_confidence_level(self):
        """Test validation with invalid confidence level."""
        config = Config()
        config.statistical.confidence_level = 0.0
        result = validate_config(config)
        assert not result.is_valid
        assert any('confidence_level' in str(e).lower() for e in result.errors)
    
    def test_validate_invalid_forecast_method(self):
        """Test validation with invalid forecast method."""
        config = Config()
        config.time_series.forecast_methods = ['invalid_method']
        result = validate_config(config)
        assert not result.is_valid
        assert any('forecast_methods' in str(e).lower() for e in result.errors)
    
    def test_validate_invalid_prediction_interval(self):
        """Test validation with invalid prediction interval."""
        config = Config()
        config.time_series.prediction_intervals = [1.5]
        result = validate_config(config)
        assert not result.is_valid
        assert any('prediction_intervals' in str(e).lower() for e in result.errors)
    
    def test_validate_invalid_platform(self):
        """Test validation with invalid platform."""
        config = Config()
        config.validation.platforms = ['invalid_platform']
        result = validate_config(config)
        assert not result.is_valid
        assert any('platforms' in str(e).lower() for e in result.errors)
    
    def test_validate_warnings(self):
        """Test validation with warnings."""
        config = Config()
        config.api.timeout = 5  # Very short timeout
        result = validate_config(config)
        # Should still be valid but have warnings
        assert result.is_valid or len(result.warnings) > 0
    
    def test_validate_empty_user_agent(self):
        """Test validation with empty user agent."""
        config = Config()
        config.api.user_agent = ""
        result = validate_config(config)
        assert not result.is_valid
        assert any('user_agent' in str(e).lower() for e in result.errors)
    
    def test_validate_negative_bootstrap_samples(self):
        """Test validation with negative bootstrap samples."""
        config = Config()
        config.statistical.bootstrap_samples = -100
        result = validate_config(config)
        assert not result.is_valid
        assert any('bootstrap_samples' in str(e).lower() for e in result.errors)
    
    def test_validate_invalid_holdout_percentage(self):
        """Test validation with invalid holdout percentage."""
        config = Config()
        config.time_series.holdout_percentage = 1.5
        result = validate_config(config)
        assert not result.is_valid
        assert any('holdout_percentage' in str(e).lower() for e in result.errors)
    
    def test_validate_event_window_consistency(self):
        """Test validation of event window consistency."""
        config = Config()
        config.causal.event_max_window = 10
        config.causal.event_post_window = 30  # Max < post
        result = validate_config(config)
        assert not result.is_valid
        assert any('event_max_window' in str(e).lower() for e in result.errors)


class TestValidationResult:
    """Tests for ValidationResult class."""
    
    def test_validation_result_add_error(self):
        """Test adding error to validation result."""
        result = ValidationResult(is_valid=True)
        result.add_error('test.field', 'Test error message')
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].field_path == 'test.field'
    
    def test_validation_result_add_warning(self):
        """Test adding warning to validation result."""
        result = ValidationResult(is_valid=True)
        result.add_warning('test.field', 'Test warning message')
        assert result.is_valid  # Warnings don't invalidate
        assert len(result.warnings) == 1
    
    def test_validation_result_summary(self):
        """Test validation result summary."""
        result = ValidationResult(is_valid=True)
        result.add_error('field1', 'Error 1')
        result.add_warning('field2', 'Warning 1')
        
        summary = result.get_summary()
        assert 'failed' in summary.lower()
        assert 'Error 1' in summary
        assert 'Warning 1' in summary


class TestConfigFromDict:
    """Tests for creating config from dictionary."""
    
    def test_config_from_dict_partial(self):
        """Test creating config from partial dictionary."""
        config_dict = {
            'api': {'timeout': 50},
            'statistical': {'significance_level': 0.01}
        }
        config = Config.from_dict(config_dict)
        assert config.api.timeout == 50
        assert config.statistical.significance_level == 0.01
        # Other values should be defaults
        assert config.time_series.seasonal_period == 7
    
    def test_config_from_dict_empty(self):
        """Test creating config from empty dictionary."""
        config = Config.from_dict({})
        # Should use all defaults
        assert config.api.timeout == 30
        assert config.statistical.significance_level == 0.05


class TestConfigToDict:
    """Tests for converting config to dictionary."""
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config.api.timeout = 100
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['api']['timeout'] == 100
        assert 'statistical' in config_dict
        assert 'time_series' in config_dict
        assert 'causal' in config_dict
        assert 'validation' in config_dict


class TestAPIConfig:
    """Tests for API configuration."""
    
    def test_api_config_defaults(self):
        """Test API config default values."""
        api_config = APIConfig()
        assert api_config.timeout == 30
        assert api_config.max_retries == 5
        assert api_config.backoff_factor == 2.0
        assert 'WikipediaHealthAnalysis' in api_config.user_agent


class TestStatisticalConfig:
    """Tests for statistical configuration."""
    
    def test_statistical_config_defaults(self):
        """Test statistical config default values."""
        stat_config = StatisticalConfig()
        assert stat_config.significance_level == 0.05
        assert stat_config.confidence_level == 0.95
        assert stat_config.bootstrap_samples == 10000


class TestTimeSeriesConfig:
    """Tests for time series configuration."""
    
    def test_time_series_config_defaults(self):
        """Test time series config default values."""
        ts_config = TimeSeriesConfig()
        assert ts_config.seasonal_period == 7
        assert 'arima' in ts_config.forecast_methods
        assert 0.95 in ts_config.prediction_intervals


class TestCausalConfig:
    """Tests for causal configuration."""
    
    def test_causal_config_defaults(self):
        """Test causal config default values."""
        causal_config = CausalConfig()
        assert causal_config.pre_period_length == 90
        assert causal_config.post_period_length == 90
        assert causal_config.placebo_iterations == 100


class TestValidationConfig:
    """Tests for validation configuration."""
    
    def test_validation_config_defaults(self):
        """Test validation config default values."""
        val_config = ValidationConfig()
        assert val_config.max_missing_percentage == 0.10
        assert val_config.max_gap_days == 3
        assert 'pageviews' in val_config.data_sources
        assert 'desktop' in val_config.platforms
