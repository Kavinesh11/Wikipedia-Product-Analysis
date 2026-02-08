"""Configuration management for Wikipedia Health Analysis System."""

import yaml
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class APIConfig:
    """Configuration for Wikimedia API endpoints."""
    
    pageviews_endpoint: str = "https://wikimedia.org/api/rest_v1/metrics/pageviews"
    editors_endpoint: str = "https://wikimedia.org/api/rest_v1/metrics/editors"
    edits_endpoint: str = "https://wikimedia.org/api/rest_v1/metrics/edits"
    timeout: int = 30
    max_retries: int = 5
    backoff_factor: float = 2.0
    user_agent: str = "WikipediaHealthAnalysis/0.1.0"


@dataclass
class StatisticalConfig:
    """Configuration for statistical analysis parameters."""
    
    significance_level: float = 0.05
    confidence_level: float = 0.95
    bootstrap_samples: int = 10000
    permutation_iterations: int = 10000
    numerical_precision: float = 1e-10
    outlier_threshold: float = 3.0  # Standard deviations
    min_data_points_trend: int = 90
    min_data_points_causal: int = 30


@dataclass
class TimeSeriesConfig:
    """Configuration for time series analysis."""
    
    seasonal_period: int = 7  # Weekly seasonality
    changepoint_min_size: int = 30
    forecast_methods: List[str] = field(default_factory=lambda: ['arima', 'prophet', 'exponential_smoothing'])
    prediction_intervals: List[float] = field(default_factory=lambda: [0.50, 0.80, 0.95])
    holdout_percentage: float = 0.10


@dataclass
class CausalConfig:
    """Configuration for causal inference."""
    
    pre_period_length: int = 90
    post_period_length: int = 90
    baseline_window: int = 90
    event_post_window: int = 30
    event_max_window: int = 180
    placebo_iterations: int = 100


@dataclass
class ValidationConfig:
    """Configuration for data validation and quality checks."""
    
    max_missing_percentage: float = 0.10
    max_gap_days: int = 3
    staleness_threshold_hours: int = 24
    data_sources: List[str] = field(default_factory=lambda: ['pageviews', 'editors', 'edits'])
    platforms: List[str] = field(default_factory=lambda: ['desktop', 'mobile-web', 'mobile-app'])


@dataclass
class Config:
    """Main configuration class."""
    
    api: APIConfig = field(default_factory=APIConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    time_series: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create Config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config instance
        """
        return cls(
            api=APIConfig(**config_dict.get('api', {})),
            statistical=StatisticalConfig(**config_dict.get('statistical', {})),
            time_series=TimeSeriesConfig(**config_dict.get('time_series', {})),
            causal=CausalConfig(**config_dict.get('causal', {})),
            validation=ValidationConfig(**config_dict.get('validation', {}))
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Config':
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: Path) -> 'Config':
        """Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            Config instance
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict:
        """Convert Config to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'api': self.api.__dict__,
            'statistical': self.statistical.__dict__,
            'time_series': self.time_series.__dict__,
            'causal': self.causal.__dict__,
            'validation': self.validation.__dict__
        }
    
    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration
        """
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, json_path: Path, indent: int = 2) -> None:
        """Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON configuration
            indent: JSON indentation level
        """
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from file or use defaults.
    
    Supports both YAML and JSON formats. Format is determined by file extension.
    
    Args:
        config_path: Optional path to configuration file (.yaml, .yml, or .json)
        
    Returns:
        Config instance
        
    Raises:
        ValueError: If file format is not supported
    """
    if config_path and config_path.exists():
        suffix = config_path.suffix.lower()
        if suffix in ['.yaml', '.yml']:
            return Config.from_yaml(config_path)
        elif suffix == '.json':
            return Config.from_json(config_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {suffix}. Use .yaml, .yml, or .json")
    return Config()
