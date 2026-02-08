"""Configuration validation for Wikipedia Health Analysis System.

This module provides validation functions to ensure configuration values
are valid and within acceptable ranges.
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from wikipedia_health.config.config import Config


logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a configuration validation error."""
    
    field_path: str
    message: str
    severity: str = 'error'  # 'error' or 'warning'
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.field_path}: {self.message}"


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    
    def add_error(self, field_path: str, message: str) -> None:
        """Add validation error."""
        self.errors.append(ValidationError(field_path, message, 'error'))
        self.is_valid = False
    
    def add_warning(self, field_path: str, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(ValidationError(field_path, message, 'warning'))
    
    def get_summary(self) -> str:
        """Get validation summary."""
        lines = []
        if self.is_valid:
            lines.append("Configuration validation passed")
        else:
            lines.append("Configuration validation failed")
        
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  - {error}")
        
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        
        return "\n".join(lines)


def validate_api_config(config: Config, result: ValidationResult) -> None:
    """Validate API configuration.
    
    Args:
        config: Configuration object
        result: ValidationResult to populate
    """
    api = config.api
    
    # Validate endpoints
    if not api.pageviews_endpoint:
        result.add_error('api.pageviews_endpoint', 'Pageviews endpoint cannot be empty')
    elif not api.pageviews_endpoint.startswith('http'):
        result.add_error('api.pageviews_endpoint', 'Pageviews endpoint must be a valid URL')
    
    if not api.editors_endpoint:
        result.add_error('api.editors_endpoint', 'Editors endpoint cannot be empty')
    elif not api.editors_endpoint.startswith('http'):
        result.add_error('api.editors_endpoint', 'Editors endpoint must be a valid URL')
    
    if not api.edits_endpoint:
        result.add_error('api.edits_endpoint', 'Edits endpoint cannot be empty')
    elif not api.edits_endpoint.startswith('http'):
        result.add_error('api.edits_endpoint', 'Edits endpoint must be a valid URL')
    
    # Validate timeout
    if api.timeout <= 0:
        result.add_error('api.timeout', 'Timeout must be positive')
    elif api.timeout < 10:
        result.add_warning('api.timeout', 'Timeout is very short (< 10 seconds), may cause failures')
    elif api.timeout > 300:
        result.add_warning('api.timeout', 'Timeout is very long (> 300 seconds)')
    
    # Validate retries
    if api.max_retries < 0:
        result.add_error('api.max_retries', 'Max retries cannot be negative')
    elif api.max_retries > 10:
        result.add_warning('api.max_retries', 'Max retries is very high (> 10)')
    
    # Validate backoff factor
    if api.backoff_factor <= 0:
        result.add_error('api.backoff_factor', 'Backoff factor must be positive')
    elif api.backoff_factor < 1.0:
        result.add_warning('api.backoff_factor', 'Backoff factor < 1.0 may not provide sufficient delay')
    
    # Validate user agent
    if not api.user_agent:
        result.add_error('api.user_agent', 'User agent cannot be empty')


def validate_statistical_config(config: Config, result: ValidationResult) -> None:
    """Validate statistical configuration.
    
    Args:
        config: Configuration object
        result: ValidationResult to populate
    """
    stat = config.statistical
    
    # Validate significance level
    if stat.significance_level <= 0 or stat.significance_level >= 1:
        result.add_error('statistical.significance_level', 'Significance level must be between 0 and 1')
    elif stat.significance_level > 0.10:
        result.add_warning('statistical.significance_level', 'Significance level > 0.10 is unusually high')
    
    # Validate confidence level
    if stat.confidence_level <= 0 or stat.confidence_level >= 1:
        result.add_error('statistical.confidence_level', 'Confidence level must be between 0 and 1')
    elif stat.confidence_level < 0.90:
        result.add_warning('statistical.confidence_level', 'Confidence level < 0.90 is unusually low')
    
    # Validate bootstrap samples
    if stat.bootstrap_samples <= 0:
        result.add_error('statistical.bootstrap_samples', 'Bootstrap samples must be positive')
    elif stat.bootstrap_samples < 1000:
        result.add_warning('statistical.bootstrap_samples', 'Bootstrap samples < 1000 may be insufficient')
    
    # Validate permutation iterations
    if stat.permutation_iterations <= 0:
        result.add_error('statistical.permutation_iterations', 'Permutation iterations must be positive')
    elif stat.permutation_iterations < 1000:
        result.add_warning('statistical.permutation_iterations', 'Permutation iterations < 1000 may be insufficient')
    
    # Validate numerical precision
    if stat.numerical_precision <= 0:
        result.add_error('statistical.numerical_precision', 'Numerical precision must be positive')
    
    # Validate outlier threshold
    if stat.outlier_threshold <= 0:
        result.add_error('statistical.outlier_threshold', 'Outlier threshold must be positive')
    elif stat.outlier_threshold < 2.0:
        result.add_warning('statistical.outlier_threshold', 'Outlier threshold < 2.0 may flag too many points')
    
    # Validate minimum data points
    if stat.min_data_points_trend <= 0:
        result.add_error('statistical.min_data_points_trend', 'Min data points for trend must be positive')
    elif stat.min_data_points_trend < 30:
        result.add_warning('statistical.min_data_points_trend', 'Min data points < 30 may be insufficient for trends')
    
    if stat.min_data_points_causal <= 0:
        result.add_error('statistical.min_data_points_causal', 'Min data points for causal must be positive')
    elif stat.min_data_points_causal < 30:
        result.add_warning('statistical.min_data_points_causal', 'Min data points < 30 may be insufficient for causal inference')


def validate_time_series_config(config: Config, result: ValidationResult) -> None:
    """Validate time series configuration.
    
    Args:
        config: Configuration object
        result: ValidationResult to populate
    """
    ts = config.time_series
    
    # Validate seasonal period
    if ts.seasonal_period <= 0:
        result.add_error('time_series.seasonal_period', 'Seasonal period must be positive')
    elif ts.seasonal_period < 2:
        result.add_warning('time_series.seasonal_period', 'Seasonal period < 2 is unusual')
    
    # Validate changepoint min size
    if ts.changepoint_min_size <= 0:
        result.add_error('time_series.changepoint_min_size', 'Changepoint min size must be positive')
    elif ts.changepoint_min_size < 10:
        result.add_warning('time_series.changepoint_min_size', 'Changepoint min size < 10 may detect spurious breaks')
    
    # Validate forecast methods
    valid_methods = ['arima', 'prophet', 'exponential_smoothing']
    if not ts.forecast_methods:
        result.add_error('time_series.forecast_methods', 'At least one forecast method must be specified')
    else:
        for method in ts.forecast_methods:
            if method not in valid_methods:
                result.add_error('time_series.forecast_methods', f'Invalid method: {method}. Must be one of {valid_methods}')
    
    # Validate prediction intervals
    if not ts.prediction_intervals:
        result.add_error('time_series.prediction_intervals', 'At least one prediction interval must be specified')
    else:
        for interval in ts.prediction_intervals:
            if interval <= 0 or interval >= 1:
                result.add_error('time_series.prediction_intervals', f'Prediction interval {interval} must be between 0 and 1')
    
    # Validate holdout percentage
    if ts.holdout_percentage <= 0 or ts.holdout_percentage >= 1:
        result.add_error('time_series.holdout_percentage', 'Holdout percentage must be between 0 and 1')
    elif ts.holdout_percentage < 0.05:
        result.add_warning('time_series.holdout_percentage', 'Holdout percentage < 0.05 may be too small')
    elif ts.holdout_percentage > 0.30:
        result.add_warning('time_series.holdout_percentage', 'Holdout percentage > 0.30 may leave insufficient training data')


def validate_causal_config(config: Config, result: ValidationResult) -> None:
    """Validate causal inference configuration.
    
    Args:
        config: Configuration object
        result: ValidationResult to populate
    """
    causal = config.causal
    
    # Validate period lengths
    if causal.pre_period_length <= 0:
        result.add_error('causal.pre_period_length', 'Pre-period length must be positive')
    elif causal.pre_period_length < 30:
        result.add_warning('causal.pre_period_length', 'Pre-period length < 30 may be insufficient for baseline')
    
    if causal.post_period_length <= 0:
        result.add_error('causal.post_period_length', 'Post-period length must be positive')
    
    if causal.baseline_window <= 0:
        result.add_error('causal.baseline_window', 'Baseline window must be positive')
    elif causal.baseline_window < 30:
        result.add_warning('causal.baseline_window', 'Baseline window < 30 may be insufficient')
    
    if causal.event_post_window <= 0:
        result.add_error('causal.event_post_window', 'Event post window must be positive')
    
    if causal.event_max_window <= 0:
        result.add_error('causal.event_max_window', 'Event max window must be positive')
    elif causal.event_max_window < causal.event_post_window:
        result.add_error('causal.event_max_window', 'Event max window must be >= event post window')
    
    # Validate placebo iterations
    if causal.placebo_iterations <= 0:
        result.add_error('causal.placebo_iterations', 'Placebo iterations must be positive')
    elif causal.placebo_iterations < 50:
        result.add_warning('causal.placebo_iterations', 'Placebo iterations < 50 may be insufficient')


def validate_validation_config(config: Config, result: ValidationResult) -> None:
    """Validate data validation configuration.
    
    Args:
        config: Configuration object
        result: ValidationResult to populate
    """
    val = config.validation
    
    # Validate max missing percentage
    if val.max_missing_percentage < 0 or val.max_missing_percentage > 1:
        result.add_error('validation.max_missing_percentage', 'Max missing percentage must be between 0 and 1')
    elif val.max_missing_percentage > 0.20:
        result.add_warning('validation.max_missing_percentage', 'Max missing percentage > 0.20 is very permissive')
    
    # Validate max gap days
    if val.max_gap_days < 0:
        result.add_error('validation.max_gap_days', 'Max gap days cannot be negative')
    elif val.max_gap_days > 7:
        result.add_warning('validation.max_gap_days', 'Max gap days > 7 may allow large data gaps')
    
    # Validate staleness threshold
    if val.staleness_threshold_hours <= 0:
        result.add_error('validation.staleness_threshold_hours', 'Staleness threshold must be positive')
    
    # Validate data sources
    valid_sources = ['pageviews', 'editors', 'edits']
    if not val.data_sources:
        result.add_error('validation.data_sources', 'At least one data source must be specified')
    else:
        for source in val.data_sources:
            if source not in valid_sources:
                result.add_error('validation.data_sources', f'Invalid source: {source}. Must be one of {valid_sources}')
    
    # Validate platforms
    valid_platforms = ['desktop', 'mobile-web', 'mobile-app', 'all']
    if not val.platforms:
        result.add_error('validation.platforms', 'At least one platform must be specified')
    else:
        for platform in val.platforms:
            if platform not in valid_platforms:
                result.add_error('validation.platforms', f'Invalid platform: {platform}. Must be one of {valid_platforms}')


def validate_config(config: Config) -> ValidationResult:
    """Validate complete configuration.
    
    Args:
        config: Configuration object to validate
    
    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult(is_valid=True)
    
    # Validate each configuration section
    validate_api_config(config, result)
    validate_statistical_config(config, result)
    validate_time_series_config(config, result)
    validate_causal_config(config, result)
    validate_validation_config(config, result)
    
    # Log results
    if not result.is_valid:
        logger.error("Configuration validation failed")
        for error in result.errors:
            logger.error(str(error))
    
    if result.warnings:
        for warning in result.warnings:
            logger.warning(str(warning))
    
    return result
