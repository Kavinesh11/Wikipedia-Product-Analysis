"""Time series analysis module.

This module provides time series decomposition, changepoint detection,
and forecasting functionality.
"""

from wikipedia_health.time_series.decomposer import TimeSeriesDecomposer
from wikipedia_health.time_series.changepoint_detector import ChangepointDetector
from wikipedia_health.time_series.forecaster import (
    Forecaster,
    ARIMAModel,
    ProphetModel,
    ExponentialSmoothingModel,
    CrossValidationResult
)

__all__ = [
    'TimeSeriesDecomposer',
    'ChangepointDetector',
    'Forecaster',
    'ARIMAModel',
    'ProphetModel',
    'ExponentialSmoothingModel',
    'CrossValidationResult'
]
