"""Unit tests for time series analysis components.

This module contains unit tests for time series decomposition, changepoint detection,
and forecasting with specific examples and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from wikipedia_health.time_series import (
    TimeSeriesDecomposer,
    ChangepointDetector,
    Forecaster
)


class TestTimeSeriesDecomposer:
    """Unit tests for TimeSeriesDecomposer class."""
    
    def test_decompose_stl_with_synthetic_seasonal_data(self):
        """Test STL decomposition with synthetic data containing known seasonality."""
        # Create synthetic data with trend and seasonality
        n = 365
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        
        # Trend: linear increase
        trend = np.linspace(100, 200, n)
        
        # Seasonality: weekly pattern
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
        
        # Noise
        np.random.seed(42)
        noise = np.random.normal(0, 2, n)
        
        # Combine components
        values = trend + seasonal + noise
        series = pd.Series(values, index=dates)
        
        # Decompose
        decomposer = TimeSeriesDecomposer()
        result = decomposer.decompose_stl(series, period=7)
        
        # Verify components exist
        assert result.trend is not None
        assert result.seasonal is not None
        assert result.residual is not None
        
        # Verify reconstruction
        reconstructed = result.reconstruct()
        assert np.allclose(series.values, reconstructed.values, rtol=1e-5, atol=1e-6)
        
        # Verify trend is increasing (should capture the linear trend)
        assert result.trend.iloc[-1] > result.trend.iloc[0]
        
        # Verify seasonal component has expected period
        # Check that seasonal pattern repeats approximately every 7 days
        seasonal_values = result.seasonal.values
        for i in range(0, len(seasonal_values) - 7, 7):
            # Seasonal values should be similar 7 days apart
            assert abs(seasonal_values[i] - seasonal_values[i + 7]) < 5
    
    def test_decompose_x13_fallback(self):
        """Test X-13 decomposition fallback for short series."""
        # Create short series
        n = 30
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        values = np.linspace(100, 120, n) + np.random.normal(0, 2, n)
        series = pd.Series(values, index=dates)
        
        # Decompose
        decomposer = TimeSeriesDecomposer()
        result = decomposer.decompose_x13(series)
        
        # Verify components exist
        assert result.trend is not None
        assert result.seasonal is not None
        assert result.residual is not None
        assert len(result.trend) == len(series)
    
    def test_extract_trend_hp_filter(self):
        """Test trend extraction using Hodrick-Prescott filter."""
        # Create data with trend and noise
        n = 100
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        trend = np.linspace(100, 200, n)
        noise = np.random.normal(0, 5, n)
        values = trend + noise
        series = pd.Series(values, index=dates)
        
        # Extract trend
        decomposer = TimeSeriesDecomposer()
        extracted_trend = decomposer.extract_trend(series, method='hp_filter')
        
        # Verify trend is smoother than original series
        assert len(extracted_trend) == len(series)
        assert extracted_trend.std() < series.std()
        
        # Verify trend is increasing
        assert extracted_trend.iloc[-1] > extracted_trend.iloc[0]
    
    def test_extract_seasonality(self):
        """Test seasonality extraction."""
        # Create data with strong seasonality
        n = 100
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 7)
        trend = 100
        values = trend + seasonal
        series = pd.Series(values, index=dates)
        
        # Extract seasonality
        decomposer = TimeSeriesDecomposer()
        extracted_seasonal = decomposer.extract_seasonality(series, period=7)
        
        # Verify seasonality has expected properties
        assert len(extracted_seasonal) == len(series)
        
        # Seasonal component should oscillate around zero
        assert abs(extracted_seasonal.mean()) < 5


class TestChangepointDetector:
    """Unit tests for ChangepointDetector class."""
    
    def test_detect_pelt_with_known_break(self):
        """Test PELT changepoint detection with a known structural break."""
        # Create data with a clear changepoint at index 50
        n = 100
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        
        # First segment: mean = 100
        np.random.seed(42)
        segment1 = np.random.normal(100, 5, 50)
        
        # Second segment: mean = 150 (clear jump)
        segment2 = np.random.normal(150, 5, 50)
        
        values = np.concatenate([segment1, segment2])
        series = pd.Series(values, index=dates)
        
        # Detect changepoints with lower penalty to be more sensitive
        detector = ChangepointDetector()
        changepoints = detector.detect_pelt(series, penalty=10, min_size=10)
        
        # Should detect at least one changepoint
        assert len(changepoints) > 0
        
        # The detected changepoint should be near index 50
        detected_indices = [cp.index for cp in changepoints]
        assert any(abs(idx - 50) < 15 for idx in detected_indices)
    
    def test_detect_binary_segmentation(self):
        """Test Binary Segmentation changepoint detection."""
        # Create data with multiple changepoints
        n = 150
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        
        segment1 = np.random.normal(100, 5, 50)
        segment2 = np.random.normal(150, 5, 50)
        segment3 = np.random.normal(120, 5, 50)
        
        values = np.concatenate([segment1, segment2, segment3])
        series = pd.Series(values, index=dates)
        
        # Detect changepoints
        detector = ChangepointDetector()
        changepoints = detector.detect_binary_segmentation(series, n_changepoints=3)
        
        # Should detect changepoints
        assert len(changepoints) > 0
        assert len(changepoints) <= 3
    
    def test_test_significance(self):
        """Test statistical significance testing of changepoints."""
        # Create data with a significant changepoint
        n = 100
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        
        segment1 = np.random.normal(100, 5, 50)
        segment2 = np.random.normal(150, 5, 50)
        
        values = np.concatenate([segment1, segment2])
        series = pd.Series(values, index=dates)
        
        # Detect changepoint
        detector = ChangepointDetector()
        changepoints = detector.detect_pelt(series, min_size=10)
        
        if len(changepoints) > 0:
            # Test significance
            is_significant, p_value = detector.test_significance(
                series, changepoints[0], alpha=0.05
            )
            
            # P-value should be valid
            assert 0 <= p_value <= 1
            
            # For this clear break, should be significant
            assert is_significant or p_value < 0.1  # Allow some tolerance


class TestForecaster:
    """Unit tests for Forecaster class."""
    
    def test_fit_arima_and_forecast(self):
        """Test ARIMA model fitting and forecasting."""
        # Create simple time series
        n = 100
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        np.random.seed(42)
        values = np.linspace(100, 150, n) + np.random.normal(0, 2, n)
        series = pd.Series(values, index=dates)
        
        # Fit ARIMA model
        forecaster = Forecaster()
        model = forecaster.fit_arima(series)
        
        # Verify model was fitted
        assert model is not None
        assert model.order is not None
        assert len(model.order) == 3
        
        # Generate forecast
        horizon = 10
        forecast = forecaster.forecast(model, horizon=horizon)
        
        # Verify forecast structure
        assert len(forecast.point_forecast) == horizon
        assert len(forecast.lower_bound) == horizon
        assert len(forecast.upper_bound) == horizon
        
        # Verify prediction intervals are valid (convert to numpy for comparison)
        assert all(forecast.lower_bound.values <= forecast.point_forecast.values)
        assert all(forecast.point_forecast.values <= forecast.upper_bound.values)
    
    def test_fit_prophet_and_forecast(self):
        """Test Prophet model fitting and forecasting."""
        # Create time series with seasonality
        n = 100
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        trend = np.linspace(100, 150, n)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
        np.random.seed(42)
        values = trend + seasonal + np.random.normal(0, 2, n)
        series = pd.Series(values, index=dates)
        
        # Fit Prophet model
        forecaster = Forecaster()
        model = forecaster.fit_prophet(series)
        
        # Verify model was fitted
        assert model is not None
        assert model.model is not None
        
        # Generate forecast
        horizon = 10
        forecast = forecaster.forecast(model, horizon=horizon)
        
        # Verify forecast structure
        assert len(forecast.point_forecast) == horizon
        assert len(forecast.lower_bound) == horizon
        assert len(forecast.upper_bound) == horizon
        
        # Verify prediction intervals are valid
        assert all(forecast.lower_bound.values <= forecast.point_forecast.values)
        assert all(forecast.point_forecast.values <= forecast.upper_bound.values)
    
    def test_fit_exponential_smoothing_and_forecast(self):
        """Test Exponential Smoothing model fitting and forecasting."""
        # Create time series with trend and seasonality
        n = 70
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        trend = np.linspace(100, 150, n)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
        np.random.seed(42)
        values = trend + seasonal + np.random.normal(0, 2, n)
        series = pd.Series(values, index=dates)
        
        # Fit Exponential Smoothing model
        forecaster = Forecaster()
        model = forecaster.fit_exponential_smoothing(series, seasonal_periods=7)
        
        # Verify model was fitted
        assert model is not None
        assert model.model is not None
        
        # Generate forecast
        horizon = 7
        forecast = forecaster.forecast(model, horizon=horizon)
        
        # Verify forecast structure
        assert len(forecast.point_forecast) == horizon
        assert len(forecast.lower_bound) == horizon
        assert len(forecast.upper_bound) == horizon
        
        # Verify prediction intervals are valid
        assert all(forecast.lower_bound.values <= forecast.point_forecast.values)
        assert all(forecast.point_forecast.values <= forecast.upper_bound.values)
    
    def test_cross_validate_arima(self):
        """Test cross-validation with ARIMA model."""
        # Create time series
        n = 100
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        values = np.linspace(100, 150, n) + np.random.normal(0, 5, n)
        series = pd.Series(values, index=dates)
        
        # Perform cross-validation
        forecaster = Forecaster()
        cv_result = forecaster.cross_validate(series, model_type='arima', n_splits=3)
        
        # Verify results
        assert cv_result is not None
        assert cv_result.mean_error > 0
        assert cv_result.std_error >= 0
        assert len(cv_result.errors) == 3
        assert cv_result.model_type == 'arima'
    
    def test_forecast_with_holdout_validation(self):
        """Test forecasting with holdout validation."""
        # Create time series
        n = 100
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        values = np.linspace(100, 150, n) + np.random.normal(0, 3, n)
        series = pd.Series(values, index=dates)
        
        # Split into train and test
        train_size = 80
        train = series.iloc[:train_size]
        test = series.iloc[train_size:]
        
        # Fit model on training data
        forecaster = Forecaster()
        model = forecaster.fit_arima(train)
        
        # Forecast
        horizon = len(test)
        forecast = forecaster.forecast(model, horizon=horizon)
        
        # Calculate error
        rmse = np.sqrt(np.mean((test.values - forecast.point_forecast.values) ** 2))
        
        # Error should be reasonable (not too large)
        assert rmse < 50  # Reasonable threshold for this synthetic data
