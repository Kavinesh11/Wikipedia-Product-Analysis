"""Unit Tests for Time Series Forecasting

Tests specific examples and edge cases for the TimeSeriesForecaster component.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.analytics.forecaster import TimeSeriesForecaster
from src.storage.dto import ForecastResult, SeasonalityPattern, SpikeEvent


class TestTimeSeriesForecaster:
    """Unit tests for TimeSeriesForecaster class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.forecaster = TimeSeriesForecaster()
        
        # Create synthetic time series data (100 days)
        n_days = 100
        start_date = datetime(2024, 1, 1)
        self.dates = [start_date + timedelta(days=i) for i in range(n_days)]
        self.views = [1000 + i * 10 + np.random.randint(-50, 50) for i in range(n_days)]
        
        self.test_data = pd.DataFrame({
            'date': self.dates,
            'views': self.views
        })
    
    def test_forecaster_initialization(self):
        """Test forecaster initializes correctly"""
        forecaster = TimeSeriesForecaster()
        assert forecaster.model_type == "prophet"
        assert len(forecaster._models) == 0
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error"""
        with pytest.raises(ValueError) as exc_info:
            TimeSeriesForecaster(model_type="invalid")
        assert "unsupported" in str(exc_info.value).lower()
    
    def test_train_with_sufficient_data(self):
        """Test training with sufficient data (>= 90 days)"""
        model = self.forecaster.train(self.test_data, "Test_Article")
        assert model is not None
        assert "Test_Article" in self.forecaster._models
    
    def test_train_with_insufficient_data(self):
        """Test training with insufficient data (< 90 days)"""
        small_data = self.test_data.head(50)
        
        with pytest.raises(ValueError) as exc_info:
            self.forecaster.train(small_data, "Test_Article")
        
        error_msg = str(exc_info.value).lower()
        assert "insufficient" in error_msg or "90" in error_msg
    
    def test_train_with_invalid_dataframe(self):
        """Test training with invalid DataFrame"""
        # Missing required columns
        invalid_data = pd.DataFrame({'wrong_col': [1, 2, 3]})
        
        with pytest.raises(ValueError) as exc_info:
            self.forecaster.train(invalid_data, "Test_Article")
        
        assert "date" in str(exc_info.value).lower() or "views" in str(exc_info.value).lower()
    
    def test_train_with_non_dataframe(self):
        """Test training with non-DataFrame input"""
        with pytest.raises(ValueError) as exc_info:
            self.forecaster.train([1, 2, 3], "Test_Article")
        
        assert "dataframe" in str(exc_info.value).lower()
    
    def test_predict_returns_correct_periods(self):
        """Test that predict returns correct number of periods"""
        model = self.forecaster.train(self.test_data, "Test_Article")
        
        for periods in [7, 14, 30, 60]:
            result = self.forecaster.predict(model, periods=periods, article="Test_Article")
            assert len(result.predictions) == periods
    
    def test_predict_has_required_columns(self):
        """Test that predictions have required columns"""
        model = self.forecaster.train(self.test_data, "Test_Article")
        result = self.forecaster.predict(model, periods=30, article="Test_Article")
        
        required_cols = ['date', 'yhat', 'yhat_lower', 'yhat_upper']
        for col in required_cols:
            assert col in result.predictions.columns
    
    def test_predict_confidence_bounds_valid(self):
        """Test that confidence bounds are valid (lower <= yhat <= upper)"""
        model = self.forecaster.train(self.test_data, "Test_Article")
        result = self.forecaster.predict(model, periods=30, article="Test_Article")
        
        for idx, row in result.predictions.iterrows():
            assert row['yhat_lower'] <= row['yhat']
            assert row['yhat'] <= row['yhat_upper']
    
    def test_predict_confidence_score_range(self):
        """Test that confidence score is in valid range [0, 1]"""
        model = self.forecaster.train(self.test_data, "Test_Article")
        result = self.forecaster.predict(model, periods=30, article="Test_Article")
        
        assert 0 <= result.confidence <= 1
    
    def test_calculate_growth_rate_positive_growth(self):
        """Test growth rate calculation with positive growth"""
        # Create data with clear positive growth
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]
        views = [1000 + i * 20 for i in range(60)]  # Linear growth
        data = pd.DataFrame({'date': dates, 'views': views})
        
        growth_rate = self.forecaster.calculate_growth_rate(data, period_days=30)
        
        # Should be positive
        assert growth_rate > 0
    
    def test_calculate_growth_rate_negative_growth(self):
        """Test growth rate calculation with negative growth"""
        # Create data with clear negative growth
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]
        views = [2000 - i * 20 for i in range(60)]  # Linear decline
        data = pd.DataFrame({'date': dates, 'views': views})
        
        growth_rate = self.forecaster.calculate_growth_rate(data, period_days=30)
        
        # Should be negative
        assert growth_rate < 0
    
    def test_calculate_growth_rate_flat_trend(self):
        """Test growth rate calculation with flat trend"""
        # Create data with no growth
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]
        views = [1000 for _ in range(60)]  # Flat
        data = pd.DataFrame({'date': dates, 'views': views})
        
        growth_rate = self.forecaster.calculate_growth_rate(data, period_days=30)
        
        # Should be close to zero
        assert abs(growth_rate) < 1.0
    
    def test_calculate_growth_rate_zero_start(self):
        """Test growth rate calculation when starting views are zero"""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]
        views = [0] + [1000 for _ in range(59)]
        data = pd.DataFrame({'date': dates, 'views': views})
        
        # Should not raise error
        growth_rate = self.forecaster.calculate_growth_rate(data, period_days=30)
        
        # Should return valid number
        assert isinstance(growth_rate, (int, float))
        assert not np.isnan(growth_rate)
        assert not np.isinf(growth_rate)
    
    def test_detect_hype_events_with_spike(self):
        """Test hype event detection with clear spike"""
        # Create data with a spike
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        views = [1000 for _ in range(100)]
        
        # Add spike in the middle (> 2 std dev above mean)
        for i in range(50, 56):
            views[i] = 5000
        
        data = pd.DataFrame({'date': dates, 'views': views})
        
        spike_events = self.forecaster.detect_hype_events(data, "Test_Article")
        
        # Should detect at least one spike
        assert len(spike_events) > 0
        
        # Check spike properties
        for event in spike_events:
            assert isinstance(event, SpikeEvent)
            assert event.magnitude > 2.0  # More than 2 std devs
            assert event.spike_type in ['sustained', 'temporary']
    
    def test_detect_hype_events_no_spike(self):
        """Test hype event detection with no spikes"""
        # Create data with no spikes (low variance)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        views = [1000 + np.random.randint(-10, 10) for _ in range(100)]
        data = pd.DataFrame({'date': dates, 'views': views})
        
        spike_events = self.forecaster.detect_hype_events(data, "Test_Article")
        
        # Should detect no spikes
        assert len(spike_events) == 0
    
    def test_detect_hype_events_sustained_vs_temporary(self):
        """Test classification of sustained vs temporary spikes"""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        views = [1000 for _ in range(100)]
        
        # Add sustained spike (> 7 days)
        for i in range(30, 40):
            views[i] = 5000
        
        # Add temporary spike (< 7 days)
        for i in range(70, 73):
            views[i] = 5000
        
        data = pd.DataFrame({'date': dates, 'views': views})
        
        spike_events = self.forecaster.detect_hype_events(data, "Test_Article")
        
        # Should detect both spikes
        assert len(spike_events) >= 2
        
        # Check that we have both types
        spike_types = [event.spike_type for event in spike_events]
        assert 'sustained' in spike_types
        assert 'temporary' in spike_types
    
    def test_detect_hype_events_insufficient_data(self):
        """Test hype event detection with insufficient data"""
        # Less than 7 days
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        views = [1000 for _ in range(5)]
        data = pd.DataFrame({'date': dates, 'views': views})
        
        spike_events = self.forecaster.detect_hype_events(data, "Test_Article")
        
        # Should return empty list
        assert len(spike_events) == 0
    
    def test_detect_seasonality(self):
        """Test seasonality detection"""
        model = self.forecaster.train(self.test_data, "Test_Article")
        seasonality = self.forecaster.detect_seasonality(model)
        
        # Should return SeasonalityPattern
        assert isinstance(seasonality, SeasonalityPattern)
        assert seasonality.period in ['weekly', 'yearly', 'monthly', 'none']
        assert 0 <= seasonality.strength <= 1
    
    def test_model_caching(self):
        """Test that trained models are cached"""
        article = "Test_Article"
        
        # Train model
        model1 = self.forecaster.train(self.test_data, article)
        
        # Get cached model
        model2 = self.forecaster.get_cached_model(article)
        
        # Should be the same object
        assert model1 is model2
    
    def test_clear_cache_specific_article(self):
        """Test clearing cache for specific article"""
        article1 = "Article_1"
        article2 = "Article_2"
        
        # Train two models
        self.forecaster.train(self.test_data, article1)
        self.forecaster.train(self.test_data, article2)
        
        # Clear cache for article1
        self.forecaster.clear_cache(article1)
        
        # article1 should not be cached
        assert self.forecaster.get_cached_model(article1) is None
        
        # article2 should still be cached
        assert self.forecaster.get_cached_model(article2) is not None
    
    def test_clear_cache_all(self):
        """Test clearing all cached models"""
        # Train multiple models
        self.forecaster.train(self.test_data, "Article_1")
        self.forecaster.train(self.test_data, "Article_2")
        
        # Clear all cache
        self.forecaster.clear_cache()
        
        # No models should be cached
        assert self.forecaster.get_cached_model("Article_1") is None
        assert self.forecaster.get_cached_model("Article_2") is None
        assert len(self.forecaster._models) == 0
    
    def test_extreme_spike(self):
        """Test handling of extreme spikes"""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        views = [1000 for _ in range(100)]
        
        # Add extreme spike
        views[50] = 1000000  # 1000x normal
        
        data = pd.DataFrame({'date': dates, 'views': views})
        
        # Should not crash
        spike_events = self.forecaster.detect_hype_events(data, "Test_Article")
        
        # Should detect the spike
        assert len(spike_events) > 0
        
        # Magnitude should be very high
        max_magnitude = max(event.magnitude for event in spike_events)
        assert max_magnitude > 10.0
    
    def test_growth_rate_with_short_period(self):
        """Test growth rate calculation with period longer than data"""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        views = [1000 + i * 100 for i in range(10)]
        data = pd.DataFrame({'date': dates, 'views': views})
        
        # Request growth rate for 30 days, but only have 10 days
        growth_rate = self.forecaster.calculate_growth_rate(data, period_days=30)
        
        # Should use all available data
        assert isinstance(growth_rate, (int, float))
        assert not np.isnan(growth_rate)
    
    def test_forecast_result_structure(self):
        """Test that ForecastResult has correct structure"""
        model = self.forecaster.train(self.test_data, "Test_Article")
        result = self.forecaster.predict(model, periods=30, article="Test_Article")
        
        # Check all required fields
        assert isinstance(result, ForecastResult)
        assert result.article == "Test_Article"
        assert isinstance(result.predictions, pd.DataFrame)
        assert isinstance(result.seasonality, SeasonalityPattern)
        assert isinstance(result.growth_rate, (int, float))
        assert isinstance(result.confidence, (int, float))
    
    def test_data_with_nan_values(self):
        """Test handling of data with NaN values"""
        data = self.test_data.copy()
        
        # Add some NaN values
        data.loc[10:15, 'views'] = np.nan
        
        # Should handle NaN by dropping them
        model = self.forecaster.train(data, "Test_Article")
        assert model is not None
    
    def test_data_with_negative_views(self):
        """Test handling of negative views (should work, as they're valid numbers)"""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
        views = [1000 - i * 20 for i in range(100)]  # Goes negative
        data = pd.DataFrame({'date': dates, 'views': views})
        
        # Should not crash (Prophet can handle negative values)
        model = self.forecaster.train(data, "Test_Article")
        assert model is not None
