"""Property-Based Tests for Time Series Forecasting

Tests correctness properties for the TimeSeriesForecaster component.
Uses Hypothesis for property-based testing with randomized inputs.
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.analytics.forecaster import TimeSeriesForecaster


# ============================================================================
# Test Strategies
# ============================================================================

@st.composite
def time_series_data(draw, min_days=90, max_days=365):
    """Generate valid time series data for testing
    
    Args:
        draw: Hypothesis draw function
        min_days: Minimum number of days
        max_days: Maximum number of days
        
    Returns:
        DataFrame with 'date' and 'views' columns
    """
    n_days = draw(st.integers(min_value=min_days, max_value=max_days))
    
    # Generate dates
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate views with some realistic patterns
    base_views = draw(st.integers(min_value=100, max_value=10000))
    trend = draw(st.floats(min_value=-0.01, max_value=0.05))
    noise_level = draw(st.floats(min_value=0.1, max_value=0.3))
    
    views = []
    for i in range(n_days):
        # Add trend
        trend_component = base_views * (1 + trend * i)
        # Add noise
        noise = np.random.normal(0, noise_level * base_views)
        view_count = max(0, int(trend_component + noise))
        views.append(view_count)
    
    return pd.DataFrame({
        'date': dates,
        'views': views
    })


@st.composite
def insufficient_time_series_data(draw):
    """Generate time series data with < 90 days
    
    Returns:
        DataFrame with insufficient data
    """
    n_days = draw(st.integers(min_value=1, max_value=89))
    
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    views = [draw(st.integers(min_value=0, max_value=10000)) for _ in range(n_days)]
    
    return pd.DataFrame({
        'date': dates,
        'views': views
    })


# ============================================================================
# Property 20: Minimum Training Data Requirement
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 20: Minimum Training Data Requirement
@given(data=insufficient_time_series_data())
@settings(max_examples=5, deadline=None)
def test_property_20_minimum_training_data_requirement(data):
    """
    Property 20: For any forecasting model training request with less than 90 days
    of historical data, the System should reject the request or issue a warning.
    
    Validates: Requirements 5.2
    """
    forecaster = TimeSeriesForecaster()
    article = "Test_Article"
    
    # Attempt to train with insufficient data should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        forecaster.train(data, article)
    
    # Verify error message mentions insufficient data
    error_msg = str(exc_info.value).lower()
    assert "insufficient" in error_msg or "minimum" in error_msg or "90" in error_msg, \
        f"Error message should mention insufficient data: {exc_info.value}"


# ============================================================================
# Property 21: Prediction Confidence Intervals
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 21: Prediction Confidence Intervals
@given(
    data=time_series_data(min_days=90, max_days=200),
    periods=st.integers(min_value=1, max_value=60)
)
@settings(max_examples=5, deadline=None)
def test_property_21_prediction_confidence_intervals(data, periods):
    """
    Property 21: For any trained forecasting model and prediction request,
    the System should return predictions with both upper and lower confidence bounds.
    
    Validates: Requirements 5.3
    """
    forecaster = TimeSeriesForecaster()
    article = "Test_Article"
    
    # Train model
    model = forecaster.train(data, article)
    
    # Generate predictions
    result = forecaster.predict(model, periods=periods, article=article)
    
    # Verify predictions dataframe has required columns
    assert 'date' in result.predictions.columns, "Predictions must have 'date' column"
    assert 'yhat' in result.predictions.columns, "Predictions must have 'yhat' column"
    assert 'yhat_lower' in result.predictions.columns, "Predictions must have 'yhat_lower' column"
    assert 'yhat_upper' in result.predictions.columns, "Predictions must have 'yhat_upper' column"
    
    # Verify we have the correct number of predictions
    assert len(result.predictions) == periods, \
        f"Should have {periods} predictions, got {len(result.predictions)}"
    
    # Verify confidence bounds are valid (lower <= yhat <= upper)
    for idx, row in result.predictions.iterrows():
        assert row['yhat_lower'] <= row['yhat'], \
            f"Lower bound {row['yhat_lower']} should be <= prediction {row['yhat']}"
        assert row['yhat'] <= row['yhat_upper'], \
            f"Prediction {row['yhat']} should be <= upper bound {row['yhat_upper']}"


# ============================================================================
# Property 22: Seasonal Pattern Detection
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 22: Seasonal Pattern Detection
@given(data=time_series_data(min_days=180, max_days=365))
@settings(max_examples=5, deadline=None)
def test_property_22_seasonal_pattern_detection(data):
    """
    Property 22: For any pageview time series with periodic patterns
    (weekly, monthly, yearly), the System should detect and report the seasonality.
    
    Validates: Requirements 5.4
    """
    forecaster = TimeSeriesForecaster()
    article = "Test_Article"
    
    # Train model
    model = forecaster.train(data, article)
    
    # Detect seasonality
    seasonality = forecaster.detect_seasonality(model)
    
    # Verify seasonality object has required fields
    assert hasattr(seasonality, 'period'), "Seasonality must have 'period' attribute"
    assert hasattr(seasonality, 'strength'), "Seasonality must have 'strength' attribute"
    assert hasattr(seasonality, 'peak_day'), "Seasonality must have 'peak_day' attribute"
    
    # Verify period is valid
    valid_periods = ['weekly', 'yearly', 'monthly', 'none']
    assert seasonality.period in valid_periods, \
        f"Period must be one of {valid_periods}, got '{seasonality.period}'"
    
    # Verify strength is in valid range [0, 1]
    assert 0 <= seasonality.strength <= 1, \
        f"Strength must be between 0 and 1, got {seasonality.strength}"
    
    # If period is detected, strength should be > 0
    if seasonality.period != 'none':
        assert seasonality.strength > 0, \
            f"If seasonality detected, strength should be > 0, got {seasonality.strength}"


# ============================================================================
# Property 23: Hype Event Flagging
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 23: Hype Event Flagging
@given(
    base_views=st.integers(min_value=100, max_value=5000),
    spike_multiplier=st.floats(min_value=3.0, max_value=10.0)
)
@settings(max_examples=5, deadline=None)
def test_property_23_hype_event_flagging(base_views, spike_multiplier):
    """
    Property 23: For any article where pageview growth exceeds 2 standard deviations
    from the mean, the System should flag it as a launch hype event.
    
    Validates: Requirements 5.5
    """
    forecaster = TimeSeriesForecaster()
    article = "Test_Article"
    
    # Create time series with a clear spike
    n_days = 100
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Normal views for most days
    views = [base_views + np.random.randint(-50, 50) for _ in range(n_days)]
    
    # Add a spike in the middle (> 2 std dev above mean)
    spike_start = 50
    spike_duration = 5
    spike_value = int(base_views * spike_multiplier)
    
    for i in range(spike_start, spike_start + spike_duration):
        views[i] = spike_value
    
    data = pd.DataFrame({
        'date': dates,
        'views': views
    })
    
    # Detect hype events
    spike_events = forecaster.detect_hype_events(data, article)
    
    # Should detect at least one spike event
    assert len(spike_events) > 0, "Should detect at least one hype event"
    
    # Verify spike event properties
    for event in spike_events:
        assert hasattr(event, 'timestamp'), "Event must have timestamp"
        assert hasattr(event, 'magnitude'), "Event must have magnitude"
        assert hasattr(event, 'duration_days'), "Event must have duration_days"
        assert hasattr(event, 'spike_type'), "Event must have spike_type"
        
        # Magnitude should be > 2 (more than 2 std devs)
        assert event.magnitude > 2.0, \
            f"Hype event magnitude should be > 2 std devs, got {event.magnitude}"
        
        # Spike type should be valid
        assert event.spike_type in ['sustained', 'temporary'], \
            f"Invalid spike_type: {event.spike_type}"


# ============================================================================
# Property 24: View Growth Rate Calculation
# ============================================================================

# Feature: wikipedia-intelligence-system, Property 24: View Growth Rate Calculation
@given(
    views_start=st.integers(min_value=1, max_value=100000),
    views_end=st.integers(min_value=1, max_value=100000),
    period_days=st.integers(min_value=1, max_value=90)
)
@settings(max_examples=5, deadline=None)
def test_property_24_view_growth_rate_calculation(views_start, views_end, period_days):
    """
    Property 24: For any pageview time series and time period,
    the System should calculate growth rate as
    ((views_end - views_start) / views_start) * 100
    
    Validates: Requirements 5.6
    """
    forecaster = TimeSeriesForecaster()
    
    # Create simple time series with known start and end values
    n_days = max(period_days + 1, 2)
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Linear interpolation between start and end
    views = np.linspace(views_start, views_end, n_days).astype(int)
    
    data = pd.DataFrame({
        'date': dates,
        'views': views
    })
    
    # Calculate growth rate
    growth_rate = forecaster.calculate_growth_rate(data, period_days=period_days)
    
    # Calculate expected growth rate
    if len(views) <= period_days:
        actual_start = views[0]
        actual_end = views[-1]
    else:
        actual_start = views[-period_days]
        actual_end = views[-1]
    
    if actual_start == 0:
        expected = 0.0 if actual_end == 0 else 100.0
    else:
        expected = ((actual_end - actual_start) / actual_start) * 100
    
    # Allow small floating point error
    assert abs(growth_rate - expected) < 0.01, \
        f"Growth rate {growth_rate} doesn't match expected {expected}"


# ============================================================================
# Additional Property Tests
# ============================================================================

@given(data=time_series_data(min_days=90, max_days=200))
@settings(max_examples=5, deadline=None)
def test_model_caching(data):
    """Test that trained models are cached correctly"""
    forecaster = TimeSeriesForecaster()
    article = "Test_Article"
    
    # Train model
    model1 = forecaster.train(data, article)
    
    # Get cached model
    model2 = forecaster.get_cached_model(article)
    
    # Should be the same model object
    assert model1 is model2, "Cached model should be the same object"
    
    # Clear cache
    forecaster.clear_cache(article)
    
    # Should no longer be cached
    model3 = forecaster.get_cached_model(article)
    assert model3 is None, "Model should not be cached after clearing"


@given(
    data=time_series_data(min_days=90, max_days=200),
    periods=st.integers(min_value=1, max_value=30)
)
@settings(max_examples=5, deadline=None)
def test_confidence_score_range(data, periods):
    """Test that confidence scores are in valid range [0, 1]"""
    forecaster = TimeSeriesForecaster()
    article = "Test_Article"
    
    model = forecaster.train(data, article)
    result = forecaster.predict(model, periods=periods, article=article)
    
    # Confidence should be between 0 and 1
    assert 0 <= result.confidence <= 1, \
        f"Confidence must be between 0 and 1, got {result.confidence}"


@given(data=time_series_data(min_days=90, max_days=200))
@settings(max_examples=5, deadline=None)
def test_growth_rate_with_zero_start(data):
    """Test growth rate calculation when starting views are zero"""
    forecaster = TimeSeriesForecaster()
    
    # Create data with zero start
    df = data.copy()
    df.loc[0, 'views'] = 0
    
    # Should not raise error
    growth_rate = forecaster.calculate_growth_rate(df, period_days=30)
    
    # Should return valid number
    assert isinstance(growth_rate, (int, float)), "Growth rate should be numeric"
    assert not np.isnan(growth_rate), "Growth rate should not be NaN"
    assert not np.isinf(growth_rate), "Growth rate should not be infinite"
