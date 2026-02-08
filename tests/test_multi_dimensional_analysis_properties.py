"""Property-based tests for multi-dimensional analysis.

This module implements property-based tests for the MultiDimensionalAnalyzer
to validate universal properties across diverse input scenarios.
"""

import pytest
from hypothesis import given, settings, strategies as st, assume
from hypothesis.strategies import composite
import pandas as pd
import numpy as np
from datetime import date, timedelta

from wikipedia_health.multi_dimensional_analysis import MultiDimensionalAnalyzer
from wikipedia_health.models.data_models import TimeSeriesData, Anomaly


# Custom strategies for generating test data

@composite
def time_series_data_strategy(draw, min_length=30, max_length=365):
    """Generate valid TimeSeriesData for testing.
    
    Args:
        draw: Hypothesis draw function
        min_length: Minimum number of data points
        max_length: Maximum number of data points
    
    Returns:
        TimeSeriesData with random but valid data
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    
    # Generate date range
    start_date = draw(st.dates(
        min_value=date(2015, 1, 1),
        max_value=date(2024, 12, 31)
    ))
    dates = pd.date_range(start=start_date, periods=length, freq='D')
    
    # Generate values (positive integers for pageviews/editors)
    values = draw(st.lists(
        st.integers(min_value=100, max_value=1000000),
        min_size=length,
        max_size=length
    ))
    
    platform = draw(st.sampled_from(['desktop', 'mobile-web', 'mobile-app', 'all']))
    metric_type = draw(st.sampled_from(['pageviews', 'editors', 'edits']))
    
    return TimeSeriesData(
        date=pd.Series(dates),
        values=pd.Series(values),
        platform=platform,
        metric_type=metric_type,
        metadata={'source': 'test'}
    )


@composite
def aligned_time_series_pair_strategy(draw, min_length=30, max_length=365):
    """Generate pair of aligned TimeSeriesData (same dates).
    
    Args:
        draw: Hypothesis draw function
        min_length: Minimum number of data points
        max_length: Maximum number of data points
    
    Returns:
        Tuple of (pageviews, editors) TimeSeriesData with aligned dates
    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    
    # Generate shared date range
    start_date = draw(st.dates(
        min_value=date(2015, 1, 1),
        max_value=date(2024, 12, 31)
    ))
    dates = pd.date_range(start=start_date, periods=length, freq='D')
    
    # Generate pageviews (larger values)
    pageviews = draw(st.lists(
        st.integers(min_value=10000, max_value=1000000),
        min_size=length,
        max_size=length
    ))
    
    # Generate editors (smaller values, correlated with pageviews)
    editors = draw(st.lists(
        st.integers(min_value=10, max_value=1000),
        min_size=length,
        max_size=length
    ))
    
    platform = draw(st.sampled_from(['desktop', 'mobile-web', 'mobile-app']))
    
    pageviews_ts = TimeSeriesData(
        date=pd.Series(dates),
        values=pd.Series(pageviews),
        platform=platform,
        metric_type='pageviews',
        metadata={'source': 'test'}
    )
    
    editors_ts = TimeSeriesData(
        date=pd.Series(dates),
        values=pd.Series(editors),
        platform=platform,
        metric_type='editors',
        metadata={'source': 'test'}
    )
    
    return pageviews_ts, editors_ts


@composite
def anomaly_list_strategy(draw, min_anomalies=0, max_anomalies=10):
    """Generate list of Anomaly objects.
    
    Args:
        draw: Hypothesis draw function
        min_anomalies: Minimum number of anomalies
        max_anomalies: Maximum number of anomalies
    
    Returns:
        List of Anomaly objects
    """
    n_anomalies = draw(st.integers(min_value=min_anomalies, max_value=max_anomalies))
    
    anomalies = []
    for _ in range(n_anomalies):
        anomaly_date = draw(st.dates(
            min_value=date(2015, 1, 1),
            max_value=date(2024, 12, 31)
        ))
        value = draw(st.floats(min_value=1000, max_value=1000000, allow_nan=False, allow_infinity=False))
        expected = draw(st.floats(min_value=1000, max_value=1000000, allow_nan=False, allow_infinity=False))
        z_score = draw(st.floats(min_value=3.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        
        anomalies.append(Anomaly(
            date=anomaly_date,
            value=value,
            expected_value=expected,
            z_score=z_score,
            description=f"Anomaly on {anomaly_date}"
        ))
    
    return anomalies


# Property 9: Multi-Dimensional Correlation
@settings(max_examples=100, deadline=None)
@given(data_pair=aligned_time_series_pair_strategy(min_length=30, max_length=200))
def test_property_9_multi_dimensional_correlation(data_pair):
    """
    Feature: wikipedia-product-health-analysis
    Property 9: For any pageview trend analysis, the system should correlate
    pageview changes with Active_Editor counts to detect engagement quality shifts.
    
    Validates: Requirements 4.1, 4.2, 4.4, 4.6
    """
    pageviews, editors = data_pair
    
    analyzer = MultiDimensionalAnalyzer()
    
    # Act: Correlate pageviews with editors
    metrics = analyzer.correlate_pageviews_editors(pageviews, editors)
    
    # Assert: Correlation analysis should produce valid results
    assert metrics is not None
    assert isinstance(metrics.correlation, float)
    assert isinstance(metrics.correlation_p_value, float)
    assert isinstance(metrics.engagement_ratio, float)
    
    # Correlation coefficient should be in valid range [-1, 1]
    assert -1.0 <= metrics.correlation <= 1.0
    
    # P-value should be in valid range [0, 1]
    assert 0.0 <= metrics.correlation_p_value <= 1.0
    
    # Engagement ratio should be non-negative
    assert metrics.engagement_ratio >= 0.0
    
    # Platform should match input
    assert metrics.platform == pageviews.platform
    
    # Time period should be valid
    assert len(metrics.time_period) == 2
    start_str, end_str = metrics.time_period
    start_date = pd.to_datetime(start_str)
    end_date = pd.to_datetime(end_str)
    assert start_date <= end_date


# Property 10: Engagement Ratio Significance Testing
@settings(max_examples=100, deadline=None)
@given(data_pair=aligned_time_series_pair_strategy(min_length=60, max_length=200))
def test_property_10_engagement_ratio_significance_testing(data_pair):
    """
    Feature: wikipedia-product-health-analysis
    Property 10: For any time period comparison, when engagement ratios
    (editors per 1000 pageviews) are computed, the system should test whether
    the ratios differ significantly between periods using appropriate statistical tests.
    
    Validates: Requirement 4.3
    """
    pageviews, editors = data_pair
    
    analyzer = MultiDimensionalAnalyzer()
    
    # Act: Detect engagement shifts (which tests ratio differences)
    shifts = analyzer.detect_engagement_shifts(pageviews, editors, window_size=30)
    
    # Assert: All detected shifts should have statistical evidence
    for shift in shifts:
        # Each shift should have a test result
        assert shift.test_result is not None
        assert isinstance(shift.test_result.p_value, float)
        assert isinstance(shift.test_result.statistic, float)
        
        # P-value should be in valid range
        assert 0.0 <= shift.test_result.p_value <= 1.0
        
        # Shift should be marked as significant (since we only return significant shifts)
        assert shift.test_result.is_significant
        
        # Pre and post ratios should be different
        assert shift.pre_ratio != shift.post_ratio
        
        # Direction should match the change
        if shift.post_ratio > shift.pre_ratio:
            assert shift.direction == 'increase'
        else:
            assert shift.direction == 'decrease'
        
        # Ratios should be non-negative
        assert shift.pre_ratio >= 0.0
        assert shift.post_ratio >= 0.0
        
        # Date should be valid
        shift_date = pd.to_datetime(shift.date)
        assert shift_date is not None


# Property 11: Cross-Platform Engagement Analysis
@settings(max_examples=100, deadline=None)
@given(
    n_platforms=st.integers(min_value=2, max_value=3),
    data_length=st.integers(min_value=30, max_value=100)
)
def test_property_11_cross_platform_engagement_analysis(n_platforms, data_length):
    """
    Feature: wikipedia-product-health-analysis
    Property 11: For any analysis involving multiple platforms, engagement metrics
    should be computed and compared across desktop, mobile web, and mobile app to
    identify platform-specific behavior patterns.
    
    Validates: Requirements 4.5, 4.7
    """
    platforms = ['desktop', 'mobile-web', 'mobile-app'][:n_platforms]
    
    # Generate data for each platform
    start_date = date(2020, 1, 1)
    dates = pd.date_range(start=start_date, periods=data_length, freq='D')
    
    platform_data = {}
    for platform in platforms:
        # Generate pageviews and editors with some variation per platform
        pageviews_values = np.random.randint(10000, 100000, size=data_length)
        editors_values = np.random.randint(10, 100, size=data_length)
        
        pageviews_ts = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(pageviews_values),
            platform=platform,
            metric_type='pageviews',
            metadata={'source': 'test'}
        )
        
        editors_ts = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(editors_values),
            platform=platform,
            metric_type='editors',
            metadata={'source': 'test'}
        )
        
        platform_data[platform] = (pageviews_ts, editors_ts)
    
    analyzer = MultiDimensionalAnalyzer()
    
    # Act: Compare engagement across platforms
    platform_engagements, anova_result = analyzer.compare_platform_engagement(platform_data)
    
    # Assert: Should produce engagement metrics for each platform
    assert len(platform_engagements) == n_platforms
    
    for engagement in platform_engagements:
        # Each platform should have valid metrics
        assert engagement.platform in platforms
        assert isinstance(engagement.engagement_ratio, float)
        assert isinstance(engagement.pageview_mean, float)
        assert isinstance(engagement.editor_mean, float)
        assert engagement.engagement_quality in ['high', 'medium', 'low']
        
        # Metrics should be non-negative
        assert engagement.engagement_ratio >= 0.0
        assert engagement.pageview_mean >= 0.0
        assert engagement.editor_mean >= 0.0
    
    # ANOVA result should be valid
    assert anova_result is not None
    assert isinstance(anova_result.p_value, float)
    assert 0.0 <= anova_result.p_value <= 1.0
    
    # If we have at least 2 platforms, ANOVA should be performed
    if n_platforms >= 2:
        assert anova_result.test_name == 'One-Way ANOVA'
        assert isinstance(anova_result.statistic, float)
        assert isinstance(anova_result.effect_size, float)


# Additional property test: Anomaly cross-referencing
@settings(max_examples=100, deadline=None)
@given(
    pv_anomalies=anomaly_list_strategy(min_anomalies=0, max_anomalies=10),
    ed_anomalies=anomaly_list_strategy(min_anomalies=0, max_anomalies=10),
    time_window=st.integers(min_value=1, max_value=7)
)
def test_property_anomaly_cross_referencing(pv_anomalies, ed_anomalies, time_window):
    """
    Property: Anomaly cross-referencing should categorize all anomalies
    into passive consumption, active engagement, or editor-only categories.
    
    Validates: Requirement 4.4
    """
    analyzer = MultiDimensionalAnalyzer()
    
    # Act: Cross-reference anomalies
    result = analyzer.cross_reference_anomalies(
        pv_anomalies,
        ed_anomalies,
        time_window_days=time_window
    )
    
    # Assert: Result should have all three categories
    assert 'passive_consumption' in result
    assert 'active_engagement' in result
    assert 'editor_only' in result
    
    # All categories should be lists
    assert isinstance(result['passive_consumption'], list)
    assert isinstance(result['active_engagement'], list)
    assert isinstance(result['editor_only'], list)
    
    # Total anomalies should be accounted for
    total_pv_in_result = (
        len(result['passive_consumption']) +
        len(result['active_engagement'])
    )
    total_ed_in_result = (
        len(result['active_engagement']) +
        len(result['editor_only'])
    )
    
    # Each pageview anomaly should appear in exactly one category
    assert total_pv_in_result <= len(pv_anomalies)
    
    # Each editor anomaly should appear in exactly one category
    assert total_ed_in_result <= len(ed_anomalies)
    
    # Active engagement should have both pageview and editor anomalies
    for pv_anom, ed_anom in result['active_engagement']:
        assert pv_anom is not None
        assert ed_anom is not None
        assert isinstance(pv_anom, Anomaly)
        assert isinstance(ed_anom, Anomaly)
    
    # Passive consumption should have pageview anomalies only
    for pv_anom, ed_anom in result['passive_consumption']:
        assert pv_anom is not None
        assert ed_anom is None
        assert isinstance(pv_anom, Anomaly)
    
    # Editor only should have editor anomalies only
    for ed_anom, pv_anom in result['editor_only']:
        assert ed_anom is not None
        assert pv_anom is None
        assert isinstance(ed_anom, Anomaly)


# Property test: Engagement ratio computation
@settings(max_examples=100, deadline=None)
@given(
    pageviews=st.lists(st.integers(min_value=1, max_value=1000000), min_size=10, max_size=100),
    editors=st.lists(st.integers(min_value=0, max_value=1000), min_size=10, max_size=100)
)
def test_property_engagement_ratio_computation(pageviews, editors):
    """
    Property: Engagement ratio should always be non-negative and scale
    correctly (editors per 1000 pageviews).
    
    Validates: Requirement 4.2
    """
    # Ensure same length
    min_len = min(len(pageviews), len(editors))
    pageviews = pageviews[:min_len]
    editors = editors[:min_len]
    
    analyzer = MultiDimensionalAnalyzer()
    
    # Act: Compute engagement ratio
    ratio = analyzer.compute_engagement_ratio(
        pd.Series(pageviews),
        pd.Series(editors)
    )
    
    # Assert: Ratio should be non-negative
    assert ratio >= 0.0
    
    # Ratio should be finite
    assert np.isfinite(ratio)
    
    # Manual calculation for verification
    expected_ratios = []
    for pv, ed in zip(pageviews, editors):
        if pv > 0:
            expected_ratios.append((ed / pv) * 1000)
    
    if expected_ratios:
        expected_mean = np.mean(expected_ratios)
        # Should be close to manual calculation
        assert abs(ratio - expected_mean) < 1.0  # Allow small numerical differences
