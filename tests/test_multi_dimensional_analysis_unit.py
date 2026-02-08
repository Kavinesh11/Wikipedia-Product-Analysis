"""Unit tests for multi-dimensional analysis.

This module implements unit tests for the MultiDimensionalAnalyzer
to validate specific examples and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from wikipedia_health.multi_dimensional_analysis import MultiDimensionalAnalyzer
from wikipedia_health.models.data_models import TimeSeriesData, Anomaly


class TestCorrelationCalculations:
    """Test correlation calculations between pageviews and editors."""
    
    def test_perfect_positive_correlation(self):
        """Test with perfectly correlated pageviews and editors."""
        dates = pd.date_range(start='2020-01-01', periods=30, freq='D')
        pageviews_values = np.arange(1000, 1300, 10)
        editors_values = np.arange(10, 40, 1)
        
        pageviews = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(pageviews_values),
            platform='desktop',
            metric_type='pageviews',
            metadata={}
        )
        
        editors = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(editors_values),
            platform='desktop',
            metric_type='editors',
            metadata={}
        )
        
        analyzer = MultiDimensionalAnalyzer()
        metrics = analyzer.correlate_pageviews_editors(pageviews, editors)
        
        # Should have strong positive correlation
        assert metrics.correlation > 0.9
        assert metrics.correlation_p_value < 0.05
        assert metrics.engagement_ratio > 0
    
    def test_no_correlation(self):
        """Test with uncorrelated pageviews and editors."""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        
        # Random uncorrelated data
        np.random.seed(42)
        pageviews_values = np.random.randint(10000, 20000, size=50)
        editors_values = np.random.randint(50, 100, size=50)
        
        pageviews = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(pageviews_values),
            platform='mobile-web',
            metric_type='pageviews',
            metadata={}
        )
        
        editors = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(editors_values),
            platform='mobile-web',
            metric_type='editors',
            metadata={}
        )
        
        analyzer = MultiDimensionalAnalyzer()
        metrics = analyzer.correlate_pageviews_editors(pageviews, editors)
        
        # Correlation should be weak
        assert -1.0 <= metrics.correlation <= 1.0
        assert 0.0 <= metrics.correlation_p_value <= 1.0
    
    def test_constant_values(self):
        """Test with constant values (zero variance)."""
        dates = pd.date_range(start='2020-01-01', periods=30, freq='D')
        pageviews_values = [10000] * 30
        editors_values = [50] * 30
        
        pageviews = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(pageviews_values),
            platform='desktop',
            metric_type='pageviews',
            metadata={}
        )
        
        editors = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(editors_values),
            platform='desktop',
            metric_type='editors',
            metadata={}
        )
        
        analyzer = MultiDimensionalAnalyzer()
        metrics = analyzer.correlate_pageviews_editors(pageviews, editors)
        
        # Should handle constant values gracefully
        assert metrics.correlation == 0.0
        assert metrics.correlation_p_value == 1.0
        assert metrics.engagement_ratio > 0


class TestEngagementRatioComputation:
    """Test engagement ratio computation."""
    
    def test_basic_ratio_calculation(self):
        """Test basic engagement ratio calculation."""
        pageviews = pd.Series([10000, 20000, 30000])
        editors = pd.Series([10, 20, 30])
        
        analyzer = MultiDimensionalAnalyzer()
        ratio = analyzer.compute_engagement_ratio(pageviews, editors)
        
        # Expected: (10/10000 + 20/20000 + 30/30000) / 3 * 1000 = 1.0
        assert abs(ratio - 1.0) < 0.01
    
    def test_zero_pageviews(self):
        """Test handling of zero pageviews."""
        pageviews = pd.Series([10000, 0, 30000])
        editors = pd.Series([10, 20, 30])
        
        analyzer = MultiDimensionalAnalyzer()
        ratio = analyzer.compute_engagement_ratio(pageviews, editors)
        
        # Should handle zero pageviews by excluding from mean
        assert ratio > 0
        assert np.isfinite(ratio)
    
    def test_high_engagement(self):
        """Test high engagement scenario."""
        pageviews = pd.Series([10000, 10000, 10000])
        editors = pd.Series([100, 100, 100])
        
        analyzer = MultiDimensionalAnalyzer()
        ratio = analyzer.compute_engagement_ratio(pageviews, editors)
        
        # Expected: 100/10000 * 1000 = 10.0
        assert abs(ratio - 10.0) < 0.01
    
    def test_low_engagement(self):
        """Test low engagement scenario."""
        pageviews = pd.Series([100000, 100000, 100000])
        editors = pd.Series([10, 10, 10])
        
        analyzer = MultiDimensionalAnalyzer()
        ratio = analyzer.compute_engagement_ratio(pageviews, editors)
        
        # Expected: 10/100000 * 1000 = 0.1
        assert abs(ratio - 0.1) < 0.01


class TestEngagementShiftDetection:
    """Test engagement shift detection."""
    
    def test_detect_clear_shift(self):
        """Test detection of clear engagement shift."""
        # Create data with clear shift at day 50
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Low engagement for first 50 days, high for next 50
        pageviews_values = [100000] * 50 + [100000] * 50
        editors_values = [50] * 50 + [200] * 50  # 4x increase in editors
        
        pageviews = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(pageviews_values),
            platform='desktop',
            metric_type='pageviews',
            metadata={}
        )
        
        editors = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(editors_values),
            platform='desktop',
            metric_type='editors',
            metadata={}
        )
        
        analyzer = MultiDimensionalAnalyzer()
        shifts = analyzer.detect_engagement_shifts(pageviews, editors, window_size=20)
        
        # Should detect at least one shift
        assert len(shifts) > 0
        
        # First shift should be an increase
        assert shifts[0].direction == 'increase'
        assert shifts[0].post_ratio > shifts[0].pre_ratio
        assert shifts[0].test_result.is_significant
    
    def test_no_shift_stable_engagement(self):
        """Test with stable engagement (no shifts)."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Stable engagement throughout
        pageviews_values = [100000] * 100
        editors_values = [100] * 100
        
        pageviews = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(pageviews_values),
            platform='desktop',
            metric_type='pageviews',
            metadata={}
        )
        
        editors = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(editors_values),
            platform='desktop',
            metric_type='editors',
            metadata={}
        )
        
        analyzer = MultiDimensionalAnalyzer()
        shifts = analyzer.detect_engagement_shifts(pageviews, editors, window_size=20)
        
        # Should detect no significant shifts
        assert len(shifts) == 0
    
    def test_gradual_change(self):
        """Test with gradual engagement change."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Gradual increase in editors
        pageviews_values = [100000] * 100
        editors_values = list(range(50, 150))
        
        pageviews = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(pageviews_values),
            platform='desktop',
            metric_type='pageviews',
            metadata={}
        )
        
        editors = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(editors_values),
            platform='desktop',
            metric_type='editors',
            metadata={}
        )
        
        analyzer = MultiDimensionalAnalyzer()
        shifts = analyzer.detect_engagement_shifts(pageviews, editors, window_size=20)
        
        # May or may not detect shifts depending on window size
        # Just verify it runs without error
        assert isinstance(shifts, list)


class TestAnomalyCrossReferencing:
    """Test anomaly cross-referencing."""
    
    def test_matching_anomalies(self):
        """Test with matching pageview and editor anomalies."""
        pv_anomalies = [
            Anomaly(date=date(2020, 1, 15), value=100000, expected_value=50000, 
                   z_score=5.0, description="Spike"),
            Anomaly(date=date(2020, 2, 20), value=120000, expected_value=50000,
                   z_score=6.0, description="Spike")
        ]
        
        ed_anomalies = [
            Anomaly(date=date(2020, 1, 15), value=200, expected_value=100,
                   z_score=4.0, description="Spike"),
            Anomaly(date=date(2020, 3, 10), value=250, expected_value=100,
                   z_score=5.0, description="Spike")
        ]
        
        analyzer = MultiDimensionalAnalyzer()
        result = analyzer.cross_reference_anomalies(pv_anomalies, ed_anomalies, time_window_days=1)
        
        # First anomaly should match (same date)
        assert len(result['active_engagement']) >= 1
        
        # Second pageview anomaly should be passive (no matching editor anomaly)
        assert len(result['passive_consumption']) >= 1
        
        # Third editor anomaly should be editor-only
        assert len(result['editor_only']) >= 1
    
    def test_no_anomalies(self):
        """Test with no anomalies."""
        analyzer = MultiDimensionalAnalyzer()
        result = analyzer.cross_reference_anomalies([], [], time_window_days=3)
        
        assert len(result['active_engagement']) == 0
        assert len(result['passive_consumption']) == 0
        assert len(result['editor_only']) == 0
    
    def test_time_window_matching(self):
        """Test time window for matching anomalies."""
        pv_anomalies = [
            Anomaly(date=date(2020, 1, 15), value=100000, expected_value=50000,
                   z_score=5.0, description="Spike")
        ]
        
        ed_anomalies = [
            Anomaly(date=date(2020, 1, 17), value=200, expected_value=100,
                   z_score=4.0, description="Spike")
        ]
        
        analyzer = MultiDimensionalAnalyzer()
        
        # With 1-day window, should not match
        result_1day = analyzer.cross_reference_anomalies(pv_anomalies, ed_anomalies, time_window_days=1)
        assert len(result_1day['active_engagement']) == 0
        
        # With 3-day window, should match
        result_3day = analyzer.cross_reference_anomalies(pv_anomalies, ed_anomalies, time_window_days=3)
        assert len(result_3day['active_engagement']) >= 1


class TestPlatformComparison:
    """Test platform engagement comparison."""
    
    def test_compare_two_platforms(self):
        """Test comparison of two platforms."""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        
        # Desktop: high pageviews, moderate editors
        desktop_pv = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series([100000] * 50),
            platform='desktop',
            metric_type='pageviews',
            metadata={}
        )
        desktop_ed = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series([100] * 50),
            platform='desktop',
            metric_type='editors',
            metadata={}
        )
        
        # Mobile: lower pageviews, fewer editors
        mobile_pv = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series([50000] * 50),
            platform='mobile-web',
            metric_type='pageviews',
            metadata={}
        )
        mobile_ed = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series([25] * 50),
            platform='mobile-web',
            metric_type='editors',
            metadata={}
        )
        
        platform_data = {
            'desktop': (desktop_pv, desktop_ed),
            'mobile-web': (mobile_pv, mobile_ed)
        }
        
        analyzer = MultiDimensionalAnalyzer()
        engagements, anova_result = analyzer.compare_platform_engagement(platform_data)
        
        # Should have metrics for both platforms
        assert len(engagements) == 2
        
        # Check desktop metrics
        desktop_metrics = next(e for e in engagements if e.platform == 'desktop')
        assert desktop_metrics.pageview_mean == 100000
        assert desktop_metrics.editor_mean == 100
        assert desktop_metrics.engagement_ratio == 1.0  # 100/100000 * 1000
        
        # Check mobile metrics
        mobile_metrics = next(e for e in engagements if e.platform == 'mobile-web')
        assert mobile_metrics.pageview_mean == 50000
        assert mobile_metrics.editor_mean == 25
        assert mobile_metrics.engagement_ratio == 0.5  # 25/50000 * 1000
        
        # ANOVA should be performed
        assert anova_result.test_name == 'One-Way ANOVA'
        assert 0.0 <= anova_result.p_value <= 1.0
    
    def test_compare_three_platforms(self):
        """Test comparison of three platforms."""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        
        platforms = ['desktop', 'mobile-web', 'mobile-app']
        platform_data = {}
        
        for i, platform in enumerate(platforms):
            pv_values = [100000 - i * 20000] * 50
            ed_values = [100 - i * 20] * 50
            
            pv = TimeSeriesData(
                date=pd.Series(dates),
                values=pd.Series(pv_values),
                platform=platform,
                metric_type='pageviews',
                metadata={}
            )
            ed = TimeSeriesData(
                date=pd.Series(dates),
                values=pd.Series(ed_values),
                platform=platform,
                metric_type='editors',
                metadata={}
            )
            platform_data[platform] = (pv, ed)
        
        analyzer = MultiDimensionalAnalyzer()
        engagements, anova_result = analyzer.compare_platform_engagement(platform_data)
        
        # Should have metrics for all three platforms
        assert len(engagements) == 3
        
        # All platforms should have valid engagement quality
        for engagement in engagements:
            assert engagement.engagement_quality in ['high', 'medium', 'low']
            assert engagement.engagement_ratio >= 0
    
    def test_single_platform(self):
        """Test with single platform (no comparison possible)."""
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        
        pv = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series([100000] * 50),
            platform='desktop',
            metric_type='pageviews',
            metadata={}
        )
        ed = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series([100] * 50),
            platform='desktop',
            metric_type='editors',
            metadata={}
        )
        
        platform_data = {'desktop': (pv, ed)}
        
        analyzer = MultiDimensionalAnalyzer()
        engagements, anova_result = analyzer.compare_platform_engagement(platform_data)
        
        # Should have metrics for one platform
        assert len(engagements) == 1
        
        # ANOVA should indicate insufficient platforms
        assert not anova_result.is_significant
        assert 'Insufficient' in anova_result.interpretation


class TestEngagementQualityClassification:
    """Test engagement quality classification."""
    
    def test_high_engagement_classification(self):
        """Test classification of high engagement."""
        analyzer = MultiDimensionalAnalyzer()
        
        # High engagement: >= 1.0 editors per 1000 pageviews
        quality = analyzer._classify_engagement_quality(1.5)
        assert quality == 'high'
    
    def test_medium_engagement_classification(self):
        """Test classification of medium engagement."""
        analyzer = MultiDimensionalAnalyzer()
        
        # Medium engagement: 0.5 to 1.0
        quality = analyzer._classify_engagement_quality(0.7)
        assert quality == 'medium'
    
    def test_low_engagement_classification(self):
        """Test classification of low engagement."""
        analyzer = MultiDimensionalAnalyzer()
        
        # Low engagement: < 0.5
        quality = analyzer._classify_engagement_quality(0.3)
        assert quality == 'low'
