"""Unit Tests for Hype Detection Engine

Tests specific examples, edge cases, and error conditions for hype detection.
"""
import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.analytics.hype_detection import HypeDetectionEngine
from src.storage.dto import SpikeEvent


class TestHypeDetectionEngine:
    """Test suite for HypeDetectionEngine class"""
    
    def test_initialization_default_threshold(self):
        """Test engine initializes with default threshold"""
        engine = HypeDetectionEngine()
        assert engine.hype_threshold == 0.75
    
    def test_initialization_custom_threshold(self):
        """Test engine initializes with custom threshold"""
        engine = HypeDetectionEngine(hype_threshold=0.5)
        assert engine.hype_threshold == 0.5
    
    def test_initialization_invalid_threshold(self):
        """Test engine rejects invalid threshold values"""
        with pytest.raises(ValueError):
            HypeDetectionEngine(hype_threshold=1.5)
        
        with pytest.raises(ValueError):
            HypeDetectionEngine(hype_threshold=-0.1)
    
    def test_calculate_hype_score_high_hype(self):
        """Test hype score calculation for high-hype scenario"""
        engine = HypeDetectionEngine()
        
        # High hype: rapid growth across all metrics
        score = engine.calculate_hype_score(
            view_velocity=800.0,  # 800% growth
            edit_growth=400.0,    # 400% growth
            content_expansion=250.0  # 250% growth
        )
        
        # Should be close to 1.0 (high hype)
        assert score > 0.8
        assert score <= 1.0
    
    def test_calculate_hype_score_medium_hype(self):
        """Test hype score calculation for medium-hype scenario"""
        engine = HypeDetectionEngine()
        
        # Medium hype: moderate growth
        score = engine.calculate_hype_score(
            view_velocity=300.0,
            edit_growth=150.0,
            content_expansion=100.0
        )
        
        # Should be in medium range
        assert 0.3 < score < 0.7
    
    def test_calculate_hype_score_low_hype(self):
        """Test hype score calculation for low-hype scenario"""
        engine = HypeDetectionEngine()
        
        # Low hype: minimal growth
        score = engine.calculate_hype_score(
            view_velocity=50.0,
            edit_growth=20.0,
            content_expansion=10.0
        )
        
        # Should be low
        assert score < 0.3
    
    def test_calculate_hype_score_zero_growth(self):
        """Test hype score calculation with zero growth"""
        engine = HypeDetectionEngine()
        
        score = engine.calculate_hype_score(
            view_velocity=0.0,
            edit_growth=0.0,
            content_expansion=0.0
        )
        
        assert score == 0.0
    
    def test_calculate_hype_score_negative_growth(self):
        """Test hype score calculation with negative growth (decline)"""
        engine = HypeDetectionEngine()
        
        # Negative growth should still produce valid score (uses absolute value)
        score = engine.calculate_hype_score(
            view_velocity=-200.0,
            edit_growth=-100.0,
            content_expansion=-50.0
        )
        
        assert 0 <= score <= 1
    
    def test_calculate_hype_score_extreme_values(self):
        """Test hype score calculation with extreme values"""
        engine = HypeDetectionEngine()
        
        # Extreme values should be capped at 1.0
        score = engine.calculate_hype_score(
            view_velocity=5000.0,  # Way above cap
            edit_growth=2000.0,
            content_expansion=1000.0
        )
        
        assert score == 1.0
    
    def test_calculate_hype_score_formula_weights(self):
        """Test that hype score uses correct formula weights"""
        engine = HypeDetectionEngine()
        
        # Test with values at normalization caps
        # Formula: 0.5 * view_velocity/1000 + 0.3 * edit_growth/500 + 0.2 * content_expansion/300
        score = engine.calculate_hype_score(
            view_velocity=1000.0,  # 1.0 after normalization
            edit_growth=500.0,     # 1.0 after normalization
            content_expansion=300.0  # 1.0 after normalization
        )
        
        # Expected: 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 1.0 = 1.0
        assert score == pytest.approx(1.0, rel=0.01)
    
    def test_calculate_attention_density_normal(self):
        """Test attention density calculation with normal pageviews"""
        engine = HypeDetectionEngine()
        
        pageviews = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7),
            'views': [1000, 1200, 1100, 1300, 1400, 1500, 1600]
        })
        
        density = engine.calculate_attention_density(pageviews, window_days=7)
        
        # Total views = 9100, window = 7 days
        expected = 9100 / 7
        assert density == pytest.approx(expected, rel=0.01)
    
    def test_calculate_attention_density_single_day(self):
        """Test attention density calculation with single day"""
        engine = HypeDetectionEngine()
        
        pageviews = pd.DataFrame({
            'date': [datetime(2024, 1, 1)],
            'views': [5000]
        })
        
        density = engine.calculate_attention_density(pageviews, window_days=1)
        
        assert density == 5000.0
    
    def test_calculate_attention_density_empty(self):
        """Test attention density calculation with empty dataframe"""
        engine = HypeDetectionEngine()
        
        pageviews = pd.DataFrame({'date': [], 'views': []})
        
        density = engine.calculate_attention_density(pageviews, window_days=7)
        
        assert density == 0.0
    
    def test_calculate_attention_density_missing_column(self):
        """Test attention density calculation with missing column"""
        engine = HypeDetectionEngine()
        
        pageviews = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7)
        })
        
        with pytest.raises(ValueError, match="must have 'views' column"):
            engine.calculate_attention_density(pageviews, window_days=7)
    
    def test_detect_attention_spikes_with_spike(self):
        """Test spike detection with clear spike pattern"""
        engine = HypeDetectionEngine()
        
        # Create data with a clear spike
        pageviews = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20),
            'views': [100, 110, 105, 115, 120, 500, 520, 510, 125, 130,
                     135, 140, 145, 150, 155, 160, 165, 170, 175, 180]
        })
        
        spikes = engine.detect_attention_spikes(pageviews)
        
        # Should detect at least one spike around days 5-7
        assert len(spikes) > 0
        assert all(isinstance(spike, SpikeEvent) for spike in spikes)
    
    def test_detect_attention_spikes_no_spike(self):
        """Test spike detection with steady growth (no spikes)"""
        engine = HypeDetectionEngine()
        
        # Steady linear growth - no spikes
        pageviews = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20),
            'views': list(range(100, 120))
        })
        
        spikes = engine.detect_attention_spikes(pageviews)
        
        # Should detect no spikes or very few
        assert len(spikes) <= 1
    
    def test_detect_attention_spikes_continuous_spike(self):
        """Test spike detection with continuous high values"""
        engine = HypeDetectionEngine()
        
        # Start low, then jump to high and stay high
        pageviews = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20),
            'views': [100, 110, 105, 115, 120] + [500] * 15
        })
        
        spikes = engine.detect_attention_spikes(pageviews)
        
        # Should detect at least one spike
        assert len(spikes) > 0
        # Verify all spikes have valid properties
        for spike in spikes:
            assert spike.duration_days >= 1
            assert spike.spike_type in ["sustained", "temporary"]
    
    def test_detect_attention_spikes_empty(self):
        """Test spike detection with empty dataframe"""
        engine = HypeDetectionEngine()
        
        pageviews = pd.DataFrame({'date': [], 'views': []})
        
        spikes = engine.detect_attention_spikes(pageviews)
        
        assert spikes == []
    
    def test_detect_attention_spikes_too_few_points(self):
        """Test spike detection with too few data points"""
        engine = HypeDetectionEngine()
        
        pageviews = pd.DataFrame({
            'date': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'views': [100, 200]
        })
        
        spikes = engine.detect_attention_spikes(pageviews)
        
        # Should handle gracefully (return empty or minimal spikes)
        assert isinstance(spikes, list)
    
    def test_detect_attention_spikes_missing_columns(self):
        """Test spike detection with missing columns"""
        engine = HypeDetectionEngine()
        
        pageviews = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10)
        })
        
        with pytest.raises(ValueError, match="must have 'date' and 'views' columns"):
            engine.detect_attention_spikes(pageviews)
    
    def test_distinguish_spike_types_sustained(self):
        """Test spike classification for sustained growth"""
        engine = HypeDetectionEngine()
        
        spike = SpikeEvent(
            timestamp=datetime(2024, 1, 1),
            magnitude=3.5,
            duration_days=10,
            spike_type="temporary"  # Will be reclassified
        )
        
        classified = engine.distinguish_spike_types(spike)
        
        assert classified.spike_type == "sustained"
        assert classified.duration_days == 10
        assert classified.magnitude == 3.5
    
    def test_distinguish_spike_types_temporary(self):
        """Test spike classification for temporary spike"""
        engine = HypeDetectionEngine()
        
        spike = SpikeEvent(
            timestamp=datetime(2024, 1, 1),
            magnitude=2.5,
            duration_days=5,
            spike_type="sustained"  # Will be reclassified
        )
        
        classified = engine.distinguish_spike_types(spike)
        
        assert classified.spike_type == "temporary"
        assert classified.duration_days == 5
    
    def test_distinguish_spike_types_boundary(self):
        """Test spike classification at 7-day boundary"""
        engine = HypeDetectionEngine()
        
        # Exactly 7 days should be temporary
        spike_7 = SpikeEvent(
            timestamp=datetime(2024, 1, 1),
            magnitude=3.0,
            duration_days=7,
            spike_type="sustained"
        )
        
        classified_7 = engine.distinguish_spike_types(spike_7)
        assert classified_7.spike_type == "temporary"
        
        # 8 days should be sustained
        spike_8 = SpikeEvent(
            timestamp=datetime(2024, 1, 1),
            magnitude=3.0,
            duration_days=8,
            spike_type="temporary"
        )
        
        classified_8 = engine.distinguish_spike_types(spike_8)
        assert classified_8.spike_type == "sustained"
    
    def test_calculate_hype_metrics_complete(self):
        """Test complete hype metrics calculation"""
        engine = HypeDetectionEngine(hype_threshold=0.75)
        
        pageviews = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=14),
            'views': [1000, 1100, 1200, 1300, 1400, 1500, 1600,
                     1700, 1800, 1900, 2000, 2100, 2200, 2300]
        })
        
        metrics = engine.calculate_hype_metrics(
            article="Test_Article",
            pageviews=pageviews,
            view_velocity=500.0,
            edit_growth=200.0,
            content_expansion=100.0,
            window_days=7
        )
        
        assert metrics.article == "Test_Article"
        assert 0 <= metrics.hype_score <= 1
        assert metrics.view_velocity == 500.0
        assert metrics.edit_growth == 200.0
        assert metrics.content_expansion == 100.0
        assert metrics.attention_density > 0
        assert isinstance(metrics.is_trending, bool)
        assert isinstance(metrics.spike_events, list)
    
    def test_calculate_hype_metrics_trending(self):
        """Test hype metrics with trending article"""
        engine = HypeDetectionEngine(hype_threshold=0.5)
        
        pageviews = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7),
            'views': [1000] * 7
        })
        
        # High growth should trigger trending
        metrics = engine.calculate_hype_metrics(
            article="Trending_Article",
            pageviews=pageviews,
            view_velocity=800.0,
            edit_growth=400.0,
            content_expansion=200.0,
            window_days=7
        )
        
        assert metrics.is_trending is True
        assert metrics.hype_score >= 0.5
    
    def test_calculate_hype_metrics_not_trending(self):
        """Test hype metrics with non-trending article"""
        engine = HypeDetectionEngine(hype_threshold=0.75)
        
        pageviews = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7),
            'views': [1000] * 7
        })
        
        # Low growth should not trigger trending
        metrics = engine.calculate_hype_metrics(
            article="Stable_Article",
            pageviews=pageviews,
            view_velocity=50.0,
            edit_growth=20.0,
            content_expansion=10.0,
            window_days=7
        )
        
        assert metrics.is_trending is False
        assert metrics.hype_score < 0.75
    
    def test_hype_score_view_velocity_dominance(self):
        """Test that view velocity has highest weight in hype score"""
        engine = HypeDetectionEngine()
        
        # High view velocity, low others
        score_high_views = engine.calculate_hype_score(
            view_velocity=800.0,
            edit_growth=0.0,
            content_expansion=0.0
        )
        
        # Low view velocity, high others
        score_low_views = engine.calculate_hype_score(
            view_velocity=0.0,
            edit_growth=400.0,
            content_expansion=200.0
        )
        
        # View velocity should have more impact (50% weight vs 30% + 20%)
        assert score_high_views > score_low_views
    
    def test_spike_detection_with_flat_data(self):
        """Test spike detection with completely flat data"""
        engine = HypeDetectionEngine()
        
        pageviews = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20),
            'views': [1000] * 20
        })
        
        spikes = engine.detect_attention_spikes(pageviews)
        
        # Flat data should have no spikes
        assert len(spikes) == 0
    
    def test_attention_density_scales_with_window(self):
        """Test that attention density scales inversely with window size"""
        engine = HypeDetectionEngine()
        
        pageviews = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=14),
            'views': [1000] * 14
        })
        
        density_7 = engine.calculate_attention_density(pageviews, window_days=7)
        density_14 = engine.calculate_attention_density(pageviews, window_days=14)
        
        # Density should be inversely proportional to window size
        assert density_7 == pytest.approx(density_14 * 2, rel=0.01)
    
    def test_spike_events_have_valid_properties(self):
        """Test that all detected spike events have valid properties"""
        engine = HypeDetectionEngine()
        
        # Create data with multiple spikes
        views = [100] * 5 + [500] * 3 + [100] * 5 + [600] * 4 + [100] * 3
        pageviews = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=len(views)),
            'views': views
        })
        
        spikes = engine.detect_attention_spikes(pageviews)
        
        for spike in spikes:
            assert spike.magnitude >= 0
            assert spike.duration_days >= 1
            assert spike.spike_type in ["sustained", "temporary"]
            assert isinstance(spike.timestamp, datetime)
