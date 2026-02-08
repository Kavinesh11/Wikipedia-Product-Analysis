"""Unit tests for causal inference module.

Tests specific scenarios and edge cases for:
- Interrupted Time Series Analysis (ITSA)
- Difference-in-Differences (DiD)
- Event Study
- Synthetic Control
"""

from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest
from pandas import Series

from wikipedia_health.causal_inference import (
    InterruptedTimeSeriesAnalyzer,
    DifferenceInDifferencesAnalyzer,
    EventStudyAnalyzer,
    SyntheticControlBuilder
)


def generate_simple_series(start_date, n_days, base_value=1000, trend=0, noise_std=10, seed=42):
    """Generate simple time series for testing."""
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    time_index = np.arange(n_days)
    values = base_value + trend * time_index + np.random.normal(0, noise_std, n_days)
    return Series(values, index=dates, name='metric')


def generate_intervention_series(start_date, n_days, intervention_day, effect_size=100, seed=42):
    """Generate time series with intervention effect."""
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    time_index = np.arange(n_days)
    values = 1000 + time_index * 0.5  # Slight upward trend
    # Add intervention effect
    values[intervention_day:] += effect_size
    values += np.random.normal(0, 20, n_days)
    return Series(values, index=dates, name='metric')


class TestInterruptedTimeSeriesAnalyzer:
    """Unit tests for ITSA."""
    
    def test_fit_with_simulated_intervention(self):
        """Test ITSA with simulated intervention data."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        
        # Generate data with intervention effect
        series = generate_intervention_series(start_date, 180, 90, effect_size=150)
        
        itsa = InterruptedTimeSeriesAnalyzer()
        model = itsa.fit(series, intervention_date, pre_period_length=90)
        
        assert model is not None
        assert model.pre_model is not None
        assert model.post_model is not None
        assert model.intervention_date == intervention_date
    
    def test_estimate_effect_detects_positive_effect(self):
        """Test that ITSA detects a positive intervention effect."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        
        # Generate data with positive effect
        series = generate_intervention_series(start_date, 180, 90, effect_size=200)
        
        itsa = InterruptedTimeSeriesAnalyzer()
        model = itsa.fit(series, intervention_date, pre_period_length=90)
        effect = itsa.estimate_effect(model, post_period_length=90)
        
        # Should detect positive effect
        assert effect.effect_size > 0
        assert effect.p_value < 0.05  # Should be significant
        assert effect.confidence_interval[0] < effect.effect_size < effect.confidence_interval[1]
    
    def test_estimate_effect_no_effect(self):
        """Test ITSA with no intervention effect."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        
        # Generate data with no effect
        series = generate_simple_series(start_date, 180, base_value=1000, trend=0.5)
        
        itsa = InterruptedTimeSeriesAnalyzer()
        model = itsa.fit(series, intervention_date, pre_period_length=90)
        effect = itsa.estimate_effect(model, post_period_length=90)
        
        # Effect should be small and not significant
        assert abs(effect.effect_size) < 100  # Small effect
        # p_value may or may not be significant due to noise
    
    def test_construct_counterfactual(self):
        """Test counterfactual construction."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        
        series = generate_intervention_series(start_date, 180, 90, effect_size=150)
        
        itsa = InterruptedTimeSeriesAnalyzer()
        model = itsa.fit(series, intervention_date, pre_period_length=90)
        
        end_date = date(2020, 6, 29)
        counterfactual = itsa.construct_counterfactual(model, (intervention_date, end_date))
        
        # Length should be approximately 90 (may be 89 or 90 depending on date range)
        assert 89 <= len(counterfactual) <= 90
        assert counterfactual.index[0] == pd.Timestamp(intervention_date)
        assert all(counterfactual > 0)
    
    def test_parallel_trends_test(self):
        """Test parallel trends assumption test."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        
        # Generate data with stable pre-trend
        series = generate_simple_series(start_date, 180, base_value=1000, trend=1.0)
        
        itsa = InterruptedTimeSeriesAnalyzer()
        model = itsa.fit(series, intervention_date, pre_period_length=90)
        
        test_result = itsa.test_parallel_trends(model)
        
        assert test_result is not None
        assert test_result.test_name == 'Parallel Trends Test (ITSA)'
        assert 0 <= test_result.p_value <= 1
    
    def test_insufficient_pre_period_raises_error(self):
        """Test that insufficient pre-period raises error."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 2, 1)
        
        series = generate_simple_series(start_date, 60, base_value=1000)
        
        itsa = InterruptedTimeSeriesAnalyzer(min_pre_period=60)
        
        with pytest.raises(ValueError, match="Pre-intervention period must be at least"):
            itsa.fit(series, intervention_date, pre_period_length=30)


class TestDifferenceInDifferencesAnalyzer:
    """Unit tests for DiD."""
    
    def test_fit_with_parallel_trends(self):
        """Test DiD with parallel pre-trends."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        
        # Generate treatment and control with parallel trends
        treatment = generate_simple_series(start_date, 180, base_value=1000, trend=1.0, seed=42)
        control = generate_simple_series(start_date, 180, base_value=800, trend=1.0, seed=43)
        
        # Add intervention effect to treatment
        treatment.iloc[90:] += 150
        
        did = DifferenceInDifferencesAnalyzer()
        model = did.fit(treatment, control, intervention_date)
        
        assert model is not None
        assert model.intervention_date == intervention_date
    
    def test_estimate_effect_detects_treatment(self):
        """Test that DiD detects treatment effect."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        
        # Generate treatment and control
        treatment = generate_simple_series(start_date, 180, base_value=1000, trend=1.0, seed=42)
        control = generate_simple_series(start_date, 180, base_value=800, trend=1.0, seed=43)
        
        # Add intervention effect to treatment
        treatment.iloc[90:] += 200
        
        did = DifferenceInDifferencesAnalyzer()
        model = did.fit(treatment, control, intervention_date)
        effect = did.estimate_effect(model)
        
        # Should detect positive effect
        assert effect.effect_size > 100  # Should be close to 200
        assert effect.p_value < 0.05  # Should be significant
    
    def test_estimate_effect_no_treatment(self):
        """Test DiD with no treatment effect."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        
        # Generate treatment and control with parallel trends, no intervention
        treatment = generate_simple_series(start_date, 180, base_value=1000, trend=1.0, seed=42)
        control = generate_simple_series(start_date, 180, base_value=800, trend=1.0, seed=43)
        
        did = DifferenceInDifferencesAnalyzer()
        model = did.fit(treatment, control, intervention_date)
        effect = did.estimate_effect(model)
        
        # Effect should be small
        assert abs(effect.effect_size) < 100
    
    def test_parallel_trends_test_satisfied(self):
        """Test parallel trends test when assumption is satisfied."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        
        # Generate with parallel trends
        treatment = generate_simple_series(start_date, 180, base_value=1000, trend=1.0, seed=42)
        control = generate_simple_series(start_date, 180, base_value=800, trend=1.0, seed=43)
        
        did = DifferenceInDifferencesAnalyzer()
        model = did.fit(treatment, control, intervention_date)
        
        test_result = did.test_parallel_trends(model, (start_date, intervention_date))
        
        assert test_result is not None
        assert test_result.test_name == 'Parallel Trends Test (DiD)'
        # With parallel trends, p-value should be high (not significant)
        assert test_result.p_value > 0.05
    
    def test_parallel_trends_test_violated(self):
        """Test parallel trends test when assumption is violated."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        
        # Generate with different trends
        treatment = generate_simple_series(start_date, 180, base_value=1000, trend=2.0, seed=42)
        control = generate_simple_series(start_date, 180, base_value=800, trend=0.5, seed=43)
        
        did = DifferenceInDifferencesAnalyzer()
        model = did.fit(treatment, control, intervention_date)
        
        test_result = did.test_parallel_trends(model, (start_date, intervention_date))
        
        # With different trends, p-value should be low (significant)
        assert test_result.p_value < 0.05
        assert test_result.is_significant
    
    def test_placebo_test(self):
        """Test placebo test for robustness."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        placebo_date = date(2020, 2, 15)
        
        # Generate with parallel trends
        treatment = generate_simple_series(start_date, 180, base_value=1000, trend=1.0, seed=42)
        control = generate_simple_series(start_date, 180, base_value=800, trend=1.0, seed=43)
        
        did = DifferenceInDifferencesAnalyzer()
        model = did.fit(treatment, control, intervention_date)
        
        placebo_result = did.placebo_test(model, placebo_date)
        
        assert placebo_result is not None
        assert placebo_result.test_name == 'Placebo Test (DiD)'
        # Placebo should not be significant
        assert placebo_result.p_value > 0.05


class TestEventStudyAnalyzer:
    """Unit tests for Event Study."""
    
    def test_fit_baseline_arima(self):
        """Test baseline fitting with ARIMA."""
        start_date = date(2020, 1, 1)
        event_date = date(2020, 4, 1)
        
        series = generate_simple_series(start_date, 180, base_value=1000, trend=1.0)
        
        event_study = EventStudyAnalyzer()
        baseline = event_study.fit_baseline(series, event_date, baseline_window=90, method='arima')
        
        assert baseline is not None
        assert baseline.model is not None
        assert baseline.model_type == 'arima'
        assert baseline.event_date == event_date
    
    def test_fit_baseline_prophet(self):
        """Test baseline fitting with Prophet."""
        start_date = date(2020, 1, 1)
        event_date = date(2020, 4, 1)
        
        series = generate_simple_series(start_date, 180, base_value=1000, trend=1.0)
        
        event_study = EventStudyAnalyzer()
        baseline = event_study.fit_baseline(series, event_date, baseline_window=90, method='prophet')
        
        assert baseline is not None
        assert baseline.model is not None
        assert baseline.model_type == 'prophet'
    
    def test_estimate_event_impact(self):
        """Test event impact estimation."""
        start_date = date(2020, 1, 1)
        event_date = date(2020, 4, 1)
        
        # Generate data with event effect
        series = generate_intervention_series(start_date, 180, 90, effect_size=250)
        
        event_study = EventStudyAnalyzer()
        baseline = event_study.fit_baseline(series, event_date, baseline_window=90, method='arima')
        impact = event_study.estimate_event_impact(series, baseline, event_date, post_window=30)
        
        assert impact is not None
        assert impact.car is not None  # Cumulative abnormal return
        assert impact.observed is not None
        assert impact.predicted is not None
        assert impact.difference is not None
        assert len(impact.observed) == len(impact.predicted)
    
    def test_test_significance(self):
        """Test significance testing."""
        start_date = date(2020, 1, 1)
        event_date = date(2020, 4, 1)
        
        # Generate data with significant event effect
        series = generate_intervention_series(start_date, 180, 90, effect_size=300)
        
        event_study = EventStudyAnalyzer()
        baseline = event_study.fit_baseline(series, event_date, baseline_window=90, method='arima')
        impact = event_study.estimate_event_impact(series, baseline, event_date, post_window=30)
        
        is_significant, p_value = event_study.test_significance(impact)
        
        assert isinstance(is_significant, (bool, np.bool_))
        assert 0 <= p_value <= 1
        # With large effect, should be significant
        assert is_significant
        assert p_value < 0.05
    
    def test_measure_persistence(self):
        """Test persistence measurement."""
        start_date = date(2020, 1, 1)
        event_date = date(2020, 4, 1)
        
        # Generate data with temporary spike
        series = generate_simple_series(start_date, 180, base_value=1000, trend=0.5)
        # Add temporary spike
        series.iloc[90:100] += 200
        
        event_study = EventStudyAnalyzer()
        half_life = event_study.measure_persistence(series, event_date, max_window=90)
        
        assert half_life is not None
        assert isinstance(half_life, int)
        assert 1 <= half_life <= 90
    
    def test_analyze_event_complete(self):
        """Test complete event analysis."""
        start_date = date(2020, 1, 1)
        event_date = date(2020, 4, 1)
        
        series = generate_intervention_series(start_date, 180, 90, effect_size=200)
        
        event_study = EventStudyAnalyzer()
        effect = event_study.analyze_event(
            series, event_date, baseline_window=90, post_window=30, method='arima'
        )
        
        assert effect is not None
        assert effect.method == 'EventStudy'
        assert effect.effect_size is not None
        assert effect.confidence_interval is not None
        assert effect.p_value is not None


class TestSyntheticControlBuilder:
    """Unit tests for Synthetic Control."""
    
    def test_construct_synthetic_control_with_known_weights(self):
        """Test synthetic control construction."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        
        # Generate treated unit
        treated = generate_simple_series(start_date, 180, base_value=1000, trend=1.0, seed=42)
        
        # Generate donor pool
        donors = [
            generate_simple_series(start_date, 180, base_value=900, trend=1.0, seed=43),
            generate_simple_series(start_date, 180, base_value=1100, trend=1.0, seed=44),
            generate_simple_series(start_date, 180, base_value=950, trend=1.0, seed=45),
        ]
        
        sc_builder = SyntheticControlBuilder(min_r_squared=0.5)
        
        pre_end = intervention_date - timedelta(days=1)
        synthetic = sc_builder.construct_synthetic_control(
            treated, donors, (start_date, pre_end)
        )
        
        assert synthetic is not None
        assert len(synthetic.weights) == 3
        assert abs(np.sum(synthetic.weights) - 1.0) < 1e-6  # Weights sum to 1
        assert all(w >= 0 for w in synthetic.weights)  # Non-negative weights
        assert synthetic.r_squared >= 0.5
    
    def test_estimate_effect_with_synthetic_control(self):
        """Test effect estimation with synthetic control."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        
        # Generate treated unit with intervention effect
        treated = generate_simple_series(start_date, 180, base_value=1000, trend=1.0, seed=42)
        treated.iloc[90:] += 200  # Add intervention effect
        
        # Generate donor pool
        donors = [
            generate_simple_series(start_date, 180, base_value=900, trend=1.0, seed=43),
            generate_simple_series(start_date, 180, base_value=1100, trend=1.0, seed=44),
            generate_simple_series(start_date, 180, base_value=950, trend=1.0, seed=45),
        ]
        
        sc_builder = SyntheticControlBuilder(min_r_squared=0.5)
        
        pre_end = intervention_date - timedelta(days=1)
        synthetic = sc_builder.construct_synthetic_control(
            treated, donors, (start_date, pre_end)
        )
        
        post_end = date(2020, 6, 29)
        effect = sc_builder.estimate_effect(
            treated, synthetic, donors, (intervention_date, post_end)
        )
        
        assert effect is not None
        assert effect.effect_size > 100  # Should detect positive effect
        assert effect.confidence_interval is not None
        assert effect.p_value is not None
    
    def test_placebo_test(self):
        """Test placebo test on donor units."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        
        # Generate donor pool (all with similar trends)
        donors = [
            generate_simple_series(start_date, 180, base_value=900 + i*50, trend=1.0, seed=42+i)
            for i in range(5)
        ]
        
        sc_builder = SyntheticControlBuilder(min_r_squared=0.5)
        
        pre_end = intervention_date - timedelta(days=1)
        post_end = date(2020, 6, 29)
        
        placebo_results = sc_builder.placebo_test(
            donors, intervention_date, (start_date, pre_end), (intervention_date, post_end)
        )
        
        assert len(placebo_results) > 0
        for result in placebo_results:
            assert result.unit_name is not None
            assert result.effect_size is not None
            assert result.pre_period_fit is not None
    
    def test_inference_from_placebo_distribution(self):
        """Test inference using placebo distribution."""
        start_date = date(2020, 1, 1)
        intervention_date = date(2020, 4, 1)
        
        # Generate treated unit with large effect
        treated = generate_simple_series(start_date, 180, base_value=1000, trend=1.0, seed=42)
        treated.iloc[90:] += 300  # Large intervention effect
        
        # Generate donor pool
        donors = [
            generate_simple_series(start_date, 180, base_value=900 + i*50, trend=1.0, seed=43+i)
            for i in range(5)
        ]
        
        sc_builder = SyntheticControlBuilder(min_r_squared=0.3)  # Lower threshold for test data
        
        pre_end = intervention_date - timedelta(days=1)
        synthetic = sc_builder.construct_synthetic_control(
            treated, donors, (start_date, pre_end)
        )
        
        post_end = date(2020, 6, 29)
        effect = sc_builder.estimate_effect(
            treated, synthetic, donors, (intervention_date, post_end)
        )
        
        placebo_results = sc_builder.placebo_test(
            donors, intervention_date, (start_date, pre_end), (intervention_date, post_end)
        )
        
        inference = sc_builder.inference(effect, placebo_results)
        
        assert inference is not None
        assert inference.test_name == 'Synthetic Control Inference (Placebo Test)'
        assert 0 <= inference.p_value <= 1
        # With synthetic data, significance depends on placebo distribution
        # Just verify the inference was computed correctly
        assert isinstance(inference.is_significant, (bool, np.bool_))
