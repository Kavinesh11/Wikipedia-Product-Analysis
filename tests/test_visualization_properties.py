"""Property-based tests for visualization and reporting.

Tests Properties 34, 35, and 36 from the design document.
"""

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import dates, lists, floats, text, sampled_from
import pandas as pd
from pandas import Series
import numpy as np
from datetime import date, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from wikipedia_health.models import (
    TimeSeriesData,
    TestResult,
    CausalEffect,
    ForecastResult,
    Finding,
)
from wikipedia_health.visualization import (
    plot_trend_with_confidence_bands,
    plot_campaign_effect,
    plot_forecast,
    plot_comparison,
    generate_summary_table,
    generate_finding_report,
    generate_evidence_report,
    create_interactive_trend_plot,
    create_interactive_campaign_plot,
    create_interactive_forecast_plot,
    create_interactive_comparison_plot,
    create_findings_dashboard,
)


# Strategy for generating valid test results
@st.composite
def generate_test_result(draw):
    """Generate valid TestResult objects."""
    test_name = draw(sampled_from([
        't-test', 'ANOVA', 'Mann-Whitney', 'Kruskal-Wallis', 'Permutation Test'
    ]))
    statistic = draw(floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
    p_value = draw(floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    effect_size = draw(floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    ci_lower = effect_size - abs(draw(floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False)))
    ci_upper = effect_size + abs(draw(floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False)))
    
    return TestResult(
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        confidence_interval=(ci_lower, ci_upper),
        is_significant=p_value < 0.05,
        alpha=0.05,
        interpretation=f"Test result for {test_name}"
    )


# Strategy for generating valid causal effects
@st.composite
def generate_causal_effect(draw, n_points=30):
    """Generate valid CausalEffect objects."""
    effect_size = draw(floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
    ci_lower = effect_size - abs(draw(floats(min_value=1, max_value=20, allow_nan=False, allow_infinity=False)))
    ci_upper = effect_size + abs(draw(floats(min_value=1, max_value=20, allow_nan=False, allow_infinity=False)))
    p_value = draw(floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    # Generate observed and counterfactual series
    base_values = draw(lists(
        floats(min_value=100, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=n_points,
        max_size=n_points
    ))
    
    observed = Series(base_values)
    counterfactual = Series([v - effect_size/n_points * i for i, v in enumerate(base_values)])
    
    start_date = date(2020, 1, 1)
    end_date = start_date + timedelta(days=n_points-1)
    
    return CausalEffect(
        effect_size=effect_size,
        confidence_interval=(ci_lower, ci_upper),
        p_value=p_value,
        method=draw(sampled_from(['ITSA', 'DiD', 'EventStudy', 'SyntheticControl'])),
        counterfactual=counterfactual,
        observed=observed,
        treatment_period=(start_date, end_date)
    )


# Strategy for generating valid forecast results
@st.composite
def generate_forecast_result(draw, horizon=30):
    """Generate valid ForecastResult objects."""
    point_forecast = draw(lists(
        floats(min_value=100, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=horizon,
        max_size=horizon
    ))
    
    # Generate bounds around point forecast
    lower_bound = [p - abs(draw(floats(min_value=10, max_value=50, allow_nan=False, allow_infinity=False))) 
                   for p in point_forecast]
    upper_bound = [p + abs(draw(floats(min_value=10, max_value=50, allow_nan=False, allow_infinity=False))) 
                   for p in point_forecast]
    
    return ForecastResult(
        point_forecast=Series(point_forecast),
        lower_bound=Series(lower_bound),
        upper_bound=Series(upper_bound),
        confidence_level=0.95,
        model_type=draw(sampled_from(['ARIMA', 'Prophet', 'ExponentialSmoothing'])),
        horizon=horizon
    )



@settings(max_examples=100, deadline=None)
@given(
    n_points=st.integers(min_value=30, max_value=100),
    has_ci=st.booleans()
)
def test_property_34_visualization_evidence_inclusion(n_points, has_ci):
    """
    Feature: wikipedia-product-health-analysis
    Property 34: For any visualization (trend plot, campaign effect plot, forecast plot,
    comparison plot), the system should include statistical evidence overlays: confidence
    bands/prediction intervals for trends and forecasts, p-values and effect sizes for
    comparisons, and significance indicators.
    
    Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5
    """
    # Generate test data
    series = Series(np.random.randn(n_points).cumsum() + 100)
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
    
    if has_ci:
        ci_lower = series - 10
        ci_upper = series + 10
        confidence_interval = (ci_lower, ci_upper)
    else:
        confidence_interval = None
    
    # Test 1: Trend plot should include confidence bands if provided
    fig = plot_trend_with_confidence_bands(
        series=series,
        dates=dates,
        confidence_interval=confidence_interval
    )
    
    assert fig is not None
    assert len(fig.axes) > 0
    
    # Check that confidence bands are plotted if provided
    if has_ci:
        # Should have at least 2 collections (line + fill)
        assert len(fig.axes[0].lines) >= 1  # Main line
        assert len(fig.axes[0].collections) >= 1 or len(fig.axes[0].patches) >= 1  # Filled area
    
    plt.close(fig)
    
    # Test 2: Campaign effect plot should include statistical annotations
    causal_effect = CausalEffect(
        effect_size=50.0,
        confidence_interval=(40.0, 60.0),
        p_value=0.001,
        method='ITSA',
        counterfactual=series * 0.9,
        observed=series,
        treatment_period=(dates[10], dates[20])
    )
    
    fig = plot_campaign_effect(causal_effect, dates=dates, show_annotations=True)
    
    assert fig is not None
    assert len(fig.axes) > 0
    # Should have text annotations with statistical info
    assert len(fig.axes[0].texts) > 0
    
    plt.close(fig)
    
    # Test 3: Forecast plot should include prediction intervals
    forecast_result = ForecastResult(
        point_forecast=Series(np.random.randn(30).cumsum() + 100),
        lower_bound=Series(np.random.randn(30).cumsum() + 90),
        upper_bound=Series(np.random.randn(30).cumsum() + 110),
        confidence_level=0.95,
        model_type='ARIMA',
        horizon=30
    )
    
    fig = plot_forecast(forecast_result, historical=series[:20], historical_dates=dates[:20])
    
    assert fig is not None
    assert len(fig.axes) > 0
    # Should have multiple lines (historical + forecast + bounds)
    assert len(fig.axes[0].lines) >= 2
    
    plt.close(fig)
    
    # Test 4: Comparison plot should include error bars and significance indicators
    groups = {
        'Group A': Series(np.random.randn(50) + 100),
        'Group B': Series(np.random.randn(50) + 105)
    }
    
    test_result = TestResult(
        test_name='t-test',
        statistic=2.5,
        p_value=0.01,
        effect_size=0.5,
        confidence_interval=(0.2, 0.8),
        is_significant=True,
        alpha=0.05,
        interpretation="Significant difference between groups"
    )
    
    fig = plot_comparison(groups, test_result=test_result, show_error_bars=True, show_significance=True)
    
    assert fig is not None
    assert len(fig.axes) > 0
    # Should have bars with error bars
    assert len(fig.axes[0].patches) >= 2  # At least 2 bars
    # Should have text annotations with p-values
    assert len(fig.axes[0].texts) > 0
    
    plt.close(fig)


@settings(max_examples=100, deadline=None)
@given(
    n_tests=st.integers(min_value=1, max_value=5),
    n_causal=st.integers(min_value=0, max_value=3)
)
def test_property_35_report_completeness(n_tests, n_causal):
    """
    Feature: wikipedia-product-health-analysis
    Property 35: For any analysis report, the system should include summary tables with
    test statistics, p-values, confidence intervals, effect sizes, and plain-language
    interpretations for all findings.
    
    Validates: Requirements 13.6
    """
    # Generate test results
    test_results = []
    for i in range(n_tests):
        test_results.append(TestResult(
            test_name=f'Test_{i}',
            statistic=np.random.randn(),
            p_value=np.random.uniform(0, 1),
            effect_size=np.random.randn(),
            confidence_interval=(np.random.randn() - 1, np.random.randn() + 1),
            is_significant=np.random.uniform(0, 1) < 0.5,
            alpha=0.05,
            interpretation=f"Interpretation for test {i}"
        ))
    
    # Generate causal effects
    causal_effects = []
    for i in range(n_causal):
        causal_effects.append(CausalEffect(
            effect_size=np.random.randn() * 10,
            confidence_interval=(np.random.randn() * 10 - 5, np.random.randn() * 10 + 5),
            p_value=np.random.uniform(0, 1),
            method='ITSA',
            counterfactual=Series(np.random.randn(30) + 100),
            observed=Series(np.random.randn(30) + 105),
            treatment_period=(date(2020, 1, 1), date(2020, 1, 30))
        ))
    
    # Test 1: Summary table should include all required columns
    summary_table = generate_summary_table(test_results, causal_effects)
    
    assert summary_table is not None
    assert len(summary_table) == n_tests + n_causal
    
    # Check required columns
    required_columns = ['Test', 'Statistic', 'p-value', 'Significant', 'Effect Size', 
                       '95% CI Lower', '95% CI Upper', 'Interpretation']
    for col in required_columns:
        assert col in summary_table.columns
    
    # Test 2: Finding report should include all sections
    finding = Finding(
        finding_id='TEST_001',
        description='Test finding description',
        evidence=test_results,
        causal_effects=causal_effects,
        confidence_level='high',
        requirements_validated=['13.6']
    )
    
    report = generate_finding_report(finding, include_evidence_details=True)
    
    assert report is not None
    assert len(report) > 0
    
    # Check that report includes key sections
    assert 'FINDING REPORT' in report
    assert 'DESCRIPTION' in report
    assert 'Confidence Level' in report
    
    if n_tests > 0:
        assert 'STATISTICAL EVIDENCE' in report
        assert 'P-value' in report or 'p-value' in report
    
    if n_causal > 0:
        assert 'CAUSAL ANALYSIS' in report
    
    assert 'RECOMMENDATIONS' in report
    
    # Test 3: Evidence report should aggregate all findings
    findings = [finding]
    evidence_report = generate_evidence_report(findings, include_cross_validation=True)
    
    assert evidence_report is not None
    assert len(evidence_report) > 0
    
    # Check that report includes aggregated statistics
    assert 'EXECUTIVE SUMMARY' in evidence_report
    assert 'Total Findings' in evidence_report
    assert 'Total Statistical Tests' in evidence_report
    assert 'FINDINGS SUMMARY' in evidence_report


@settings(max_examples=100, deadline=None)
@given(
    n_points=st.integers(min_value=30, max_value=100),
    has_ci=st.booleans()
)
def test_property_36_interactive_dashboard_elements(n_points, has_ci):
    """
    Feature: wikipedia-product-health-analysis
    Property 36: For any dashboard, the system should provide interactive elements
    (hover for confidence intervals, click for detailed test results, filter by time period)
    and methodology tooltips explaining each statistical measure.
    
    Validates: Requirements 13.7
    """
    # Generate test data
    series = Series(np.random.randn(n_points).cumsum() + 100)
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
    
    if has_ci:
        ci_lower = series - 10
        ci_upper = series + 10
        confidence_interval = (ci_lower, ci_upper)
    else:
        confidence_interval = None
    
    # Test 1: Interactive trend plot should have hover tooltips
    fig = create_interactive_trend_plot(
        series=series,
        dates=dates,
        confidence_interval=confidence_interval,
        show_methodology_tooltip=True
    )
    
    assert fig is not None
    assert len(fig.data) > 0
    
    # Check that traces have hover information
    for trace in fig.data:
        # Plotly traces should have hoverinfo or hovertext
        assert hasattr(trace, 'hoverinfo') or hasattr(trace, 'hovertext')
    
    # Check for range slider (time period filtering)
    assert fig.layout.xaxis.rangeslider is not None
    assert fig.layout.xaxis.rangeslider.visible == True
    
    # Check for methodology annotations
    if has_ci:
        assert len(fig.layout.annotations) > 0
    
    # Test 2: Interactive campaign plot should have detailed hover info
    causal_effect = CausalEffect(
        effect_size=50.0,
        confidence_interval=(40.0, 60.0),
        p_value=0.001,
        method='ITSA',
        counterfactual=series * 0.9,
        observed=series,
        treatment_period=(dates[10], dates[20])
    )
    
    fig = create_interactive_campaign_plot(causal_effect, dates=dates)
    
    assert fig is not None
    assert len(fig.data) > 0
    
    # Check for hover information on traces
    for trace in fig.data:
        assert hasattr(trace, 'hoverinfo') or hasattr(trace, 'hovertext')
    
    # Check for statistical annotations
    assert len(fig.layout.annotations) > 0
    
    # Check for range slider
    assert fig.layout.xaxis.rangeslider is not None
    
    # Test 3: Interactive forecast plot should have uncertainty hover info
    forecast_result = ForecastResult(
        point_forecast=Series(np.random.randn(30).cumsum() + 100),
        lower_bound=Series(np.random.randn(30).cumsum() + 90),
        upper_bound=Series(np.random.randn(30).cumsum() + 110),
        confidence_level=0.95,
        model_type='ARIMA',
        horizon=30
    )
    
    fig = create_interactive_forecast_plot(
        forecast_result,
        historical=series[:20],
        historical_dates=dates[:20]
    )
    
    assert fig is not None
    assert len(fig.data) > 0
    
    # Check for hover information
    for trace in fig.data:
        assert hasattr(trace, 'hoverinfo') or hasattr(trace, 'hovertext')
    
    # Check for methodology annotations
    assert len(fig.layout.annotations) > 0
    
    # Test 4: Interactive comparison plot should have detailed statistics
    groups = {
        'Group A': Series(np.random.randn(50) + 100),
        'Group B': Series(np.random.randn(50) + 105)
    }
    
    test_result = TestResult(
        test_name='t-test',
        statistic=2.5,
        p_value=0.01,
        effect_size=0.5,
        confidence_interval=(0.2, 0.8),
        is_significant=True,
        alpha=0.05,
        interpretation="Significant difference between groups"
    )
    
    fig = create_interactive_comparison_plot(groups, test_result=test_result)
    
    assert fig is not None
    assert len(fig.data) > 0
    
    # Check for hover information with detailed statistics
    for trace in fig.data:
        assert hasattr(trace, 'hoverinfo') or hasattr(trace, 'hovertext')
    
    # Check for statistical annotations
    assert len(fig.layout.annotations) > 0
    
    # Test 5: Findings dashboard should have multiple interactive components
    findings = [
        Finding(
            finding_id=f'FINDING_{i}',
            description=f'Test finding {i}',
            evidence=[TestResult(
                test_name='t-test',
                statistic=np.random.randn(),
                p_value=np.random.uniform(0, 1),
                effect_size=np.random.randn(),
                confidence_interval=(np.random.randn() - 1, np.random.randn() + 1),
                is_significant=np.random.uniform(0, 1) < 0.5,
                alpha=0.05,
                interpretation="Test interpretation"
            )],
            confidence_level=np.random.choice(['high', 'medium', 'low'])
        )
        for i in range(3)
    ]
    
    fig = create_findings_dashboard(findings)
    
    assert fig is not None
    # Dashboard should have multiple subplots
    assert len(fig.data) > 0
    
    # Check that all traces have hover information
    for trace in fig.data:
        assert hasattr(trace, 'hoverinfo') or hasattr(trace, 'hovertemplate')


# Additional helper tests for edge cases
@settings(max_examples=50, deadline=None)
@given(test_result=generate_test_result())
def test_visualization_with_various_test_results(test_result):
    """Test that visualizations handle various test result configurations."""
    groups = {
        'Group A': Series(np.random.randn(30) + 100),
        'Group B': Series(np.random.randn(30) + 105)
    }
    
    fig = plot_comparison(groups, test_result=test_result)
    assert fig is not None
    plt.close(fig)


@settings(max_examples=50, deadline=None)
@given(causal_effect=generate_causal_effect())
def test_visualization_with_various_causal_effects(causal_effect):
    """Test that visualizations handle various causal effect configurations."""
    dates = pd.date_range(start='2020-01-01', periods=len(causal_effect.observed), freq='D')
    
    fig = plot_campaign_effect(causal_effect, dates=dates)
    assert fig is not None
    plt.close(fig)


@settings(max_examples=50, deadline=None)
@given(forecast_result=generate_forecast_result())
def test_visualization_with_various_forecasts(forecast_result):
    """Test that visualizations handle various forecast configurations."""
    fig = plot_forecast(forecast_result)
    assert fig is not None
    plt.close(fig)
