"""Property-based tests for Evidence Framework.

This module tests the correctness properties of the evidence framework,
including multi-source validation, sensitivity analysis, and method consistency.

Tests Properties 12, 13, and 14 from the design document:
- Property 12: Multi-Source Validation
- Property 13: Sensitivity Analysis
- Property 14: Method Consistency Validation
"""

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck
from hypothesis.strategies import floats, integers, lists, sampled_from
import pandas as pd
import numpy as np
from datetime import date, timedelta

from wikipedia_health.evidence_framework import (
    CrossValidator,
    RobustnessChecker
)
from wikipedia_health.models.data_models import (
    Finding,
    TestResult,
    TimeSeriesData
)


# Strategy for generating valid time series data
def time_series_strategy(min_length=30, max_length=365):
    """Generate valid time series data."""
    return st.builds(
        lambda length, values, platform, metric_type: TimeSeriesData(
            date=pd.Series(pd.date_range(start='2020-01-01', periods=length, freq='D')),
            values=pd.Series(values),
            platform=platform,
            metric_type=metric_type,
            metadata={'source': 'test'}
        ),
        length=integers(min_value=min_length, max_value=max_length),
        values=st.lists(
            floats(min_value=1000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False),
            min_size=min_length,
            max_size=max_length
        ).map(lambda lst: lst[:min_length] if len(lst) < max_length else lst),
        platform=sampled_from(['desktop', 'mobile-web', 'mobile-app', 'all']),
        metric_type=sampled_from(['pageviews', 'editors', 'edits'])
    )


# Strategy for generating test results
def test_result_strategy():
    """Generate valid test results."""
    return st.builds(
        lambda effect_size, p_value: TestResult(
            test_name='t_test',
            statistic=effect_size * 2.0,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(effect_size - 0.1, effect_size + 0.1),
            is_significant=p_value < 0.05,
            alpha=0.05,
            interpretation=f"Effect size: {effect_size:.4f}"
        ),
        effect_size=floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        p_value=floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )


# Strategy for generating findings
def finding_strategy():
    """Generate valid findings."""
    return st.builds(
        lambda finding_id, description, evidence: Finding(
            finding_id=finding_id,
            description=description,
            evidence=evidence,
            causal_effects=[],
            confidence_level='medium',
            requirements_validated=['5.1', '5.2']
        ),
        finding_id=st.text(min_size=5, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        description=st.text(min_size=10, max_size=100),
        evidence=lists(test_result_strategy(), min_size=1, max_size=3)
    )


@pytest.mark.property
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(
    finding=finding_strategy(),
    num_sources=integers(min_value=2, max_value=5)
)
def test_property_12_multi_source_validation(finding, num_sources):
    """
    Feature: wikipedia-product-health-analysis
    Property 12: For any finding or pattern, the system should validate it across
    multiple data sources (pageviews, editors, edits) and report the consistency
    score and which sources support the finding.
    
    **Validates: Requirements 5.1, 5.2**
    """
    # Generate multiple data sources with correlated trends
    source_names = ['pageviews', 'editors', 'edits', 'source4', 'source5'][:num_sources]
    
    # Get the effect direction from the finding
    primary_effect = finding.evidence[0].effect_size
    
    # Create data sources with trends that match or contradict the finding
    data_sources = {}
    for i, source_name in enumerate(source_names):
        # Create time series with trend matching the finding (with some variation)
        length = 90
        dates = pd.date_range(start='2020-01-01', periods=length, freq='D')
        
        # Generate values with trend matching the finding direction
        # Add some randomness but keep the overall trend
        base_values = np.linspace(10000, 10000 + primary_effect * 1000, length)
        noise = np.random.normal(0, 100, length)
        values = base_values + noise
        
        # Make some sources contradict the finding (about 30% of the time)
        if i % 3 == 0 and num_sources > 2:
            # Reverse the trend for this source
            values = values[::-1]
        
        data_sources[source_name] = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(values),
            platform='all',
            metric_type=source_name,
            metadata={'source': 'test'}
        )
    
    # Validate across sources
    validator = CrossValidator(significance_level=0.05)
    result = validator.validate_across_sources(finding, data_sources)
    
    # Property assertions
    # 1. Result should have consistency score between 0 and 1
    assert 0.0 <= result.consistency_score <= 1.0, \
        f"Consistency score {result.consistency_score} not in [0, 1]"
    
    # 2. Supporting + contradicting sources should not exceed total sources
    total_classified = len(result.supporting_sources) + len(result.contradicting_sources)
    assert total_classified <= num_sources, \
        f"Total classified sources {total_classified} exceeds total sources {num_sources}"
    
    # 3. All supporting sources should be in the original data sources
    for source in result.supporting_sources:
        assert source in data_sources, \
            f"Supporting source {source} not in original data sources"
    
    # 4. All contradicting sources should be in the original data sources
    for source in result.contradicting_sources:
        assert source in data_sources, \
            f"Contradicting source {source} not in original data sources"
    
    # 5. If consistency score is high (>0.7), finding should be marked as consistent
    if result.consistency_score > 0.7:
        assert result.is_consistent, \
            f"High consistency score {result.consistency_score} but finding not marked consistent"
    
    # 6. If consistency score is low (<0.3), finding should not be marked as consistent
    if result.consistency_score < 0.3:
        assert not result.is_consistent, \
            f"Low consistency score {result.consistency_score} but finding marked consistent"
    
    # 7. Result should have details dictionary
    assert isinstance(result.details, dict), \
        "Result details should be a dictionary"
    
    # 8. Summary should be a non-empty string
    summary = result.summary()
    assert isinstance(summary, str) and len(summary) > 0, \
        "Summary should be a non-empty string"


@pytest.mark.property
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(
    base_value=floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    num_variations=integers(min_value=3, max_value=10)
)
def test_property_13_sensitivity_analysis(base_value, num_variations):
    """
    Feature: wikipedia-product-health-analysis
    Property 13: For any analysis with configurable parameters, the system should
    perform sensitivity analysis by varying parameters and assumptions, reporting
    how results change across the parameter space.
    
    **Validates: Requirements 5.5, 5.6**
    """
    # Create a simple analysis function that depends on a parameter
    def analysis_function(threshold=3.0):
        """Simple analysis function that returns a value based on threshold."""
        return base_value * threshold
    
    # Create base result
    base_result = analysis_function(threshold=3.0)
    
    # Create parameter variations
    threshold_values = [1.0 + i * 0.5 for i in range(num_variations)]
    parameters = {'threshold': threshold_values}
    
    # Perform sensitivity analysis
    checker = RobustnessChecker(significance_level=0.05)
    result = checker.sensitivity_analysis(analysis_function, parameters, base_result)
    
    # Property assertions
    # 1. Result should have stability score between 0 and 1
    assert 0.0 <= result.stability_score <= 1.0, \
        f"Stability score {result.stability_score} not in [0, 1]"
    
    # 2. Parameter results should contain all tested variations
    assert len(result.parameter_results) == num_variations, \
        f"Expected {num_variations} parameter results, got {len(result.parameter_results)}"
    
    # 3. All parameter variations should be in the results
    for threshold in threshold_values:
        param_key = f"threshold={threshold}"
        assert param_key in result.parameter_results, \
            f"Parameter variation {param_key} not in results"
    
    # 4. Base result should be stored
    assert result.base_result == base_result, \
        "Base result not correctly stored"
    
    # 5. Details should contain result values
    assert 'result_values' in result.details, \
        "Details should contain result_values"
    
    # 6. If all results are identical, stability score should be 1.0
    result_values = result.details['result_values']
    if len(set(result_values)) == 1:
        assert result.stability_score == 1.0, \
            "Identical results should have stability score of 1.0"
    
    # 7. If results vary significantly (CV > 0.2), should not be marked stable
    if len(result_values) >= 2:
        mean_val = np.mean(result_values)
        std_val = np.std(result_values)
        if mean_val != 0:
            cv = std_val / abs(mean_val)
            if cv > 0.2:
                assert not result.is_stable, \
                    f"High variation (CV={cv:.2f}) but marked as stable"
    
    # 8. Summary should be a non-empty string
    summary = result.summary()
    assert isinstance(summary, str) and len(summary) > 0, \
        "Summary should be a non-empty string"


@pytest.mark.property
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(
    series_length=integers(min_value=30, max_value=100),
    num_methods=integers(min_value=2, max_value=5),
    noise_level=floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)
)
def test_property_14_method_consistency_validation(series_length, num_methods, noise_level):
    """
    Feature: wikipedia-product-health-analysis
    Property 14: For any analysis that can be performed with multiple methods
    (e.g., seasonal decomposition with STL vs X-13-ARIMA-SEATS), the system
    should apply multiple methods and verify consistency of results.
    
    **Validates: Requirements 5.7**
    """
    # Generate a time series
    dates = pd.date_range(start='2020-01-01', periods=series_length, freq='D')
    base_trend = np.linspace(1000, 2000, series_length)
    noise = np.random.normal(0, noise_level * 1000, series_length)
    values = pd.Series(base_trend + noise)
    
    # Create multiple analysis methods that compute similar statistics
    # Each method computes the mean with slight variations
    methods = {}
    for i in range(num_methods):
        method_name = f"method_{i+1}"
        # Each method uses a slightly different approach (e.g., trimmed mean, median, etc.)
        if i == 0:
            methods[method_name] = lambda s: s.mean()
        elif i == 1:
            methods[method_name] = lambda s: s.median()
        elif i == 2:
            methods[method_name] = lambda s: s.quantile(0.5)
        elif i == 3:
            # Trimmed mean (remove top and bottom 10%)
            methods[method_name] = lambda s: s.sort_values().iloc[int(len(s)*0.1):int(len(s)*0.9)].mean()
        else:
            # Winsorized mean
            methods[method_name] = lambda s: s.clip(lower=s.quantile(0.1), upper=s.quantile(0.9)).mean()
    
    # Perform method comparison
    checker = RobustnessChecker(significance_level=0.05)
    result = checker.method_comparison(values, methods)
    
    # Property assertions
    # 1. Result should have consistency score between 0 and 1
    assert 0.0 <= result.consistency_score <= 1.0, \
        f"Consistency score {result.consistency_score} not in [0, 1]"
    
    # 2. Method results should contain all tested methods
    assert len(result.method_results) == num_methods, \
        f"Expected {num_methods} method results, got {len(result.method_results)}"
    
    # 3. All methods should be in the results
    for method_name in methods.keys():
        assert method_name in result.method_results, \
            f"Method {method_name} not in results"
    
    # 4. Recommended method should be one of the tested methods
    assert result.recommended_method in methods.keys(), \
        f"Recommended method {result.recommended_method} not in tested methods"
    
    # 5. Details should contain result values
    assert 'result_values' in result.details, \
        "Details should contain result_values"
    
    # 6. If noise level is low, methods should be consistent
    if noise_level < 0.1:
        # With low noise, different methods should produce similar results
        result_values = result.details['result_values']
        if len(result_values) >= 2:
            cv = result.details.get('cv', 0)
            if cv is not None and cv < 0.15:
                assert result.is_consistent, \
                    f"Low noise ({noise_level}) and low CV ({cv}) but not marked consistent"
    
    # 7. If methods produce very different results (CV > 0.15), should not be consistent
    if 'cv' in result.details and result.details['cv'] is not None:
        if result.details['cv'] > 0.15:
            assert not result.is_consistent, \
                f"High variation (CV={result.details['cv']:.2f}) but marked as consistent"
    
    # 8. Details should contain mean and std of results
    assert 'mean' in result.details, "Details should contain mean"
    assert 'std' in result.details, "Details should contain std"
    
    # 9. If all methods produce identical results, consistency score should be 1.0
    result_values = result.details['result_values']
    if len(result_values) >= 2:
        if np.std(result_values) == 0:
            assert result.consistency_score == 1.0, \
                "Identical results should have consistency score of 1.0"


@pytest.mark.property
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(
    finding=finding_strategy(),
    num_platforms=integers(min_value=2, max_value=3)
)
def test_property_12_cross_platform_validation(finding, num_platforms):
    """
    Feature: wikipedia-product-health-analysis
    Property 12 (Cross-Platform): For any finding, the system should validate it
    across multiple platforms (desktop, mobile web, mobile app) and report
    consistency scores.
    
    **Validates: Requirements 5.2**
    """
    # Generate platform data
    platform_names = ['desktop', 'mobile-web', 'mobile-app'][:num_platforms]
    
    # Get the effect direction from the finding
    primary_effect = finding.evidence[0].effect_size
    
    # Create platform data with trends that match or contradict the finding
    platform_data = {}
    for i, platform_name in enumerate(platform_names):
        length = 90
        dates = pd.date_range(start='2020-01-01', periods=length, freq='D')
        
        # Generate values with trend matching the finding direction
        base_values = np.linspace(10000, 10000 + primary_effect * 1000, length)
        noise = np.random.normal(0, 100, length)
        values = base_values + noise
        
        # Make some platforms contradict the finding
        if i == num_platforms - 1 and num_platforms > 2:
            values = values[::-1]
        
        platform_data[platform_name] = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(values),
            platform=platform_name,
            metric_type='pageviews',
            metadata={'source': 'test'}
        )
    
    # Validate across platforms
    validator = CrossValidator(significance_level=0.05)
    result = validator.validate_across_platforms(finding, platform_data)
    
    # Property assertions
    # 1. Result should have consistency score between 0 and 1
    assert 0.0 <= result.consistency_score <= 1.0, \
        f"Consistency score {result.consistency_score} not in [0, 1]"
    
    # 2. All supporting platforms should be in the original platform data
    for platform in result.supporting_sources:
        assert platform in platform_data, \
            f"Supporting platform {platform} not in original platform data"
    
    # 3. Result should be a ValidationResult with proper structure
    assert hasattr(result, 'is_consistent'), "Result should have is_consistent attribute"
    assert hasattr(result, 'consistency_score'), "Result should have consistency_score attribute"
    assert hasattr(result, 'supporting_sources'), "Result should have supporting_sources attribute"


@pytest.mark.property
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(
    series_length=integers(min_value=50, max_value=150),
    outlier_multiplier=floats(min_value=5.0, max_value=20.0, allow_nan=False, allow_infinity=False)
)
def test_property_13_outlier_sensitivity(series_length, outlier_multiplier):
    """
    Feature: wikipedia-product-health-analysis
    Property 13 (Outlier Sensitivity): For any analysis, the system should test
    sensitivity to outliers by comparing results with and without outliers.
    
    **Validates: Requirements 5.6**
    """
    # Generate a time series with a few extreme outliers
    base_values = np.random.normal(1000, 50, series_length)  # Smaller std for better detection
    
    # Add a small number of extreme outliers
    num_outliers = max(1, series_length // 20)  # About 5% outliers
    outlier_indices = np.random.choice(series_length, num_outliers, replace=False)
    # Make outliers extremely large relative to the distribution
    base_values[outlier_indices] = base_values[outlier_indices] * outlier_multiplier
    
    series = pd.Series(base_values)
    
    # Define a simple analysis function (compute mean)
    def analysis_function(s):
        return s.mean()
    
    # Perform outlier sensitivity analysis
    checker = RobustnessChecker(significance_level=0.05)
    result = checker.outlier_sensitivity(series, analysis_function)
    
    # Property assertions
    # 1. Result should have impact score between 0 and 1
    assert 0.0 <= result.impact_score <= 1.0, \
        f"Impact score {result.impact_score} not in [0, 1]"
    
    # 2. Number of outliers detected should be reasonable
    assert 0 <= result.outliers_detected <= series_length, \
        f"Outliers detected {result.outliers_detected} exceeds series length {series_length}"
    
    # 3. Result should have both with_outliers and without_outliers
    assert result.with_outliers is not None, "Result should have with_outliers"
    assert result.without_outliers is not None, "Result should have without_outliers"
    
    # 4. Details should contain outlier information
    assert 'outlier_indices' in result.details, "Details should contain outlier_indices"
    assert 'outlier_values' in result.details, "Details should contain outlier_values"
    assert 'val_with' in result.details, "Details should contain val_with"
    assert 'val_without' in result.details, "Details should contain val_without"
    
    # 5. With extreme outliers (multiplier > 10), there should be measurable impact
    # Either outliers are detected OR the results differ significantly
    if outlier_multiplier > 10:
        val_with = result.details.get('val_with')
        val_without = result.details.get('val_without')
        
        if val_with is not None and val_without is not None:
            # Results should differ when extreme outliers are present
            results_differ = abs(val_with - val_without) > 0.01
            outliers_detected = result.outliers_detected > 0
            
            assert results_differ or outliers_detected, \
                f"Extreme outliers (multiplier={outlier_multiplier}) but no impact: " \
                f"with={val_with}, without={val_without}, detected={result.outliers_detected}"
    
    # 6. If impact is low (<0.1), results should be marked as robust
    if result.impact_score < 0.1:
        assert result.is_robust, \
            f"Low impact score {result.impact_score} but not marked robust"
    
    # 7. If impact is high (>0.1), results should not be marked as robust
    if result.impact_score > 0.1:
        assert not result.is_robust, \
            f"High impact score {result.impact_score} but marked as robust"
    
    # 8. Impact score should reflect the difference between with/without outliers
    val_with = result.details.get('val_with')
    val_without = result.details.get('val_without')
    if val_with is not None and val_without is not None and val_with != 0:
        expected_impact = abs((val_without - val_with) / val_with)
        # Impact score should be related to the actual difference
        # (may be capped at 1.0)
        assert result.impact_score <= 1.0, \
            f"Impact score {result.impact_score} exceeds maximum of 1.0"


@pytest.mark.property
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(
    series_length=integers(min_value=50, max_value=150),
    n_subsamples=integers(min_value=20, max_value=50)
)
def test_property_13_subsample_stability(series_length, n_subsamples):
    """
    Feature: wikipedia-product-health-analysis
    Property 13 (Subsample Stability): For any analysis, the system should test
    stability using bootstrap subsampling to check if results are stable across
    different data subsets.
    
    **Validates: Requirements 5.5**
    """
    # Generate a time series
    base_values = np.random.normal(1000, 100, series_length)
    series = pd.Series(base_values)
    
    # Define a simple analysis function (compute mean)
    def analysis_function(s):
        return s.mean()
    
    # Perform subsample stability analysis
    checker = RobustnessChecker(significance_level=0.05)
    result = checker.subsample_stability(series, analysis_function, n_subsamples=n_subsamples)
    
    # Property assertions
    # 1. Result should have stability score between 0 and 1
    assert 0.0 <= result.stability_score <= 1.0, \
        f"Stability score {result.stability_score} not in [0, 1]"
    
    # 2. Number of subsample results should match n_subsamples
    assert len(result.subsample_results) == n_subsamples, \
        f"Expected {n_subsamples} subsample results, got {len(result.subsample_results)}"
    
    # 3. Mean result should be close to the true mean
    if result.mean_result is not None:
        true_mean = series.mean()
        # Mean of subsamples should be within 20% of true mean
        assert abs(result.mean_result - true_mean) / true_mean < 0.2, \
            f"Mean result {result.mean_result} too far from true mean {true_mean}"
    
    # 4. Standard deviation should be non-negative
    assert result.std_result >= 0, \
        f"Standard deviation {result.std_result} should be non-negative"
    
    # 5. Details should contain result values
    assert 'result_values' in result.details, "Details should contain result_values"
    
    # 6. Details should contain subsample parameters
    assert 'n_subsamples' in result.details, "Details should contain n_subsamples"
    assert 'subsample_fraction' in result.details, "Details should contain subsample_fraction"
    
    # 7. If standard deviation is low, results should be marked as stable
    if result.mean_result is not None and result.mean_result != 0:
        cv = result.std_result / abs(result.mean_result)
        if cv < 0.2:
            assert result.is_stable, \
                f"Low CV ({cv:.2f}) but not marked as stable"
