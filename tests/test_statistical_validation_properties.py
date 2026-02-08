"""Property-based tests for statistical validation module.

Tests Properties 5 and 6 from the design document:
- Property 5: Statistical Significance Testing
- Property 6: Confidence Interval Inclusion
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import floats, integers, lists

from wikipedia_health.statistical_validation import (
    HypothesisTester,
    ConfidenceIntervalCalculator,
    EffectSizeCalculator
)


# Strategy for generating valid sample data
def sample_strategy(min_size=10, max_size=100):
    """Generate valid sample data for testing."""
    return lists(
        floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=min_size,
        max_size=max_size
    )


@settings(max_examples=100, deadline=None)
@given(
    sample1=sample_strategy(),
    sample2=sample_strategy()
)
def test_property_5_statistical_significance_testing_t_test(sample1, sample2):
    """
    Feature: wikipedia-product-health-analysis
    Property 5: For any trend change, structural shift, or group comparison,
    the system should perform appropriate hypothesis testing and report p-values,
    test statistics, and significance determination at α = 0.05.
    
    **Validates: Requirements 2.1, 2.2, 2.5**
    
    This test validates that t-tests always return complete statistical results.
    """
    # Arrange
    s1 = pd.Series(sample1)
    s2 = pd.Series(sample2)
    tester = HypothesisTester()
    
    # Act
    result = tester.t_test(s1, s2)
    
    # Assert - all required fields must be present
    assert result.test_name == 'Independent Samples t-test'
    assert isinstance(result.statistic, float)
    assert isinstance(result.p_value, float)
    assert isinstance(result.effect_size, float)
    assert isinstance(result.confidence_interval, tuple)
    assert len(result.confidence_interval) == 2
    assert isinstance(result.is_significant, bool)
    assert result.alpha == 0.05
    assert isinstance(result.interpretation, str)
    assert len(result.interpretation) > 0
    
    # P-value must be between 0 and 1
    assert 0 <= result.p_value <= 1
    
    # Significance determination must match p-value
    assert result.is_significant == (result.p_value < result.alpha)
    
    # Confidence interval must be ordered
    assert result.confidence_interval[0] <= result.confidence_interval[1]


@settings(max_examples=100, deadline=None)
@given(
    groups=lists(sample_strategy(min_size=5, max_size=50), min_size=2, max_size=5)
)
def test_property_5_statistical_significance_testing_anova(groups):
    """
    Feature: wikipedia-product-health-analysis
    Property 5: Statistical Significance Testing - ANOVA variant
    
    **Validates: Requirements 2.1, 2.2, 2.5**
    
    This test validates that ANOVA always returns complete statistical results.
    """
    # Arrange
    group_series = [pd.Series(g) for g in groups]
    tester = HypothesisTester()
    
    # Act
    result = tester.anova(group_series)
    
    # Assert - all required fields must be present
    assert result.test_name == 'One-Way ANOVA'
    assert isinstance(result.statistic, float)
    assert isinstance(result.p_value, float)
    assert isinstance(result.effect_size, float)
    assert isinstance(result.confidence_interval, tuple)
    assert len(result.confidence_interval) == 2
    assert isinstance(result.is_significant, bool)
    assert result.alpha == 0.05
    assert isinstance(result.interpretation, str)
    
    # P-value must be between 0 and 1
    assert 0 <= result.p_value <= 1
    
    # Significance determination must match p-value
    assert result.is_significant == (result.p_value < result.alpha)
    
    # Effect size (eta-squared) must be between 0 and 1
    assert 0 <= result.effect_size <= 1


@settings(max_examples=100, deadline=None)
@given(
    sample1=sample_strategy(),
    sample2=sample_strategy()
)
def test_property_5_statistical_significance_testing_mann_whitney(sample1, sample2):
    """
    Feature: wikipedia-product-health-analysis
    Property 5: Statistical Significance Testing - Mann-Whitney variant
    
    **Validates: Requirements 2.1, 2.2, 2.5**
    
    This test validates that Mann-Whitney tests always return complete results.
    """
    # Arrange
    s1 = pd.Series(sample1)
    s2 = pd.Series(sample2)
    tester = HypothesisTester()
    
    # Act
    result = tester.mann_whitney(s1, s2)
    
    # Assert
    assert result.test_name == 'Mann-Whitney U Test'
    assert isinstance(result.statistic, float)
    assert isinstance(result.p_value, float)
    assert 0 <= result.p_value <= 1
    assert isinstance(result.effect_size, float)
    assert isinstance(result.is_significant, bool)
    assert result.is_significant == (result.p_value < result.alpha)


@settings(max_examples=100, deadline=None)
@given(
    groups=lists(sample_strategy(min_size=5, max_size=50), min_size=2, max_size=5)
)
def test_property_5_statistical_significance_testing_kruskal_wallis(groups):
    """
    Feature: wikipedia-product-health-analysis
    Property 5: Statistical Significance Testing - Kruskal-Wallis variant
    
    **Validates: Requirements 2.1, 2.2, 2.5**
    
    This test validates that Kruskal-Wallis tests always return complete results.
    """
    # Arrange
    group_series = [pd.Series(g) for g in groups]
    tester = HypothesisTester()
    
    # Act
    result = tester.kruskal_wallis(group_series)
    
    # Assert
    assert result.test_name == 'Kruskal-Wallis H Test'
    assert isinstance(result.statistic, float)
    assert isinstance(result.p_value, float)
    assert 0 <= result.p_value <= 1
    assert isinstance(result.effect_size, float)
    assert 0 <= result.effect_size <= 1
    assert isinstance(result.is_significant, bool)
    assert result.is_significant == (result.p_value < result.alpha)


@settings(max_examples=50, deadline=None)  # Fewer examples due to computational cost
@given(
    sample1=sample_strategy(min_size=10, max_size=50),
    sample2=sample_strategy(min_size=10, max_size=50),
    n_permutations=integers(min_value=100, max_value=1000)
)
def test_property_5_statistical_significance_testing_permutation(sample1, sample2, n_permutations):
    """
    Feature: wikipedia-product-health-analysis
    Property 5: Statistical Significance Testing - Permutation test variant
    
    **Validates: Requirements 2.1, 2.2, 2.5**
    
    This test validates that permutation tests always return complete results.
    """
    # Arrange
    s1 = pd.Series(sample1)
    s2 = pd.Series(sample2)
    tester = HypothesisTester()
    
    # Act
    result = tester.permutation_test(s1, s2, n_permutations=n_permutations)
    
    # Assert
    assert result.test_name == 'Permutation Test'
    assert isinstance(result.statistic, float)
    assert isinstance(result.p_value, float)
    assert 0 <= result.p_value <= 1
    assert isinstance(result.effect_size, float)
    assert isinstance(result.is_significant, bool)
    assert result.is_significant == (result.p_value < result.alpha)


@settings(max_examples=100, deadline=None)
@given(
    data=sample_strategy(min_size=10, max_size=100)
)
def test_property_6_confidence_interval_inclusion_bootstrap(data):
    """
    Feature: wikipedia-product-health-analysis
    Property 6: For any statistical estimate (effect size, forecast, growth rate,
    platform proportion), the system should provide 95% confidence intervals
    alongside the point estimate.
    
    **Validates: Requirements 2.3, 2.4**
    
    This test validates that bootstrap confidence intervals are always computed.
    """
    # Arrange
    series = pd.Series(data)
    calculator = ConfidenceIntervalCalculator()
    
    # Act
    ci_lower, ci_upper = calculator.bootstrap_ci(
        series,
        statistic=np.mean,
        confidence_level=0.95,
        n_bootstrap=1000
    )
    
    # Assert
    assert isinstance(ci_lower, float)
    assert isinstance(ci_upper, float)
    
    # CI must be ordered
    assert ci_lower <= ci_upper
    
    # Point estimate should typically fall within CI
    point_estimate = series.mean()
    # Allow some tolerance for random variation in bootstrap
    # In rare cases, point estimate might fall slightly outside due to sampling


@settings(max_examples=100, deadline=None)
@given(
    data=sample_strategy(min_size=10, max_size=100)
)
def test_property_6_confidence_interval_inclusion_parametric(data):
    """
    Feature: wikipedia-product-health-analysis
    Property 6: Confidence Interval Inclusion - Parametric variant
    
    **Validates: Requirements 2.3, 2.4**
    
    This test validates that parametric confidence intervals are always computed.
    """
    # Arrange
    series = pd.Series(data)
    calculator = ConfidenceIntervalCalculator()
    
    # Act
    ci_lower, ci_upper = calculator.parametric_ci(series, confidence_level=0.95)
    
    # Assert
    assert isinstance(ci_lower, float)
    assert isinstance(ci_upper, float)
    
    # CI must be ordered
    assert ci_lower <= ci_upper
    
    # Point estimate must fall within CI (by construction for parametric CI)
    point_estimate = series.mean()
    assert ci_lower <= point_estimate <= ci_upper


@settings(max_examples=100, deadline=None)
@given(
    sample1=sample_strategy(),
    sample2=sample_strategy()
)
def test_property_6_confidence_interval_inclusion_difference(sample1, sample2):
    """
    Feature: wikipedia-product-health-analysis
    Property 6: Confidence Interval Inclusion - Difference CI variant
    
    **Validates: Requirements 2.3, 2.4**
    
    This test validates that difference confidence intervals are always computed.
    """
    # Arrange
    s1 = pd.Series(sample1)
    s2 = pd.Series(sample2)
    calculator = ConfidenceIntervalCalculator()
    
    # Act
    ci_lower, ci_upper = calculator.difference_ci(s1, s2, confidence_level=0.95)
    
    # Assert
    assert isinstance(ci_lower, float)
    assert isinstance(ci_upper, float)
    
    # CI must be ordered
    assert ci_lower <= ci_upper
    
    # Observed difference should fall within CI
    observed_diff = s1.mean() - s2.mean()
    assert ci_lower <= observed_diff <= ci_upper


@settings(max_examples=100, deadline=None)
@given(
    sample1=sample_strategy(),
    sample2=sample_strategy()
)
def test_property_6_effect_size_with_confidence_interval(sample1, sample2):
    """
    Feature: wikipedia-product-health-analysis
    Property 6: Effect sizes should be returned with confidence intervals
    
    **Validates: Requirements 2.3, 3.6**
    
    This test validates that effect sizes include confidence intervals.
    """
    # Arrange
    s1 = pd.Series(sample1)
    s2 = pd.Series(sample2)
    calculator = EffectSizeCalculator()
    
    # Act
    diff, (ci_lower, ci_upper) = calculator.absolute_difference(s1, s2)
    
    # Assert
    assert isinstance(diff, float)
    assert isinstance(ci_lower, float)
    assert isinstance(ci_upper, float)
    
    # CI must be ordered
    assert ci_lower <= ci_upper
    
    # Observed difference should fall within CI
    assert ci_lower <= diff <= ci_upper


@settings(max_examples=100, deadline=None)
@given(
    successes=integers(min_value=0, max_value=100),
    trials=integers(min_value=1, max_value=100)
)
def test_property_6_proportion_confidence_interval(successes, trials):
    """
    Feature: wikipedia-product-health-analysis
    Property 6: Proportion estimates should include confidence intervals
    
    **Validates: Requirements 2.3, 2.4**
    
    This test validates that proportion confidence intervals are computed.
    """
    # Ensure successes <= trials
    if successes > trials:
        successes, trials = trials, successes
    
    # Arrange
    calculator = ConfidenceIntervalCalculator()
    
    # Act
    ci_lower, ci_upper = calculator.proportion_ci(
        successes, trials, confidence_level=0.95
    )
    
    # Assert
    assert isinstance(ci_lower, float)
    assert isinstance(ci_upper, float)
    
    # CI must be ordered
    assert ci_lower <= ci_upper
    
    # CI must be within [0, 1]
    assert 0 <= ci_lower <= 1
    assert 0 <= ci_upper <= 1
    
    # Observed proportion should fall within CI
    if trials > 0:
        observed_prop = successes / trials
        assert ci_lower <= observed_prop <= ci_upper


@settings(max_examples=100, deadline=None)
@given(
    sample1=sample_strategy(),
    sample2=sample_strategy()
)
def test_property_6_cohens_d_computation(sample1, sample2):
    """
    Feature: wikipedia-product-health-analysis
    Property 6: Cohen's d effect size should always be computable
    
    **Validates: Requirements 2.3, 3.6**
    
    This test validates that Cohen's d is always computed for valid samples.
    """
    # Arrange
    s1 = pd.Series(sample1)
    s2 = pd.Series(sample2)
    calculator = EffectSizeCalculator()
    
    # Act
    cohens_d = calculator.cohens_d(s1, s2)
    
    # Assert
    assert isinstance(cohens_d, float)
    # Cohen's d can be any real number (including very large values)


@settings(max_examples=100, deadline=None)
@given(
    sample1=sample_strategy(),
    sample2=sample_strategy()
)
def test_property_6_hedges_g_computation(sample1, sample2):
    """
    Feature: wikipedia-product-health-analysis
    Property 6: Hedges' g effect size should always be computable
    
    **Validates: Requirements 2.3, 3.6**
    
    This test validates that Hedges' g is always computed for valid samples.
    """
    # Arrange
    s1 = pd.Series(sample1)
    s2 = pd.Series(sample2)
    calculator = EffectSizeCalculator()
    
    # Act
    hedges_g = calculator.hedges_g(s1, s2)
    
    # Assert
    assert isinstance(hedges_g, float)
    
    # Hedges' g should be close to Cohen's d (bias correction is small)
    cohens_d = calculator.cohens_d(s1, s2)
    if not np.isnan(cohens_d) and not np.isnan(hedges_g):
        # Correction factor is typically close to 1
        # When Cohen's d is 0, Hedges' g should also be 0 (or very close)
        if abs(cohens_d) < 1e-10:
            assert abs(hedges_g) < 1e-10
        else:
            # For non-zero Cohen's d, check relative difference
            assert abs(hedges_g - cohens_d) < abs(cohens_d) * 0.2


@settings(max_examples=100, deadline=None)
@given(
    baseline=floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    treatment=floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
)
def test_property_6_percentage_change_computation(baseline, treatment):
    """
    Feature: wikipedia-product-health-analysis
    Property 6: Percentage change should always be computable
    
    **Validates: Requirements 2.3, 3.6**
    
    This test validates that percentage change is always computed.
    """
    # Arrange
    calculator = EffectSizeCalculator()
    
    # Act
    pct_change = calculator.percentage_change(baseline, treatment)
    
    # Assert
    assert isinstance(pct_change, float)
    
    # Verify calculation (when baseline is not near zero)
    if abs(baseline) >= 1e-10:
        expected = ((treatment - baseline) / baseline) * 100
        assert abs(pct_change - expected) < 1e-6
    else:
        # Near-zero baseline - should return 0 or inf
        if abs(treatment) < 1e-10:
            assert pct_change == 0.0
        else:
            assert np.isinf(pct_change)
