"""Unit tests for statistical validation module.

Tests specific examples with known distributions and analytical solutions.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats

from wikipedia_health.statistical_validation import (
    HypothesisTester,
    ConfidenceIntervalCalculator,
    EffectSizeCalculator
)


class TestHypothesisTester:
    """Unit tests for HypothesisTester class."""
    
    def test_t_test_with_known_difference(self):
        """Test t-test with samples that have a known mean difference."""
        # Arrange
        np.random.seed(42)
        sample1 = pd.Series(np.random.normal(10, 2, 100))
        sample2 = pd.Series(np.random.normal(12, 2, 100))
        tester = HypothesisTester()
        
        # Act
        result = tester.t_test(sample1, sample2)
        
        # Assert
        assert result.test_name == 'Independent Samples t-test'
        assert result.p_value < 0.05  # Should be significant
        assert result.is_significant is True
        assert result.effect_size < 0  # sample1 mean < sample2 mean
        assert result.confidence_interval[0] < result.confidence_interval[1]
    
    def test_t_test_with_identical_samples(self):
        """Test t-test with identical samples (no difference)."""
        # Arrange
        np.random.seed(42)
        sample = pd.Series(np.random.normal(10, 2, 100))
        tester = HypothesisTester()
        
        # Act
        result = tester.t_test(sample, sample)
        
        # Assert
        assert result.p_value > 0.05  # Should not be significant
        assert result.is_significant is False
        assert abs(result.effect_size) < 0.01  # Effect size should be near zero
    
    def test_anova_with_three_groups(self):
        """Test ANOVA with three groups having different means."""
        # Arrange
        np.random.seed(42)
        group1 = pd.Series(np.random.normal(10, 2, 50))
        group2 = pd.Series(np.random.normal(12, 2, 50))
        group3 = pd.Series(np.random.normal(14, 2, 50))
        tester = HypothesisTester()
        
        # Act
        result = tester.anova([group1, group2, group3])
        
        # Assert
        assert result.test_name == 'One-Way ANOVA'
        assert result.p_value < 0.05  # Should be significant
        assert result.is_significant is True
        assert 0 <= result.effect_size <= 1  # Eta-squared is bounded
    
    def test_anova_with_identical_groups(self):
        """Test ANOVA with groups having the same distribution."""
        # Arrange
        np.random.seed(42)
        group1 = pd.Series(np.random.normal(10, 2, 50))
        group2 = pd.Series(np.random.normal(10, 2, 50))
        group3 = pd.Series(np.random.normal(10, 2, 50))
        tester = HypothesisTester()
        
        # Act
        result = tester.anova([group1, group2, group3])
        
        # Assert
        assert result.p_value > 0.05  # Should not be significant
        assert result.is_significant is False
    
    def test_mann_whitney_with_different_distributions(self):
        """Test Mann-Whitney with samples from different distributions."""
        # Arrange
        np.random.seed(42)
        sample1 = pd.Series(np.random.exponential(2, 100))
        sample2 = pd.Series(np.random.exponential(3, 100))
        tester = HypothesisTester()
        
        # Act
        result = tester.mann_whitney(sample1, sample2)
        
        # Assert
        assert result.test_name == 'Mann-Whitney U Test'
        assert result.p_value < 0.05  # Should be significant
        assert result.is_significant is True
    
    def test_kruskal_wallis_with_different_groups(self):
        """Test Kruskal-Wallis with groups from different distributions."""
        # Arrange
        np.random.seed(42)
        group1 = pd.Series(np.random.exponential(2, 50))
        group2 = pd.Series(np.random.exponential(3, 50))
        group3 = pd.Series(np.random.exponential(4, 50))
        tester = HypothesisTester()
        
        # Act
        result = tester.kruskal_wallis([group1, group2, group3])
        
        # Assert
        assert result.test_name == 'Kruskal-Wallis H Test'
        assert result.p_value < 0.05  # Should be significant
        assert result.is_significant is True
    
    def test_permutation_test_with_known_difference(self):
        """Test permutation test with samples having a known difference."""
        # Arrange
        np.random.seed(42)
        sample1 = pd.Series(np.random.normal(10, 2, 50))
        sample2 = pd.Series(np.random.normal(12, 2, 50))
        tester = HypothesisTester()
        
        # Act
        result = tester.permutation_test(sample1, sample2, n_permutations=1000)
        
        # Assert
        assert result.test_name == 'Permutation Test'
        assert result.p_value < 0.05  # Should be significant
        assert result.is_significant is True
        assert result.statistic < 0  # sample1 mean < sample2 mean


class TestConfidenceIntervalCalculator:
    """Unit tests for ConfidenceIntervalCalculator class."""
    
    def test_bootstrap_ci_for_mean(self):
        """Test bootstrap CI for mean with known distribution."""
        # Arrange
        np.random.seed(42)
        data = pd.Series(np.random.normal(10, 2, 100))
        calculator = ConfidenceIntervalCalculator()
        
        # Act
        ci_lower, ci_upper = calculator.bootstrap_ci(
            data, statistic=np.mean, n_bootstrap=1000
        )
        
        # Assert
        assert ci_lower < data.mean() < ci_upper
        assert ci_upper - ci_lower > 0  # CI has positive width
    
    def test_bootstrap_ci_for_median(self):
        """Test bootstrap CI for median."""
        # Arrange
        np.random.seed(42)
        data = pd.Series(np.random.normal(10, 2, 100))
        calculator = ConfidenceIntervalCalculator()
        
        # Act
        ci_lower, ci_upper = calculator.bootstrap_ci(
            data, statistic=np.median, n_bootstrap=1000
        )
        
        # Assert
        assert ci_lower < data.median() < ci_upper
    
    def test_parametric_ci_matches_analytical_solution(self):
        """Test parametric CI matches analytical t-distribution solution."""
        # Arrange
        np.random.seed(42)
        data = pd.Series(np.random.normal(10, 2, 100))
        calculator = ConfidenceIntervalCalculator()
        
        # Act
        ci_lower, ci_upper = calculator.parametric_ci(data, confidence_level=0.95)
        
        # Assert - compare with scipy's implementation
        mean = data.mean()
        se = data.sem()
        t_critical = stats.t.ppf(0.975, len(data) - 1)
        expected_lower = mean - t_critical * se
        expected_upper = mean + t_critical * se
        
        assert abs(ci_lower - expected_lower) < 1e-6
        assert abs(ci_upper - expected_upper) < 1e-6
    
    def test_difference_ci_parametric(self):
        """Test confidence interval for difference between means."""
        # Arrange
        np.random.seed(42)
        sample1 = pd.Series(np.random.normal(10, 2, 50))
        sample2 = pd.Series(np.random.normal(12, 2, 50))
        calculator = ConfidenceIntervalCalculator()
        
        # Act
        ci_lower, ci_upper = calculator.difference_ci(
            sample1, sample2, method='parametric'
        )
        
        # Assert
        observed_diff = sample1.mean() - sample2.mean()
        assert ci_lower < observed_diff < ci_upper
        assert ci_lower < 0 < ci_upper or ci_lower < 0 and ci_upper < 0
    
    def test_proportion_ci_wilson_method(self):
        """Test Wilson score interval for proportions."""
        # Arrange
        calculator = ConfidenceIntervalCalculator()
        
        # Act - 50 successes out of 100 trials
        ci_lower, ci_upper = calculator.proportion_ci(
            successes=50, trials=100, method='wilson'
        )
        
        # Assert
        assert 0 < ci_lower < 0.5 < ci_upper < 1
        assert ci_upper - ci_lower > 0
    
    def test_proportion_ci_extreme_cases(self):
        """Test proportion CI with extreme cases (0% and 100%)."""
        # Arrange
        calculator = ConfidenceIntervalCalculator()
        
        # Act - 0 successes
        ci_lower_0, ci_upper_0 = calculator.proportion_ci(
            successes=0, trials=100, method='wilson'
        )
        
        # Act - 100 successes
        ci_lower_100, ci_upper_100 = calculator.proportion_ci(
            successes=100, trials=100, method='wilson'
        )
        
        # Assert
        assert ci_lower_0 >= 0
        assert ci_upper_0 < 0.1  # Should be small
        assert ci_lower_100 > 0.9  # Should be large
        assert ci_upper_100 <= 1


class TestEffectSizeCalculator:
    """Unit tests for EffectSizeCalculator class."""
    
    def test_cohens_d_with_known_effect(self):
        """Test Cohen's d with samples having a known effect size."""
        # Arrange
        np.random.seed(42)
        # Create samples with 1 SD difference (Cohen's d ≈ 1)
        sample1 = pd.Series(np.random.normal(10, 2, 100))
        sample2 = pd.Series(np.random.normal(12, 2, 100))
        calculator = EffectSizeCalculator()
        
        # Act
        cohens_d = calculator.cohens_d(sample1, sample2)
        
        # Assert
        assert -1.5 < cohens_d < -0.5  # Should be around -1
    
    def test_cohens_d_zero_effect(self):
        """Test Cohen's d with identical samples (zero effect)."""
        # Arrange
        np.random.seed(42)
        sample = pd.Series(np.random.normal(10, 2, 100))
        calculator = EffectSizeCalculator()
        
        # Act
        cohens_d = calculator.cohens_d(sample, sample)
        
        # Assert
        assert abs(cohens_d) < 0.01  # Should be near zero
    
    def test_hedges_g_bias_correction(self):
        """Test that Hedges' g applies bias correction."""
        # Arrange
        np.random.seed(42)
        sample1 = pd.Series(np.random.normal(10, 2, 20))  # Small sample
        sample2 = pd.Series(np.random.normal(12, 2, 20))
        calculator = EffectSizeCalculator()
        
        # Act
        cohens_d = calculator.cohens_d(sample1, sample2)
        hedges_g = calculator.hedges_g(sample1, sample2)
        
        # Assert
        # Hedges' g should be slightly smaller than Cohen's d (correction factor < 1)
        assert abs(hedges_g) < abs(cohens_d)
        assert abs(hedges_g - cohens_d) < 0.1  # Correction is small
    
    def test_percentage_change_calculation(self):
        """Test percentage change with known values."""
        # Arrange
        calculator = EffectSizeCalculator()
        
        # Act & Assert
        assert calculator.percentage_change(100, 110) == 10.0
        assert calculator.percentage_change(100, 90) == -10.0
        assert calculator.percentage_change(50, 100) == 100.0
        assert calculator.percentage_change(100, 50) == -50.0
    
    def test_absolute_difference_with_ci(self):
        """Test absolute difference calculation with confidence interval."""
        # Arrange
        np.random.seed(42)
        sample1 = pd.Series(np.random.normal(10, 2, 50))
        sample2 = pd.Series(np.random.normal(12, 2, 50))
        calculator = EffectSizeCalculator()
        
        # Act
        diff, (ci_lower, ci_upper) = calculator.absolute_difference(sample1, sample2)
        
        # Assert
        assert diff == sample1.mean() - sample2.mean()
        assert ci_lower < diff < ci_upper
        assert diff < 0  # sample1 mean < sample2 mean
    
    def test_relative_risk_calculation(self):
        """Test relative risk with known values."""
        # Arrange
        calculator = EffectSizeCalculator()
        
        # Act - 20% risk in exposed, 10% risk in control
        rr, (ci_lower, ci_upper) = calculator.relative_risk(
            exposed_events=20, exposed_total=100,
            control_events=10, control_total=100
        )
        
        # Assert
        assert abs(rr - 2.0) < 0.01  # RR should be 2.0
        assert ci_lower < rr < ci_upper
        assert rr > 1  # Increased risk
    
    def test_odds_ratio_calculation(self):
        """Test odds ratio with known values."""
        # Arrange
        calculator = EffectSizeCalculator()
        
        # Act - 20% vs 10% event rates
        or_value, (ci_lower, ci_upper) = calculator.odds_ratio(
            exposed_events=20, exposed_total=100,
            control_events=10, control_total=100
        )
        
        # Assert
        # OR = (20/80) / (10/90) = 0.25 / 0.111 ≈ 2.25
        assert 2.0 < or_value < 2.5
        assert ci_lower < or_value < ci_upper
    
    def test_correlation_effect_size(self):
        """Test correlation as effect size."""
        # Arrange
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = 0.7 * x + np.random.normal(0, 0.5, 100)  # Correlation ≈ 0.7
        sample1 = pd.Series(x)
        sample2 = pd.Series(y)
        calculator = EffectSizeCalculator()
        
        # Act
        r, (ci_lower, ci_upper) = calculator.correlation_effect_size(sample1, sample2)
        
        # Assert
        assert 0.5 < r < 0.9  # Should be around 0.7
        assert ci_lower < r < ci_upper


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_series(self):
        """Test handling of empty series."""
        # Arrange
        empty = pd.Series([])
        sample = pd.Series([1, 2, 3, 4, 5])
        tester = HypothesisTester()
        calculator = ConfidenceIntervalCalculator()
        effect_calc = EffectSizeCalculator()
        
        # Act & Assert - should handle gracefully
        result = tester.t_test(empty, sample)
        assert np.isnan(result.p_value) or result.p_value >= 0
        
        ci = calculator.parametric_ci(empty)
        assert np.isnan(ci[0]) and np.isnan(ci[1])
        
        cohens_d = effect_calc.cohens_d(empty, sample)
        assert np.isnan(cohens_d)
    
    def test_series_with_nan_values(self):
        """Test handling of NaN values in series."""
        # Arrange
        sample1 = pd.Series([1, 2, np.nan, 4, 5])
        sample2 = pd.Series([2, 3, 4, np.nan, 6])
        tester = HypothesisTester()
        
        # Act
        result = tester.t_test(sample1, sample2)
        
        # Assert - should drop NaN and compute on valid data
        assert not np.isnan(result.p_value)
        assert isinstance(result.is_significant, bool)
    
    def test_single_value_series(self):
        """Test handling of series with single value."""
        # Arrange
        single = pd.Series([5.0])
        sample = pd.Series([1, 2, 3, 4, 5])
        calculator = ConfidenceIntervalCalculator()
        
        # Act
        ci = calculator.parametric_ci(single)
        
        # Assert - CI should be defined but may be wide or degenerate
        assert isinstance(ci[0], float)
        assert isinstance(ci[1], float)
