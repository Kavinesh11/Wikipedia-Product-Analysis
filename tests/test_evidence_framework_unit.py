"""Unit tests for Evidence Framework validation logic.

This module tests specific scenarios for cross-source validation,
sensitivity analysis, and robustness checks with concrete examples.

Tests Requirements 5.1, 5.5, 5.6 from the requirements document.
"""

import pytest
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


# Fixtures for test data

@pytest.fixture
def sample_finding():
    """Create a sample finding with positive effect."""
    test_result = TestResult(
        test_name='t_test',
        statistic=2.5,
        p_value=0.01,
        effect_size=0.5,
        confidence_interval=(0.3, 0.7),
        is_significant=True,
        alpha=0.05,
        interpretation="Significant positive effect"
    )
    
    return Finding(
        finding_id='test_finding_001',
        description='Test finding with positive trend',
        evidence=[test_result],
        causal_effects=[],
        confidence_level='high',
        requirements_validated=['5.1', '5.2']
    )


@pytest.fixture
def sample_time_series():
    """Create a sample time series with upward trend."""
    dates = pd.date_range(start='2020-01-01', periods=90, freq='D')
    # Linear upward trend with some noise
    values = np.linspace(10000, 15000, 90) + np.random.normal(0, 100, 90)
    
    return TimeSeriesData(
        date=pd.Series(dates),
        values=pd.Series(values),
        platform='all',
        metric_type='pageviews',
        metadata={'source': 'test'}
    )


@pytest.fixture
def cross_validator():
    """Create a CrossValidator instance."""
    return CrossValidator(significance_level=0.05)


@pytest.fixture
def robustness_checker():
    """Create a RobustnessChecker instance."""
    return RobustnessChecker(significance_level=0.05)


# Tests for cross-source validation

class TestCrossSourceValidation:
    """Test cross-source validation with multi-source data."""
    
    def test_validate_with_all_supporting_sources(self, cross_validator, sample_finding):
        """Test validation when all sources support the finding."""
        # Create three data sources with consistent upward trends
        data_sources = {}
        for source_name in ['pageviews', 'editors', 'edits']:
            dates = pd.date_range(start='2020-01-01', periods=90, freq='D')
            # All have positive trends matching the finding
            values = np.linspace(10000, 15000, 90) + np.random.normal(0, 50, 90)
            
            data_sources[source_name] = TimeSeriesData(
                date=pd.Series(dates),
                values=pd.Series(values),
                platform='all',
                metric_type=source_name,
                metadata={'source': 'test'}
            )
        
        # Validate
        result = cross_validator.validate_across_sources(sample_finding, data_sources)
        
        # Assertions
        assert result.is_consistent, "Should be consistent when all sources support"
        assert result.consistency_score > 0.5, "Consistency score should be high"
        assert len(result.supporting_sources) >= 2, "Should have multiple supporting sources"
        assert len(result.contradicting_sources) == 0, "Should have no contradicting sources"
    
    def test_validate_with_mixed_sources(self, cross_validator, sample_finding):
        """Test validation when sources are mixed (some support, some contradict)."""
        data_sources = {}
        
        # Two sources with positive trends (supporting)
        for source_name in ['pageviews', 'editors']:
            dates = pd.date_range(start='2020-01-01', periods=90, freq='D')
            values = np.linspace(10000, 15000, 90) + np.random.normal(0, 50, 90)
            
            data_sources[source_name] = TimeSeriesData(
                date=pd.Series(dates),
                values=pd.Series(values),
                platform='all',
                metric_type=source_name,
                metadata={'source': 'test'}
            )
        
        # One source with negative trend (contradicting)
        dates = pd.date_range(start='2020-01-01', periods=90, freq='D')
        values = np.linspace(15000, 10000, 90) + np.random.normal(0, 50, 90)
        
        data_sources['edits'] = TimeSeriesData(
            date=pd.Series(dates),
            values=pd.Series(values),
            platform='all',
            metric_type='edits',
            metadata={'source': 'test'}
        )
        
        # Validate
        result = cross_validator.validate_across_sources(sample_finding, data_sources)
        
        # Assertions
        assert 0.0 <= result.consistency_score <= 1.0, "Consistency score in valid range"
        assert len(result.supporting_sources) > 0, "Should have some supporting sources"
        assert len(result.contradicting_sources) > 0, "Should have some contradicting sources"
    
    def test_validate_with_empty_sources(self, cross_validator, sample_finding):
        """Test validation with no data sources."""
        result = cross_validator.validate_across_sources(sample_finding, {})
        
        # Assertions
        assert not result.is_consistent, "Should not be consistent with no sources"
        assert result.consistency_score == 0.0, "Consistency score should be 0"
        assert len(result.supporting_sources) == 0, "Should have no supporting sources"
        assert 'error' in result.details, "Should have error in details"
    
    def test_validate_with_insufficient_data(self, cross_validator, sample_finding):
        """Test validation with data sources that have insufficient data points."""
        # Create data source with only 1 data point
        dates = pd.date_range(start='2020-01-01', periods=1, freq='D')
        values = pd.Series([10000])
        
        data_sources = {
            'pageviews': TimeSeriesData(
                date=pd.Series(dates),
                values=values,
                platform='all',
                metric_type='pageviews',
                metadata={'source': 'test'}
            )
        }
        
        # Validate
        result = cross_validator.validate_across_sources(sample_finding, data_sources)
        
        # Assertions
        assert 0.0 <= result.consistency_score <= 1.0, "Consistency score in valid range"
        # With insufficient data, the source should not be strongly supporting
        assert result.consistency_score < 1.0, "Should not have perfect consistency with 1 data point"


class TestCrossPlatformValidation:
    """Test cross-platform validation."""
    
    def test_validate_across_platforms_consistent(self, cross_validator, sample_finding):
        """Test validation when all platforms show consistent patterns."""
        platform_data = {}
        
        for platform in ['desktop', 'mobile-web', 'mobile-app']:
            dates = pd.date_range(start='2020-01-01', periods=90, freq='D')
            values = np.linspace(10000, 15000, 90) + np.random.normal(0, 50, 90)
            
            platform_data[platform] = TimeSeriesData(
                date=pd.Series(dates),
                values=pd.Series(values),
                platform=platform,
                metric_type='pageviews',
                metadata={'source': 'test'}
            )
        
        result = cross_validator.validate_across_platforms(sample_finding, platform_data)
        
        assert result.is_consistent, "Should be consistent across platforms"
        assert result.consistency_score > 0.5, "Consistency score should be high"
        assert len(result.supporting_sources) >= 2, "Should have multiple supporting platforms"
    
    def test_validate_across_platforms_empty(self, cross_validator, sample_finding):
        """Test validation with no platform data."""
        result = cross_validator.validate_across_platforms(sample_finding, {})
        
        assert not result.is_consistent, "Should not be consistent with no platforms"
        assert result.consistency_score == 0.0, "Consistency score should be 0"
        assert 'error' in result.details, "Should have error in details"


class TestBenchmarkComparison:
    """Test benchmark comparison functionality."""
    
    def test_compare_to_benchmark_positive_correlation(self, cross_validator):
        """Test comparison with positively correlated benchmark."""
        # Create metric and benchmark with positive correlation
        metric = pd.Series(np.linspace(1000, 2000, 100) + np.random.normal(0, 50, 100))
        benchmark = pd.Series(np.linspace(1000, 2000, 100) + np.random.normal(0, 50, 100))
        
        result = cross_validator.compare_to_benchmark(metric, benchmark)
        
        assert result.correlation > 0, "Should have positive correlation"
        assert 0.0 <= result.p_value <= 1.0, "P-value in valid range"
        assert result.relative_performance > 0, "Relative performance should be positive"
        assert result.is_aligned, "Trends should be aligned"
    
    def test_compare_to_benchmark_negative_correlation(self, cross_validator):
        """Test comparison with negatively correlated benchmark."""
        # Create metric and benchmark with negative correlation
        metric = pd.Series(np.linspace(1000, 2000, 100))
        benchmark = pd.Series(np.linspace(2000, 1000, 100))
        
        result = cross_validator.compare_to_benchmark(metric, benchmark)
        
        assert result.correlation < 0, "Should have negative correlation"
        assert not result.is_aligned, "Trends should not be aligned"
    
    def test_compare_to_benchmark_empty_data(self, cross_validator):
        """Test comparison with empty data."""
        metric = pd.Series([])
        benchmark = pd.Series([])
        
        result = cross_validator.compare_to_benchmark(metric, benchmark)
        
        assert result.correlation == 0.0, "Correlation should be 0 for empty data"
        assert result.p_value == 1.0, "P-value should be 1.0 for empty data"
        assert 'error' in result.details, "Should have error in details"
    
    def test_compare_to_benchmark_different_lengths(self, cross_validator):
        """Test comparison with different length series."""
        metric = pd.Series(np.linspace(1000, 2000, 100))
        benchmark = pd.Series(np.linspace(1000, 2000, 50))
        
        result = cross_validator.compare_to_benchmark(metric, benchmark)
        
        # Should handle different lengths by truncating
        assert result.correlation is not None, "Should compute correlation"
        assert 0.0 <= result.p_value <= 1.0, "P-value in valid range"


# Tests for sensitivity analysis

class TestSensitivityAnalysis:
    """Test sensitivity analysis with parameter sweeps."""
    
    def test_sensitivity_with_stable_results(self, robustness_checker):
        """Test sensitivity analysis when results are stable across parameters."""
        # Analysis function that returns constant result regardless of parameter
        def analysis_function(threshold=3.0):
            return 100.0  # Always returns same value
        
        base_result = 100.0
        parameters = {'threshold': [1.0, 2.0, 3.0, 4.0, 5.0]}
        
        result = robustness_checker.sensitivity_analysis(
            analysis_function, parameters, base_result
        )
        
        assert result.is_stable, "Should be stable with constant results"
        assert result.stability_score == 1.0, "Stability score should be 1.0"
        assert len(result.parameter_results) == 5, "Should have 5 parameter variations"
    
    def test_sensitivity_with_varying_results(self, robustness_checker):
        """Test sensitivity analysis when results vary with parameters."""
        # Analysis function that varies linearly with parameter
        def analysis_function(threshold=3.0):
            return threshold * 10.0
        
        base_result = 30.0
        parameters = {'threshold': [1.0, 2.0, 3.0, 4.0, 5.0]}
        
        result = robustness_checker.sensitivity_analysis(
            analysis_function, parameters, base_result
        )
        
        assert 0.0 <= result.stability_score <= 1.0, "Stability score in valid range"
        assert len(result.parameter_results) == 5, "Should have 5 parameter variations"
        assert 'result_values' in result.details, "Should have result values"
        assert len(result.details['result_values']) == 5, "Should have 5 result values"
    
    def test_sensitivity_with_multiple_parameters(self, robustness_checker):
        """Test sensitivity analysis with multiple parameters."""
        def analysis_function(alpha=0.05, beta=1.0):
            return alpha * beta * 100
        
        base_result = 5.0
        parameters = {
            'alpha': [0.01, 0.05, 0.10],
            'beta': [0.5, 1.0, 1.5]
        }
        
        result = robustness_checker.sensitivity_analysis(
            analysis_function, parameters, base_result
        )
        
        # Should test all parameter variations (3 + 3 = 6 total)
        assert len(result.parameter_results) == 6, "Should have 6 parameter variations"
        assert result.base_result == base_result, "Base result should be stored"
    
    def test_sensitivity_with_function_errors(self, robustness_checker):
        """Test sensitivity analysis when function raises errors."""
        def analysis_function(threshold=3.0):
            if threshold < 2.0:
                raise ValueError("Threshold too low")
            return threshold * 10.0
        
        base_result = 30.0
        parameters = {'threshold': [1.0, 2.0, 3.0, 4.0]}
        
        result = robustness_checker.sensitivity_analysis(
            analysis_function, parameters, base_result
        )
        
        # Should handle errors gracefully
        assert len(result.parameter_results) == 4, "Should have 4 parameter variations"
        # Check that error is recorded
        assert 'error' in result.parameter_results['threshold=1.0'], \
            "Should record error for invalid parameter"


# Tests for outlier sensitivity

class TestOutlierSensitivity:
    """Test robustness checks with outlier injection."""
    
    def test_outlier_sensitivity_with_extreme_outliers(self, robustness_checker):
        """Test outlier sensitivity with extreme outliers."""
        # Create series with extreme outliers
        base_values = np.random.normal(1000, 50, 100)
        base_values[0] = 10000  # Extreme outlier
        base_values[50] = 10000  # Another extreme outlier
        series = pd.Series(base_values)
        
        def analysis_function(s):
            return s.mean()
        
        result = robustness_checker.outlier_sensitivity(series, analysis_function)
        
        assert result.outliers_detected > 0, "Should detect outliers"
        assert result.with_outliers is not None, "Should have result with outliers"
        assert result.without_outliers is not None, "Should have result without outliers"
        assert result.impact_score > 0, "Should have measurable impact"
        assert not result.is_robust, "Should not be robust with extreme outliers"
    
    def test_outlier_sensitivity_without_outliers(self, robustness_checker):
        """Test outlier sensitivity when no outliers present."""
        # Create series without outliers
        series = pd.Series(np.random.normal(1000, 50, 100))
        
        def analysis_function(s):
            return s.mean()
        
        result = robustness_checker.outlier_sensitivity(series, analysis_function)
        
        # May detect 0 or very few outliers
        assert result.outliers_detected >= 0, "Outliers detected should be non-negative"
        assert result.is_robust, "Should be robust without significant outliers"
        assert result.impact_score < 0.1, "Impact should be low"
    
    def test_outlier_sensitivity_with_function_returning_object(self, robustness_checker):
        """Test outlier sensitivity with function returning complex object."""
        series = pd.Series(np.random.normal(1000, 50, 100))
        series[0] = 5000  # Add outlier
        
        # Function returns a TestResult object
        def analysis_function(s):
            return TestResult(
                test_name='mean_test',
                statistic=s.mean(),
                p_value=0.05,
                effect_size=s.mean() / 1000,
                confidence_interval=(s.mean() - 100, s.mean() + 100),
                is_significant=True,
                alpha=0.05,
                interpretation="Test result"
            )
        
        result = robustness_checker.outlier_sensitivity(series, analysis_function)
        
        # Should extract effect_size from TestResult
        assert result.with_outliers is not None, "Should have result with outliers"
        assert result.without_outliers is not None, "Should have result without outliers"
        assert 'val_with' in result.details, "Should extract value from with_outliers"
        assert 'val_without' in result.details, "Should extract value from without_outliers"


# Tests for method comparison

class TestMethodComparison:
    """Test method comparison functionality."""
    
    def test_method_comparison_consistent_methods(self, robustness_checker):
        """Test method comparison when methods produce consistent results."""
        series = pd.Series(np.random.normal(1000, 50, 100))
        
        # Methods that should produce similar results
        methods = {
            'mean': lambda s: s.mean(),
            'median': lambda s: s.median(),
            'trimmed_mean': lambda s: s.sort_values().iloc[10:90].mean()
        }
        
        result = robustness_checker.method_comparison(series, methods)
        
        assert result.is_consistent, "Methods should be consistent"
        assert result.consistency_score > 0.7, "Consistency score should be high"
        assert len(result.method_results) == 3, "Should have 3 method results"
        assert result.recommended_method in methods, "Recommended method should be valid"
    
    def test_method_comparison_inconsistent_methods(self, robustness_checker):
        """Test method comparison when methods produce inconsistent results."""
        # Use a series with known range to ensure min/max are very different from mean
        series = pd.Series([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        
        # Methods that produce very different results
        methods = {
            'mean': lambda s: s.mean(),
            'min': lambda s: s.min(),
            'max': lambda s: s.max()
        }
        
        result = robustness_checker.method_comparison(series, methods)
        
        # With min=100, max=1000, mean=550, CV should be high
        assert not result.is_consistent, "Methods should not be consistent"
        assert result.consistency_score < 0.7, "Consistency score should be relatively low"
        assert len(result.method_results) == 3, "Should have 3 method results"
    
    def test_method_comparison_with_errors(self, robustness_checker):
        """Test method comparison when some methods fail."""
        series = pd.Series(np.random.normal(1000, 50, 100))
        
        methods = {
            'mean': lambda s: s.mean(),
            'failing_method': lambda s: 1 / 0,  # Will raise ZeroDivisionError
            'median': lambda s: s.median()
        }
        
        result = robustness_checker.method_comparison(series, methods)
        
        # Should handle errors gracefully
        assert len(result.method_results) == 3, "Should have 3 method results"
        assert 'error' in result.method_results['failing_method'], \
            "Should record error for failing method"


# Tests for subsample stability

class TestSubsampleStability:
    """Test subsample stability analysis."""
    
    def test_subsample_stability_stable_statistic(self, robustness_checker):
        """Test subsample stability with a stable statistic (mean)."""
        # Create series with normal distribution
        series = pd.Series(np.random.normal(1000, 50, 200))
        
        def analysis_function(s):
            return s.mean()
        
        result = robustness_checker.subsample_stability(
            series, analysis_function, n_subsamples=50
        )
        
        assert len(result.subsample_results) == 50, "Should have 50 subsample results"
        assert result.mean_result is not None, "Should have mean result"
        assert result.std_result >= 0, "Standard deviation should be non-negative"
        assert result.is_stable, "Mean should be stable across subsamples"
        assert result.stability_score > 0.5, "Stability score should be high"
    
    def test_subsample_stability_with_small_subsamples(self, robustness_checker):
        """Test subsample stability with small number of subsamples."""
        series = pd.Series(np.random.normal(1000, 50, 100))
        
        def analysis_function(s):
            return s.mean()
        
        result = robustness_checker.subsample_stability(
            series, analysis_function, n_subsamples=10
        )
        
        assert len(result.subsample_results) == 10, "Should have 10 subsample results"
        assert 'n_subsamples' in result.details, "Should record n_subsamples"
        assert result.details['n_subsamples'] == 10, "Should match requested subsamples"
    
    def test_subsample_stability_with_custom_fraction(self, robustness_checker):
        """Test subsample stability with custom subsample fraction."""
        series = pd.Series(np.random.normal(1000, 50, 100))
        
        def analysis_function(s):
            return s.mean()
        
        result = robustness_checker.subsample_stability(
            series, analysis_function, n_subsamples=30, subsample_fraction=0.5
        )
        
        assert len(result.subsample_results) == 30, "Should have 30 subsample results"
        assert result.details['subsample_fraction'] == 0.5, "Should record subsample fraction"
    
    def test_subsample_stability_with_volatile_statistic(self, robustness_checker):
        """Test subsample stability with a volatile statistic (max)."""
        series = pd.Series(np.random.normal(1000, 200, 100))
        
        def analysis_function(s):
            return s.max()  # Max is more volatile than mean
        
        result = robustness_checker.subsample_stability(
            series, analysis_function, n_subsamples=50
        )
        
        # Max should be less stable than mean
        assert len(result.subsample_results) == 50, "Should have 50 subsample results"
        assert result.std_result > 0, "Should have non-zero standard deviation"
        # May or may not be marked as stable depending on the data


# Integration tests

class TestValidationIntegration:
    """Integration tests for validation logic."""
    
    def test_full_validation_workflow(self, cross_validator, sample_finding):
        """Test complete validation workflow across sources and platforms."""
        # Create multi-source data
        data_sources = {}
        for source in ['pageviews', 'editors', 'edits']:
            dates = pd.date_range(start='2020-01-01', periods=90, freq='D')
            values = np.linspace(10000, 15000, 90) + np.random.normal(0, 50, 90)
            
            data_sources[source] = TimeSeriesData(
                date=pd.Series(dates),
                values=pd.Series(values),
                platform='all',
                metric_type=source,
                metadata={'source': 'test'}
            )
        
        # Validate across sources
        source_result = cross_validator.validate_across_sources(sample_finding, data_sources)
        
        # Create platform data
        platform_data = {}
        for platform in ['desktop', 'mobile-web']:
            dates = pd.date_range(start='2020-01-01', periods=90, freq='D')
            values = np.linspace(10000, 15000, 90) + np.random.normal(0, 50, 90)
            
            platform_data[platform] = TimeSeriesData(
                date=pd.Series(dates),
                values=pd.Series(values),
                platform=platform,
                metric_type='pageviews',
                metadata={'source': 'test'}
            )
        
        # Validate across platforms
        platform_result = cross_validator.validate_across_platforms(sample_finding, platform_data)
        
        # Both validations should succeed
        assert source_result.is_consistent, "Source validation should succeed"
        assert platform_result.is_consistent, "Platform validation should succeed"
        assert source_result.consistency_score > 0.5, "Source consistency should be high"
        assert platform_result.consistency_score > 0.5, "Platform consistency should be high"
    
    def test_robustness_workflow(self, robustness_checker):
        """Test complete robustness checking workflow."""
        series = pd.Series(np.random.normal(1000, 50, 100))
        
        def analysis_function(s):
            return s.mean()
        
        # Test sensitivity
        parameters = {'threshold': [1.0, 2.0, 3.0]}
        sensitivity_result = robustness_checker.sensitivity_analysis(
            lambda threshold: analysis_function(series) * threshold,
            parameters,
            analysis_function(series)
        )
        
        # Test outlier sensitivity
        outlier_result = robustness_checker.outlier_sensitivity(series, analysis_function)
        
        # Test method comparison
        methods = {
            'mean': lambda s: s.mean(),
            'median': lambda s: s.median()
        }
        method_result = robustness_checker.method_comparison(series, methods)
        
        # Test subsample stability
        stability_result = robustness_checker.subsample_stability(
            series, analysis_function, n_subsamples=30
        )
        
        # All checks should complete successfully
        assert sensitivity_result is not None, "Sensitivity analysis should complete"
        assert outlier_result is not None, "Outlier sensitivity should complete"
        assert method_result is not None, "Method comparison should complete"
        assert stability_result is not None, "Subsample stability should complete"
        
        # All should have valid scores
        assert 0.0 <= sensitivity_result.stability_score <= 1.0
        assert 0.0 <= outlier_result.impact_score <= 1.0
        assert 0.0 <= method_result.consistency_score <= 1.0
        assert 0.0 <= stability_result.stability_score <= 1.0
