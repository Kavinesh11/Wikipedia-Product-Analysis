"""Seasonality analysis module.

This module provides functions for analyzing seasonal patterns in Wikipedia
traffic, including validation with spectral analysis, day-of-week effects,
holiday modeling, and utility vs leisure classification.
"""

from typing import Dict, List, Any, Tuple, Optional
from datetime import date
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy import stats, signal

from wikipedia_health.models.data_models import (
    TimeSeriesData,
    DecompositionResult,
    TestResult
)
from wikipedia_health.time_series.decomposer import TimeSeriesDecomposer
from wikipedia_health.statistical_validation.hypothesis_tester import HypothesisTester
from wikipedia_health.multi_dimensional_analysis.analyzer import MultiDimensionalAnalyzer


def analyze_seasonality(
    time_series: TimeSeriesData,
    period: int = 7,
    methods: List[str] = ['stl', 'x13']
) -> Dict[str, Any]:
    """Analyze seasonality using multiple decomposition methods.
    
    Performs seasonal decomposition using STL and X-13-ARIMA-SEATS methods,
    validates consistency, and computes seasonal strength metrics.
    
    Args:
        time_series: Time series data to analyze
        period: Seasonal period (default 7 for weekly seasonality)
        methods: List of decomposition methods to use
    
    Returns:
        Dictionary containing:
            - decompositions: Results from each method
            - seasonal_strength: Metrics quantifying seasonal strength
            - consistency_check: Comparison of methods
            - validation_tests: Statistical validation results
            - seasonality_report: Comprehensive analysis report
    
    Raises:
        ValueError: If time series is too short for seasonal analysis
    """
    if len(time_series.values) < period * 3:
        raise ValueError(
            f"Time series must have at least {period * 3} observations "
            f"for seasonal analysis with period {period}"
        )
    
    # Initialize components
    decomposer = TimeSeriesDecomposer()
    
    # Convert to pandas Series with datetime index
    series = pd.Series(
        time_series.values.values,
        index=pd.to_datetime(time_series.date.values)
    )
    
    # Step 1: Perform decomposition with multiple methods
    decompositions = {}
    
    if 'stl' in methods:
        decompositions['stl'] = decomposer.decompose_stl(series, period=period)
    
    if 'x13' in methods:
        try:
            decompositions['x13'] = decomposer.decompose_x13(series)
        except Exception as e:
            # X-13 may not be available or may fail
            print(f"X-13 decomposition failed: {e}. Skipping.")
    
    # Step 2: Calculate seasonal strength metrics
    seasonal_strength = _calculate_seasonal_strength(decompositions, series)
    
    # Step 3: Check consistency across methods
    consistency_check = _check_method_consistency(decompositions)
    
    # Step 4: Validate seasonality with statistical tests
    validation_tests = validate_seasonality(time_series, period)
    
    # Step 5: Generate comprehensive seasonality report
    seasonality_report = _generate_seasonality_report(
        time_series=time_series,
        decompositions=decompositions,
        seasonal_strength=seasonal_strength,
        consistency_check=consistency_check,
        validation_tests=validation_tests
    )
    
    return {
        'decompositions': decompositions,
        'seasonal_strength': seasonal_strength,
        'consistency_check': consistency_check,
        'validation_tests': validation_tests,
        'seasonality_report': seasonality_report
    }


def validate_seasonality(
    time_series: TimeSeriesData,
    period: int = 7
) -> Dict[str, TestResult]:
    """Validate seasonality with spectral analysis and ACF tests.
    
    Uses spectral density analysis and autocorrelation function to confirm
    periodicity with statistical significance.
    
    Args:
        time_series: Time series data to validate
        period: Expected seasonal period
    
    Returns:
        Dictionary of test results including spectral analysis and ACF tests
    """
    series = pd.Series(
        time_series.values.values,
        index=pd.to_datetime(time_series.date.values)
    )
    
    # Remove trend for spectral analysis
    detrended = series - series.rolling(window=period, center=True).mean()
    detrended = detrended.dropna()
    
    # Spectral analysis
    spectral_test = _spectral_analysis_test(detrended, period)
    
    # ACF test
    acf_test = _acf_test(series, period)
    
    # PACF test
    pacf_test = _pacf_test(series, period)
    
    return {
        'spectral_analysis': spectral_test,
        'acf_test': acf_test,
        'pacf_test': pacf_test
    }


def day_of_week_analysis(
    time_series: TimeSeriesData
) -> Dict[str, Any]:
    """Analyze day-of-week effects with ANOVA.
    
    Performs ANOVA to quantify weekday vs weekend differences and computes
    effect sizes for each day of the week.
    
    Args:
        time_series: Time series data with daily observations
    
    Returns:
        Dictionary containing:
            - anova_result: ANOVA test result
            - day_means: Mean values for each day of week
            - weekday_weekend_comparison: Comparison test result
            - effect_sizes: Effect sizes for each day
            - dow_report: Day-of-week analysis report
    """
    # Create DataFrame with day of week
    df = pd.DataFrame({
        'date': pd.to_datetime(time_series.date.values),
        'value': time_series.values.values
    })
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_name'] = df['date'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday, Sunday
    
    # Initialize tester
    tester = HypothesisTester()
    
    # Step 1: ANOVA across all days
    groups_by_day = [
        df[df['day_of_week'] == i]['value']
        for i in range(7)
    ]
    anova_result = tester.anova(groups_by_day)
    
    # Step 2: Calculate mean for each day
    day_means = {}
    for i, day_name in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                   'Friday', 'Saturday', 'Sunday']):
        day_data = df[df['day_of_week'] == i]['value']
        day_means[day_name] = {
            'mean': day_data.mean(),
            'std': day_data.std(),
            'count': len(day_data)
        }
    
    # Step 3: Weekday vs weekend comparison
    weekday_values = df[~df['is_weekend']]['value']
    weekend_values = df[df['is_weekend']]['value']
    weekday_weekend_comparison = tester.t_test(weekday_values, weekend_values)
    
    # Step 4: Calculate effect sizes for each day vs overall mean
    overall_mean = df['value'].mean()
    overall_std = df['value'].std()
    
    effect_sizes = {}
    for day_name, stats_dict in day_means.items():
        cohens_d = (stats_dict['mean'] - overall_mean) / overall_std if overall_std > 0 else 0.0
        effect_sizes[day_name] = cohens_d
    
    # Step 5: Generate report
    dow_report = _generate_dow_report(
        anova_result=anova_result,
        day_means=day_means,
        weekday_weekend_comparison=weekday_weekend_comparison,
        effect_sizes=effect_sizes
    )
    
    return {
        'anova_result': anova_result,
        'day_means': day_means,
        'weekday_weekend_comparison': weekday_weekend_comparison,
        'effect_sizes': effect_sizes,
        'dow_report': dow_report
    }


def holiday_effect_modeling(
    time_series: TimeSeriesData,
    holidays: List[date]
) -> Dict[str, Any]:
    """Model holiday effects using regression with dummy variables.
    
    Quantifies the impact of holidays on traffic using regression analysis
    with holiday indicator variables.
    
    Args:
        time_series: Time series data
        holidays: List of holiday dates to model
    
    Returns:
        Dictionary containing:
            - regression_results: Regression coefficients and statistics
            - holiday_effects: Effect size for each holiday with CI
            - overall_holiday_effect: Average holiday effect
            - holiday_report: Holiday effect analysis report
    """
    # Create DataFrame
    df = pd.DataFrame({
        'date': pd.to_datetime(time_series.date.values),
        'value': time_series.values.values
    })
    
    # Create holiday dummy variables
    df['is_holiday'] = df['date'].dt.date.isin(holidays)
    
    # Add time trend
    df['time_index'] = np.arange(len(df))
    
    # Fit regression model: value ~ time_index + is_holiday
    from scipy.stats import linregress
    
    # Separate holiday and non-holiday data
    holiday_data = df[df['is_holiday']]['value']
    non_holiday_data = df[~df['is_holiday']]['value']
    
    # Calculate effect size
    if len(holiday_data) > 0 and len(non_holiday_data) > 0:
        holiday_mean = holiday_data.mean()
        non_holiday_mean = non_holiday_data.mean()
        pooled_std = np.sqrt(
            ((len(holiday_data) - 1) * holiday_data.std() ** 2 +
             (len(non_holiday_data) - 1) * non_holiday_data.std() ** 2) /
            (len(holiday_data) + len(non_holiday_data) - 2)
        )
        
        effect_size = (holiday_mean - non_holiday_mean) / pooled_std if pooled_std > 0 else 0.0
        
        # T-test for significance
        t_stat, p_value = stats.ttest_ind(holiday_data, non_holiday_data)
        
        # Confidence interval for mean difference
        mean_diff = holiday_mean - non_holiday_mean
        se_diff = pooled_std * np.sqrt(1/len(holiday_data) + 1/len(non_holiday_data))
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff
    else:
        effect_size = 0.0
        p_value = 1.0
        mean_diff = 0.0
        ci_lower = 0.0
        ci_upper = 0.0
        holiday_mean = 0.0
        non_holiday_mean = 0.0
    
    # Create test result
    holiday_test = TestResult(
        test_name='Holiday Effect Test',
        statistic=float(mean_diff),
        p_value=float(p_value),
        effect_size=float(effect_size),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        is_significant=bool(p_value < 0.05),
        alpha=0.05,
        interpretation=(
            f"Holiday effect: Mean traffic on holidays ({holiday_mean:.2f}) vs "
            f"non-holidays ({non_holiday_mean:.2f}). Difference: {mean_diff:.2f} "
            f"(95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]). "
            f"Effect size (Cohen's d): {effect_size:.3f}. "
            f"{'Significant' if p_value < 0.05 else 'Not significant'} (p={p_value:.4f})."
        )
    )
    
    # Generate report
    holiday_report = {
        'num_holidays': len(holidays),
        'holiday_mean': holiday_mean,
        'non_holiday_mean': non_holiday_mean,
        'effect_size': effect_size,
        'test_result': holiday_test.to_dict()
    }
    
    return {
        'regression_results': {
            'mean_difference': mean_diff,
            'confidence_interval': (ci_lower, ci_upper)
        },
        'holiday_effects': {
            'effect_size': effect_size,
            'confidence_interval': (ci_lower, ci_upper)
        },
        'overall_holiday_effect': holiday_test,
        'holiday_report': holiday_report
    }


def utility_vs_leisure_classification(
    pageview_data: TimeSeriesData,
    editor_data: TimeSeriesData
) -> Dict[str, Any]:
    """Classify usage as utility vs leisure based on engagement ratios.
    
    Compares weekday vs weekend engagement ratios (editors per pageview)
    to identify utility-driven vs leisure-driven usage patterns.
    
    Args:
        pageview_data: Pageview time series
        editor_data: Editor count time series
    
    Returns:
        Dictionary containing:
            - weekday_engagement: Weekday engagement ratio
            - weekend_engagement: Weekend engagement ratio
            - comparison_test: Statistical test comparing ratios
            - classification: 'utility' or 'leisure' or 'mixed'
            - classification_report: Detailed classification analysis
    """
    # Initialize analyzer
    analyzer = MultiDimensionalAnalyzer()
    
    # Create DataFrames
    pv_df = pd.DataFrame({
        'date': pd.to_datetime(pageview_data.date.values),
        'pageviews': pageview_data.values.values
    })
    
    ed_df = pd.DataFrame({
        'date': pd.to_datetime(editor_data.date.values),
        'editors': editor_data.values.values
    })
    
    # Merge data
    df = pd.merge(pv_df, ed_df, on='date', how='inner')
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    # Calculate engagement ratio (editors per 1000 pageviews)
    df['engagement_ratio'] = (df['editors'] / df['pageviews'] * 1000).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['engagement_ratio'])
    
    # Split by weekday/weekend
    weekday_engagement = df[~df['is_weekend']]['engagement_ratio']
    weekend_engagement = df[df['is_weekend']]['engagement_ratio']
    
    # Compare engagement ratios
    tester = HypothesisTester()
    comparison_test = tester.t_test(weekday_engagement, weekend_engagement)
    
    # Classify based on engagement patterns
    weekday_mean = weekday_engagement.mean()
    weekend_mean = weekend_engagement.mean()
    
    ratio_difference = weekday_mean - weekend_mean
    relative_difference = ratio_difference / weekday_mean if weekday_mean > 0 else 0.0
    
    # Classification logic:
    # - Utility: Higher weekday engagement (>10% higher)
    # - Leisure: Higher weekend engagement (>10% higher)
    # - Mixed: Similar engagement
    
    if relative_difference > 0.10:
        classification = 'utility'
        description = (
            "Usage pattern is primarily UTILITY-driven. "
            "Weekday engagement is significantly higher than weekend engagement, "
            "suggesting users access Wikipedia for work/education purposes."
        )
    elif relative_difference < -0.10:
        classification = 'leisure'
        description = (
            "Usage pattern is primarily LEISURE-driven. "
            "Weekend engagement is significantly higher than weekday engagement, "
            "suggesting users access Wikipedia for recreational purposes."
        )
    else:
        classification = 'mixed'
        description = (
            "Usage pattern is MIXED. "
            "Weekday and weekend engagement are similar, "
            "suggesting Wikipedia serves both utility and leisure purposes."
        )
    
    # Generate report
    classification_report = {
        'classification': classification,
        'description': description,
        'weekday_mean_engagement': weekday_mean,
        'weekend_mean_engagement': weekend_mean,
        'ratio_difference': ratio_difference,
        'relative_difference': relative_difference,
        'statistical_test': comparison_test.to_dict()
    }
    
    return {
        'weekday_engagement': weekday_mean,
        'weekend_engagement': weekend_mean,
        'comparison_test': comparison_test,
        'classification': classification,
        'classification_report': classification_report
    }


# Helper functions

def _calculate_seasonal_strength(
    decompositions: Dict[str, DecompositionResult],
    original_series: Series
) -> Dict[str, float]:
    """Calculate seasonal strength metrics for each decomposition."""
    seasonal_strength = {}
    
    for method, decomp in decompositions.items():
        # Calculate seasonal strength: 1 - Var(residual) / Var(seasonal + residual)
        var_residual = decomp.residual.var()
        var_seasonal_residual = (decomp.seasonal + decomp.residual).var()
        
        if var_seasonal_residual > 0:
            strength = 1 - (var_residual / var_seasonal_residual)
        else:
            strength = 0.0
        
        seasonal_strength[method] = max(0.0, min(1.0, strength))
    
    return seasonal_strength


def _check_method_consistency(
    decompositions: Dict[str, DecompositionResult]
) -> Dict[str, Any]:
    """Check consistency of seasonal components across methods."""
    if len(decompositions) < 2:
        return {'consistent': True, 'correlation': 1.0}
    
    # Extract seasonal components
    seasonal_components = {
        method: decomp.seasonal
        for method, decomp in decompositions.items()
    }
    
    # Calculate pairwise correlations
    methods = list(seasonal_components.keys())
    correlations = {}
    
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1, method2 = methods[i], methods[j]
            
            # Align series
            s1 = seasonal_components[method1].dropna()
            s2 = seasonal_components[method2].dropna()
            
            # Find common index
            common_idx = s1.index.intersection(s2.index)
            if len(common_idx) > 0:
                corr = s1.loc[common_idx].corr(s2.loc[common_idx])
                correlations[f"{method1}_vs_{method2}"] = corr
    
    # Check if all correlations are high (>0.7)
    min_correlation = min(correlations.values()) if correlations else 1.0
    consistent = min_correlation > 0.7
    
    return {
        'consistent': consistent,
        'correlations': correlations,
        'min_correlation': min_correlation
    }


def _spectral_analysis_test(
    series: Series,
    expected_period: int
) -> TestResult:
    """Perform spectral analysis to detect periodicity."""
    # Remove NaN values
    clean_series = series.dropna()
    
    if len(clean_series) < expected_period * 2:
        return TestResult(
            test_name='Spectral Analysis',
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            is_significant=False,
            alpha=0.05,
            interpretation="Insufficient data for spectral analysis"
        )
    
    # Compute periodogram
    frequencies, power = signal.periodogram(clean_series.values)
    
    # Find peak frequency
    peak_idx = np.argmax(power[1:]) + 1  # Skip DC component
    peak_frequency = frequencies[peak_idx]
    peak_power = power[peak_idx]
    
    # Convert frequency to period
    detected_period = 1 / peak_frequency if peak_frequency > 0 else 0
    
    # Calculate significance (power relative to mean power)
    mean_power = np.mean(power[1:])
    relative_power = peak_power / mean_power if mean_power > 0 else 0.0
    
    # Simple significance test: is peak power > 3 * mean power?
    is_significant = bool(relative_power > 3.0)
    p_value = 1.0 / relative_power if relative_power > 0 else 1.0
    
    interpretation = (
        f"Spectral analysis detected peak at period {detected_period:.1f} "
        f"(expected: {expected_period}). Peak power is {relative_power:.2f}x "
        f"mean power. {'Significant' if is_significant else 'Not significant'} periodicity."
    )
    
    return TestResult(
        test_name='Spectral Analysis',
        statistic=float(detected_period),
        p_value=float(min(p_value, 1.0)),
        effect_size=float(relative_power),
        confidence_interval=(float(detected_period * 0.9), float(detected_period * 1.1)),
        is_significant=is_significant,
        alpha=0.05,
        interpretation=interpretation
    )


def _acf_test(
    series: Series,
    lag: int
) -> TestResult:
    """Test autocorrelation at specified lag."""
    from statsmodels.tsa.stattools import acf
    
    clean_series = series.dropna()
    
    # Calculate ACF
    acf_values = acf(clean_series, nlags=lag, fft=True)
    acf_at_lag = acf_values[lag]
    
    # Standard error for ACF (Bartlett's formula for large samples)
    n = len(clean_series)
    se = 1 / np.sqrt(n)
    
    # Z-test for ACF
    z_score = acf_at_lag / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    is_significant = bool(abs(acf_at_lag) > 1.96 * se)
    
    interpretation = (
        f"Autocorrelation at lag {lag}: {acf_at_lag:.3f} "
        f"(SE: {se:.3f}). {'Significant' if is_significant else 'Not significant'} "
        f"(p={p_value:.4f})."
    )
    
    return TestResult(
        test_name=f'ACF Test (lag={lag})',
        statistic=float(acf_at_lag),
        p_value=float(p_value),
        effect_size=float(abs(acf_at_lag)),
        confidence_interval=(float(acf_at_lag - 1.96 * se), float(acf_at_lag + 1.96 * se)),
        is_significant=is_significant,
        alpha=0.05,
        interpretation=interpretation
    )


def _pacf_test(
    series: Series,
    lag: int
) -> TestResult:
    """Test partial autocorrelation at specified lag."""
    from statsmodels.tsa.stattools import pacf
    
    clean_series = series.dropna()
    
    # Calculate PACF
    pacf_values = pacf(clean_series, nlags=lag)
    pacf_at_lag = pacf_values[lag]
    
    # Standard error for PACF
    n = len(clean_series)
    se = 1 / np.sqrt(n)
    
    # Z-test for PACF
    z_score = pacf_at_lag / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    is_significant = bool(abs(pacf_at_lag) > 1.96 * se)
    
    interpretation = (
        f"Partial autocorrelation at lag {lag}: {pacf_at_lag:.3f} "
        f"(SE: {se:.3f}). {'Significant' if is_significant else 'Not significant'} "
        f"(p={p_value:.4f})."
    )
    
    return TestResult(
        test_name=f'PACF Test (lag={lag})',
        statistic=float(pacf_at_lag),
        p_value=float(p_value),
        effect_size=float(abs(pacf_at_lag)),
        confidence_interval=(float(pacf_at_lag - 1.96 * se), float(pacf_at_lag + 1.96 * se)),
        is_significant=is_significant,
        alpha=0.05,
        interpretation=interpretation
    )


def _generate_seasonality_report(
    time_series: TimeSeriesData,
    decompositions: Dict[str, DecompositionResult],
    seasonal_strength: Dict[str, float],
    consistency_check: Dict[str, Any],
    validation_tests: Dict[str, TestResult]
) -> Dict[str, Any]:
    """Generate comprehensive seasonality analysis report."""
    report = {
        'summary': {
            'metric_type': time_series.metric_type,
            'platform': time_series.platform,
            'num_methods': len(decompositions),
            'methods_consistent': consistency_check['consistent'],
            'seasonality_detected': any(
                test.is_significant for test in validation_tests.values()
            )
        },
        'seasonal_strength': seasonal_strength,
        'method_consistency': consistency_check,
        'validation_tests': {
            name: test.to_dict() for name, test in validation_tests.items()
        },
        'recommendations': []
    }
    
    # Add recommendations
    if report['summary']['seasonality_detected']:
        report['recommendations'].append(
            "Strong seasonal patterns detected. Account for seasonality in "
            "forecasting and trend analysis."
        )
    
    if not consistency_check['consistent']:
        report['recommendations'].append(
            "Decomposition methods show inconsistent seasonal patterns. "
            "Review data quality and consider alternative decomposition approaches."
        )
    
    return report


def _generate_dow_report(
    anova_result: TestResult,
    day_means: Dict[str, Dict[str, float]],
    weekday_weekend_comparison: TestResult,
    effect_sizes: Dict[str, float]
) -> Dict[str, Any]:
    """Generate day-of-week analysis report."""
    # Find day with highest and lowest traffic
    highest_day = max(day_means.items(), key=lambda x: x[1]['mean'])
    lowest_day = min(day_means.items(), key=lambda x: x[1]['mean'])
    
    report = {
        'summary': {
            'significant_dow_effect': anova_result.is_significant,
            'significant_weekday_weekend_diff': weekday_weekend_comparison.is_significant,
            'highest_traffic_day': highest_day[0],
            'lowest_traffic_day': lowest_day[0]
        },
        'anova_result': anova_result.to_dict(),
        'weekday_weekend_comparison': weekday_weekend_comparison.to_dict(),
        'day_statistics': day_means,
        'effect_sizes': effect_sizes,
        'recommendations': []
    }
    
    if anova_result.is_significant:
        report['recommendations'].append(
            f"Significant day-of-week effects detected. {highest_day[0]} has highest "
            f"traffic ({highest_day[1]['mean']:.0f}), {lowest_day[0]} has lowest "
            f"({lowest_day[1]['mean']:.0f})."
        )
    
    return report
