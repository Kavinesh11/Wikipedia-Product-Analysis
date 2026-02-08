"""Structural shift analysis module.

This module provides functions for detecting and analyzing structural shifts
in Wikipedia traffic patterns, including temporal alignment testing with
external events and pre/post comparison analysis.
"""

from typing import List, Dict, Any, Tuple, Optional
from datetime import date, timedelta
import pandas as pd
from pandas import Series
import numpy as np
from scipy import stats

from wikipedia_health.models.data_models import (
    TimeSeriesData,
    Changepoint,
    TestResult,
    Finding
)
from wikipedia_health.time_series.changepoint_detector import ChangepointDetector
from wikipedia_health.time_series.decomposer import TimeSeriesDecomposer
from wikipedia_health.statistical_validation.hypothesis_tester import HypothesisTester


def analyze_structural_shifts(
    time_series: TimeSeriesData,
    min_segment_size: int = 30,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """Analyze structural shifts in time series data.
    
    Orchestrates changepoint detection using multiple algorithms (PELT, Binary
    Segmentation, Bayesian) and requires consensus. Tests statistical significance
    of detected shifts and generates comprehensive shift reports.
    
    Args:
        time_series: Time series data to analyze
        min_segment_size: Minimum size of segments between changepoints
        significance_level: Alpha level for statistical tests (default 0.05)
    
    Returns:
        Dictionary containing:
            - changepoints: List of detected Changepoint objects
            - consensus_changepoints: Changepoints detected by multiple methods
            - test_results: Statistical test results for each changepoint
            - decomposition: Time series decomposition result
            - shift_report: Comprehensive report with all required elements
    
    Raises:
        ValueError: If time series is too short for analysis
    """
    if len(time_series.values) < min_segment_size * 3:
        raise ValueError(
            f"Time series must have at least {min_segment_size * 3} observations "
            f"for structural shift analysis"
        )
    
    # Initialize components
    detector = ChangepointDetector()
    decomposer = TimeSeriesDecomposer()
    tester = HypothesisTester()
    
    # Convert to pandas Series with datetime index
    series = pd.Series(
        time_series.values.values,
        index=pd.to_datetime(time_series.date.values)
    )
    
    # Step 1: Decompose time series to extract trend
    decomposition = decomposer.decompose_stl(series, period=7)
    trend_series = decomposition.trend
    
    # Step 2: Detect changepoints using multiple methods
    changepoints_pelt = detector.detect_pelt(
        trend_series,
        penalty=None,
        min_size=min_segment_size
    )
    
    changepoints_binseg = detector.detect_binary_segmentation(
        trend_series,
        n_changepoints=5
    )
    
    changepoints_bayesian = detector.detect_bayesian(
        trend_series,
        prior_scale=0.05
    )
    
    # Step 3: Find consensus changepoints (detected by at least 2 methods)
    all_changepoints = {
        'pelt': changepoints_pelt,
        'binseg': changepoints_binseg,
        'bayesian': changepoints_bayesian
    }
    
    consensus_changepoints = _find_consensus_changepoints(
        all_changepoints,
        tolerance_days=7
    )
    
    # Step 4: Test statistical significance of consensus changepoints
    test_results = []
    significant_changepoints = []
    
    for cp in consensus_changepoints:
        is_significant, p_value = detector.test_significance(
            series,
            cp,
            alpha=significance_level
        )
        
        if is_significant:
            # Perform additional t-test for effect size
            idx = cp.index
            pre_segment = series.iloc[:idx]
            post_segment = series.iloc[idx:]
            
            test_result = tester.t_test(pre_segment, post_segment)
            test_results.append(test_result)
            significant_changepoints.append(cp)
    
    # Step 5: Calculate growth rates for pre/post periods
    growth_rates = []
    for cp in significant_changepoints:
        pre_rate, post_rate, rate_ci = _calculate_growth_rates(
            series,
            cp.index,
            confidence_level=0.95
        )
        growth_rates.append({
            'changepoint_date': cp.date,
            'pre_break_rate': pre_rate,
            'post_break_rate': post_rate,
            'rate_change': post_rate - pre_rate,
            'confidence_interval': rate_ci
        })
    
    # Step 6: Generate comprehensive shift report
    shift_report = _generate_shift_report(
        time_series=time_series,
        changepoints=significant_changepoints,
        test_results=test_results,
        growth_rates=growth_rates,
        decomposition=decomposition
    )
    
    return {
        'changepoints': {
            'pelt': changepoints_pelt,
            'binseg': changepoints_binseg,
            'bayesian': changepoints_bayesian
        },
        'consensus_changepoints': consensus_changepoints,
        'significant_changepoints': significant_changepoints,
        'test_results': test_results,
        'growth_rates': growth_rates,
        'decomposition': decomposition,
        'shift_report': shift_report
    }


def temporal_alignment_test(
    changepoint_date: date,
    external_event_date: date,
    time_series: TimeSeriesData,
    tolerance_days: int = 14,
    n_permutations: int = 1000
) -> TestResult:
    """Test whether temporal alignment between shift and external event is significant.
    
    Uses permutation testing to determine if the observed temporal proximity
    between a structural shift and an external event is statistically significant
    beyond chance coincidence.
    
    Args:
        changepoint_date: Date of detected structural shift
        external_event_date: Date of external event (e.g., ChatGPT launch)
        time_series: Original time series data
        tolerance_days: Maximum days between changepoint and event to consider aligned
        n_permutations: Number of random permutations for significance testing
    
    Returns:
        TestResult with alignment test statistics and p-value
    
    Raises:
        ValueError: If dates are outside time series range
    """
    # Validate dates
    series_start = pd.to_datetime(time_series.date.values[0])
    series_end = pd.to_datetime(time_series.date.values[-1])
    
    if not (series_start <= pd.Timestamp(changepoint_date) <= series_end):
        raise ValueError("Changepoint date is outside time series range")
    
    if not (series_start <= pd.Timestamp(external_event_date) <= series_end):
        raise ValueError("External event date is outside time series range")
    
    # Calculate observed temporal distance
    observed_distance = abs((changepoint_date - external_event_date).days)
    
    # Check if within tolerance
    is_aligned = observed_distance <= tolerance_days
    
    # Perform permutation test
    # Generate random dates within the time series range
    total_days = (series_end - series_start).days
    rng = np.random.RandomState(42)
    
    random_distances = []
    for _ in range(n_permutations):
        # Generate random changepoint date
        random_offset = rng.randint(0, total_days + 1)
        random_cp_date = series_start + timedelta(days=int(random_offset))
        
        # Calculate distance to external event
        random_distance = abs((random_cp_date.date() - external_event_date).days)
        random_distances.append(random_distance)
    
    random_distances = np.array(random_distances)
    
    # Calculate p-value: proportion of random distances <= observed distance
    p_value = np.mean(random_distances <= observed_distance)
    
    # Calculate effect size (standardized distance)
    mean_random_distance = np.mean(random_distances)
    std_random_distance = np.std(random_distances)
    effect_size = (mean_random_distance - observed_distance) / std_random_distance if std_random_distance > 0 else 0.0
    
    # Calculate confidence interval
    ci_lower = np.percentile(random_distances, 2.5)
    ci_upper = np.percentile(random_distances, 97.5)
    
    # Determine significance
    alpha = 0.05
    is_significant = bool(p_value < alpha)
    
    # Generate interpretation
    interpretation = (
        f"Temporal alignment test: Changepoint ({changepoint_date}) is "
        f"{observed_distance} days from external event ({external_event_date}). "
        f"This alignment is {'significant' if is_significant else 'not significant'} "
        f"(p={p_value:.4f}) based on {n_permutations} random permutations. "
        f"Mean random distance: {mean_random_distance:.1f} days (95% CI: [{ci_lower:.1f}, {ci_upper:.1f}])."
    )
    
    return TestResult(
        test_name='Temporal Alignment Test',
        statistic=float(observed_distance),
        p_value=float(p_value),
        effect_size=float(effect_size),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        is_significant=is_significant,
        alpha=alpha,
        interpretation=interpretation
    )


def pre_post_comparison(
    time_series: TimeSeriesData,
    changepoint: Changepoint,
    test_type: str = 'auto'
) -> TestResult:
    """Compare pre-shift and post-shift periods with statistical tests.
    
    Performs appropriate statistical tests (t-test or Mann-Whitney) to confirm
    mean differences between pre-changepoint and post-changepoint periods.
    
    Args:
        time_series: Time series data
        changepoint: Detected changepoint to analyze
        test_type: Type of test to perform ('t-test', 'mann-whitney', 'auto')
                  'auto' selects based on normality tests
    
    Returns:
        TestResult with comparison statistics
    
    Raises:
        ValueError: If changepoint index is invalid
    """
    if changepoint.index <= 0 or changepoint.index >= len(time_series.values):
        raise ValueError("Changepoint index is outside valid range")
    
    # Split series at changepoint
    pre_values = time_series.values.iloc[:changepoint.index]
    post_values = time_series.values.iloc[changepoint.index:]
    
    if len(pre_values) < 2 or len(post_values) < 2:
        raise ValueError("Insufficient data in pre or post period")
    
    # Initialize tester
    tester = HypothesisTester()
    
    # Auto-select test type based on normality
    if test_type == 'auto':
        # Test normality using Shapiro-Wilk test
        _, p_pre = stats.shapiro(pre_values) if len(pre_values) <= 5000 else (0, 0.05)
        _, p_post = stats.shapiro(post_values) if len(post_values) <= 5000 else (0, 0.05)
        
        # Use t-test if both are normal, otherwise Mann-Whitney
        if p_pre > 0.05 and p_post > 0.05:
            test_type = 't-test'
        else:
            test_type = 'mann-whitney'
    
    # Perform selected test
    if test_type == 't-test':
        result = tester.t_test(pre_values, post_values, alternative='two-sided')
    elif test_type == 'mann-whitney':
        result = tester.mann_whitney(pre_values, post_values)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    return result


def _find_consensus_changepoints(
    changepoints_by_method: Dict[str, List[Changepoint]],
    tolerance_days: int = 7
) -> List[Changepoint]:
    """Find changepoints detected by multiple methods.
    
    Args:
        changepoints_by_method: Dictionary mapping method names to changepoint lists
        tolerance_days: Maximum days between changepoints to consider them the same
    
    Returns:
        List of consensus changepoints (detected by at least 2 methods)
    """
    # Flatten all changepoints with method labels
    all_cps = []
    for method, cps in changepoints_by_method.items():
        for cp in cps:
            all_cps.append((method, cp))
    
    if not all_cps:
        return []
    
    # Group changepoints by proximity
    consensus = []
    used_indices = set()
    
    for i, (method1, cp1) in enumerate(all_cps):
        if i in used_indices:
            continue
        
        # Find all changepoints within tolerance
        cluster = [(method1, cp1)]
        used_indices.add(i)
        
        for j, (method2, cp2) in enumerate(all_cps):
            if j in used_indices or j <= i:
                continue
            
            # Check if dates are within tolerance
            days_diff = abs((cp1.date - cp2.date).days)
            if days_diff <= tolerance_days:
                cluster.append((method2, cp2))
                used_indices.add(j)
        
        # If detected by at least 2 methods, add to consensus
        unique_methods = set(method for method, _ in cluster)
        if len(unique_methods) >= 2:
            # Use the changepoint with highest confidence
            best_cp = max(cluster, key=lambda x: x[1].confidence)[1]
            consensus.append(best_cp)
    
    # Sort by date
    consensus.sort(key=lambda cp: cp.date)
    
    return consensus


def _calculate_growth_rates(
    series: Series,
    changepoint_index: int,
    confidence_level: float = 0.95
) -> Tuple[float, float, Tuple[float, float]]:
    """Calculate growth rates for pre and post periods.
    
    Args:
        series: Time series data
        changepoint_index: Index of changepoint
        confidence_level: Confidence level for intervals
    
    Returns:
        Tuple of (pre_rate, post_rate, confidence_interval_for_difference)
    """
    # Split series
    pre_series = series.iloc[:changepoint_index]
    post_series = series.iloc[changepoint_index:]
    
    # Calculate growth rates using linear regression
    def fit_growth_rate(data: Series) -> Tuple[float, float]:
        """Fit linear trend and return growth rate and std error."""
        if len(data) < 2:
            return 0.0, 0.0
        
        x = np.arange(len(data))
        y = data.values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Convert slope to percentage growth rate
        mean_value = np.mean(y)
        growth_rate = (slope / mean_value * 100) if mean_value != 0 else 0.0
        
        return growth_rate, std_err
    
    pre_rate, pre_se = fit_growth_rate(pre_series)
    post_rate, post_se = fit_growth_rate(post_series)
    
    # Calculate confidence interval for rate difference
    rate_diff = post_rate - pre_rate
    se_diff = np.sqrt(pre_se**2 + post_se**2)
    
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    ci_lower = rate_diff - z_score * se_diff
    ci_upper = rate_diff + z_score * se_diff
    
    return pre_rate, post_rate, (ci_lower, ci_upper)


def _generate_shift_report(
    time_series: TimeSeriesData,
    changepoints: List[Changepoint],
    test_results: List[TestResult],
    growth_rates: List[Dict[str, Any]],
    decomposition: Any
) -> Dict[str, Any]:
    """Generate comprehensive structural shift report.
    
    Args:
        time_series: Original time series data
        changepoints: List of significant changepoints
        test_results: Statistical test results
        growth_rates: Growth rate analysis results
        decomposition: Time series decomposition
    
    Returns:
        Dictionary containing comprehensive shift report
    """
    report = {
        'summary': {
            'metric_type': time_series.metric_type,
            'platform': time_series.platform,
            'analysis_period': (
                time_series.date.iloc[0],
                time_series.date.iloc[-1]
            ),
            'num_shifts_detected': len(changepoints),
            'detection_methods': ['PELT', 'Binary Segmentation', 'Bayesian']
        },
        'structural_shifts': [],
        'statistical_evidence': [],
        'growth_rate_analysis': growth_rates,
        'recommendations': []
    }
    
    # Add details for each changepoint
    for i, cp in enumerate(changepoints):
        shift_detail = {
            'shift_id': i + 1,
            'date': cp.date,
            'direction': cp.direction,
            'magnitude': cp.magnitude,
            'confidence': cp.confidence,
            'pre_mean': cp.pre_mean,
            'post_mean': cp.post_mean,
            'percentage_change': ((cp.post_mean - cp.pre_mean) / cp.pre_mean * 100) if cp.pre_mean != 0 else 0.0
        }
        report['structural_shifts'].append(shift_detail)
    
    # Add statistical test results
    for test in test_results:
        report['statistical_evidence'].append(test.to_dict())
    
    # Generate recommendations
    if len(changepoints) > 0:
        report['recommendations'].append(
            "Investigate external factors (product changes, AI search impact, "
            "policy changes) that coincide with detected structural shifts."
        )
        
        # Check for recent shifts
        if changepoints:
            latest_shift = max(changepoints, key=lambda cp: cp.date)
            days_since_shift = (date.today() - latest_shift.date).days
            if days_since_shift < 90:
                report['recommendations'].append(
                    f"Recent structural shift detected {days_since_shift} days ago. "
                    "Monitor closely to assess if this represents a permanent change."
                )
    
    return report
