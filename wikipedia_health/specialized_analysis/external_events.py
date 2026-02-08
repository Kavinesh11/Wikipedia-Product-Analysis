"""External event analysis module.

This module provides functions for analyzing Wikipedia's response to external
shocks using event study methodology and comparing responses across event categories.
"""

from typing import Dict, List, Any, Tuple, Optional
from datetime import date, timedelta
import pandas as pd
from pandas import Series
import numpy as np
from scipy import stats

from wikipedia_health.models.data_models import (
    TimeSeriesData,
    TestResult,
    ForecastResult
)
from wikipedia_health.causal_inference.event_study import EventStudyAnalyzer
from wikipedia_health.statistical_validation.hypothesis_tester import HypothesisTester


def analyze_external_event(
    time_series: TimeSeriesData,
    event_date: date,
    event_name: str,
    event_category: str,
    baseline_window_days: int = 90,
    post_window_days: int = 30
) -> Dict[str, Any]:
    """Analyze external event impact using event study methodology.
    
    Implements event study methodology with baseline forecasting to measure
    how Wikipedia responds to external shocks compared to baseline predictions.
    
    Args:
        time_series: Time series data
        event_date: Date of external event
        event_name: Name/description of event
        event_category: Category (political, natural_disaster, celebrity, scientific)
        baseline_window_days: Days before event to use for baseline model
        post_window_days: Days after event to analyze
    
    Returns:
        Dictionary containing:
            - baseline_forecast: Baseline prediction with intervals
            - observed_values: Actual observed values
            - cumulative_abnormal_return: CAR with significance
            - peak_impact: Peak impact magnitude and timing
            - decay_analysis: Half-life of traffic decay
            - event_report: Comprehensive event impact report
    
    Raises:
        ValueError: If time series is too short or dates are invalid
    """
    # Initialize analyzer
    event_analyzer = EventStudyAnalyzer()
    
    # Convert to pandas Series with datetime index
    series = pd.Series(
        time_series.values.values,
        index=pd.to_datetime(time_series.date.values)
    )
    
    # Step 1: Fit baseline model
    baseline_model = event_analyzer.fit_baseline(
        series,
        event_date,
        baseline_window=baseline_window_days
    )
    
    # Step 2: Estimate event impact
    event_impact = event_analyzer.estimate_event_impact(
        series,
        baseline_model,
        event_date,
        post_window=post_window_days
    )
    
    # Step 3: Test significance
    is_significant, p_value = event_analyzer.test_significance(
        event_impact,
        alpha=0.05
    )
    
    # Step 4: Calculate cumulative abnormal return (CAR)
    car_result = _calculate_car(
        observed=event_impact.observed,
        predicted=event_impact.predicted,
        prediction_interval=event_impact.prediction_interval
    )
    
    # Step 5: Identify peak impact
    peak_impact = _identify_peak_impact(
        observed=event_impact.observed,
        predicted=event_impact.predicted,
        event_date=event_date
    )
    
    # Step 6: Measure decay half-life
    decay_analysis = event_analyzer.measure_persistence(
        series,
        event_date,
        max_window=min(post_window_days * 2, 180)
    )
    
    # Step 7: Generate comprehensive event report
    event_report = _generate_event_report(
        time_series=time_series,
        event_name=event_name,
        event_category=event_category,
        event_date=event_date,
        event_impact=event_impact,
        car_result=car_result,
        peak_impact=peak_impact,
        decay_analysis=decay_analysis,
        is_significant=is_significant,
        p_value=p_value
    )
    
    return {
        'baseline_forecast': {
            'predicted': event_impact.predicted,
            'prediction_interval': event_impact.prediction_interval
        },
        'observed_values': event_impact.observed,
        'cumulative_abnormal_return': car_result,
        'peak_impact': peak_impact,
        'decay_analysis': decay_analysis,
        'significance_test': {
            'is_significant': is_significant,
            'p_value': p_value
        },
        'event_report': event_report
    }


def event_category_comparison(
    event_results: List[Dict[str, Any]],
    event_categories: List[str]
) -> Dict[str, Any]:
    """Compare response magnitudes across event categories using ANOVA.
    
    Tests whether response magnitudes differ significantly across event
    categories (political, natural disaster, celebrity, scientific).
    
    Args:
        event_results: List of event analysis results
        event_categories: List of event categories for each result
    
    Returns:
        Dictionary with ANOVA results and category comparisons
    
    Raises:
        ValueError: If fewer than 2 categories or mismatched lengths
    """
    if len(event_results) != len(event_categories):
        raise ValueError("Number of event results must match number of categories")
    
    # Group results by category
    category_impacts = {}
    for result, category in zip(event_results, event_categories):
        if category not in category_impacts:
            category_impacts[category] = []
        
        # Extract CAR (cumulative abnormal return) as impact measure
        car = result.get('cumulative_abnormal_return', {}).get('car', 0.0)
        category_impacts[category].append(car)
    
    if len(category_impacts) < 2:
        raise ValueError("Need at least 2 different categories for comparison")
    
    # Convert to Series for each category
    category_series = {
        cat: pd.Series(impacts)
        for cat, impacts in category_impacts.items()
    }
    
    # Perform ANOVA
    tester = HypothesisTester()
    anova_result = tester.anova(list(category_series.values()))
    
    # Calculate descriptive statistics for each category
    category_stats = {}
    for category, impacts in category_impacts.items():
        impacts_array = np.array(impacts)
        category_stats[category] = {
            'mean_impact': float(np.mean(impacts_array)),
            'median_impact': float(np.median(impacts_array)),
            'std_impact': float(np.std(impacts_array)),
            'count': len(impacts_array)
        }
    
    # Perform pairwise comparisons
    pairwise_comparisons = {}
    categories = list(category_series.keys())
    
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            cat1, cat2 = categories[i], categories[j]
            
            # T-test for pairwise comparison
            comparison_result = tester.t_test(
                category_series[cat1],
                category_series[cat2]
            )
            
            pairwise_comparisons[f"{cat1}_vs_{cat2}"] = comparison_result.to_dict()
    
    # Generate comparison report
    comparison_report = {
        'anova_result': anova_result.to_dict(),
        'category_statistics': category_stats,
        'pairwise_comparisons': pairwise_comparisons,
        'summary': {
            'num_categories': len(category_impacts),
            'total_events': len(event_results),
            'significant_difference': anova_result.is_significant,
            'highest_impact_category': max(
                category_stats.items(),
                key=lambda x: x[1]['mean_impact']
            )[0],
            'lowest_impact_category': min(
                category_stats.items(),
                key=lambda x: x[1]['mean_impact']
            )[0]
        }
    }
    
    return comparison_report


# Helper functions

def _calculate_car(
    observed: Series,
    predicted: Series,
    prediction_interval: Tuple[Series, Series]
) -> Dict[str, Any]:
    """Calculate cumulative abnormal return (CAR)."""
    # Calculate abnormal returns (observed - predicted)
    abnormal_returns = observed.values - predicted.values
    
    # Cumulative abnormal return
    car = np.sum(abnormal_returns)
    
    # Calculate standard error of CAR
    se_car = np.sqrt(len(abnormal_returns)) * np.std(abnormal_returns)
    
    # Z-score for CAR
    z_score = car / se_car if se_car > 0 else 0.0
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Check if observed values exceed prediction interval
    lower_bound, upper_bound = prediction_interval
    exceeds_interval = np.sum(
        (observed.values < lower_bound.values) | (observed.values > upper_bound.values)
    )
    proportion_exceeds = exceeds_interval / len(observed)
    
    # Confidence interval for CAR
    ci_lower = car - 1.96 * se_car
    ci_upper = car + 1.96 * se_car
    
    return {
        'car': float(car),
        'se_car': float(se_car),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'confidence_interval': (float(ci_lower), float(ci_upper)),
        'is_significant': bool(p_value < 0.05),
        'proportion_exceeds_interval': float(proportion_exceeds),
        'interpretation': (
            f"Cumulative abnormal return: {car:.2f} "
            f"(95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]). "
            f"{'Significant' if p_value < 0.05 else 'Not significant'} "
            f"(p={p_value:.4f}). "
            f"{proportion_exceeds:.1%} of observations exceed 95% prediction interval."
        )
    }


def _identify_peak_impact(
    observed: Series,
    predicted: Series,
    event_date: date
) -> Dict[str, Any]:
    """Identify peak impact magnitude and timing."""
    # Calculate deviations
    deviations = observed.values - predicted.values
    
    # Find peak (maximum absolute deviation)
    peak_idx = np.argmax(np.abs(deviations))
    peak_value = deviations[peak_idx]
    peak_date = observed.index[peak_idx].date()
    
    # Days from event to peak
    days_to_peak = (peak_date - event_date).days
    
    # Percentage deviation
    predicted_at_peak = predicted.values[peak_idx]
    percentage_deviation = (peak_value / predicted_at_peak * 100) if predicted_at_peak != 0 else 0.0
    
    return {
        'peak_magnitude': float(peak_value),
        'peak_date': peak_date,
        'days_to_peak': int(days_to_peak),
        'percentage_deviation': float(percentage_deviation),
        'observed_at_peak': float(observed.values[peak_idx]),
        'predicted_at_peak': float(predicted_at_peak)
    }


def _generate_event_report(
    time_series: TimeSeriesData,
    event_name: str,
    event_category: str,
    event_date: date,
    event_impact: Any,
    car_result: Dict[str, Any],
    peak_impact: Dict[str, Any],
    decay_analysis: int,
    is_significant: bool,
    p_value: float
) -> Dict[str, Any]:
    """Generate comprehensive event impact report."""
    report = {
        'event_info': {
            'name': event_name,
            'category': event_category,
            'date': event_date,
            'metric_type': time_series.metric_type,
            'platform': time_series.platform
        },
        'impact_summary': {
            'significant': is_significant,
            'p_value': p_value,
            'cumulative_abnormal_return': car_result['car'],
            'car_confidence_interval': car_result['confidence_interval']
        },
        'peak_impact': peak_impact,
        'decay_analysis': {
            'half_life_days': decay_analysis,
            'interpretation': (
                f"Traffic returned to baseline within {decay_analysis} days" 
                if decay_analysis > 0 else
                "Traffic did not return to baseline within analysis window"
            )
        },
        'detailed_statistics': {
            'car_statistics': car_result,
            'proportion_exceeds_interval': car_result['proportion_exceeds_interval']
        },
        'recommendations': []
    }
    
    # Generate recommendations
    if is_significant:
        if car_result['car'] > 0:
            report['recommendations'].append(
                f"Event '{event_name}' caused a significant INCREASE in traffic "
                f"(CAR: {car_result['car']:.0f}). Wikipedia serves as an important "
                "information source during this type of event."
            )
        else:
            report['recommendations'].append(
                f"Event '{event_name}' caused a significant DECREASE in traffic "
                f"(CAR: {car_result['car']:.0f}). Investigate potential causes."
            )
    else:
        report['recommendations'].append(
            f"Event '{event_name}' did not have a statistically significant "
            "impact on traffic."
        )
    
    # Add category-specific insights
    if event_category == 'political':
        report['recommendations'].append(
            "Political events typically drive traffic to related articles. "
            "Monitor for sustained interest or quick decay."
        )
    elif event_category == 'natural_disaster':
        report['recommendations'].append(
            "Natural disasters often cause immediate traffic spikes. "
            "Ensure article quality and update frequency during such events."
        )
    
    return report
