"""Campaign effectiveness analysis module.

This module provides functions for evaluating campaign effectiveness using
causal inference methods, duration analysis, and cross-campaign comparisons.
"""

from typing import Dict, List, Any, Tuple, Optional
from datetime import date, timedelta
import pandas as pd
from pandas import Series
import numpy as np
from scipy import stats

from wikipedia_health.models.data_models import (
    TimeSeriesData,
    CausalEffect,
    TestResult
)
from wikipedia_health.causal_inference.interrupted_time_series import (
    InterruptedTimeSeriesAnalyzer
)
from wikipedia_health.causal_inference.synthetic_control import SyntheticControlBuilder
from wikipedia_health.statistical_validation.hypothesis_tester import HypothesisTester


def evaluate_campaign(
    time_series: TimeSeriesData,
    campaign_start_date: date,
    campaign_end_date: Optional[date] = None,
    donor_series: Optional[List[TimeSeriesData]] = None,
    pre_period_days: int = 90,
    post_period_days: int = 90
) -> Dict[str, Any]:
    """Evaluate campaign effectiveness using ITSA and synthetic controls.
    
    Orchestrates interrupted time series analysis and synthetic control methods
    to measure campaign impact with statistical rigor.
    
    Args:
        time_series: Time series data for treated unit
        campaign_start_date: Date when campaign started
        campaign_end_date: Date when campaign ended (optional)
        donor_series: List of control time series for synthetic control (optional)
        pre_period_days: Days before campaign to use for baseline
        post_period_days: Days after campaign to analyze
    
    Returns:
        Dictionary containing:
            - itsa_result: Interrupted time series analysis result
            - synthetic_control_result: Synthetic control result (if donors provided)
            - duration_analysis: Immediate/short/long-term effects
            - permutation_test: Permutation test for significance
            - campaign_report: Comprehensive campaign evaluation report
    
    Raises:
        ValueError: If time series is too short or dates are invalid
    """
    # Initialize analyzers
    itsa_analyzer = InterruptedTimeSeriesAnalyzer(min_pre_period=60)
    
    # Convert to pandas Series with datetime index
    series = pd.Series(
        time_series.values.values,
        index=pd.to_datetime(time_series.date.values)
    )
    
    # Step 1: Perform ITSA
    itsa_model = itsa_analyzer.fit(
        series,
        campaign_start_date,
        pre_period_length=pre_period_days
    )
    
    itsa_effect = itsa_analyzer.estimate_effect(
        itsa_model,
        post_period_length=post_period_days
    )
    
    # Step 2: Perform synthetic control if donors provided
    synthetic_control_result = None
    if donor_series and len(donor_series) > 0:
        sc_builder = SyntheticControlBuilder()
        
        # Convert donor series to pandas Series
        donor_series_list = [
            pd.Series(
                ts.values.values,
                index=pd.to_datetime(ts.date.values)
            )
            for ts in donor_series
        ]
        
        # Define pre-period
        pre_start = campaign_start_date - timedelta(days=pre_period_days)
        pre_period = (pre_start, campaign_start_date)
        
        # Construct synthetic control
        synthetic_control = sc_builder.construct_synthetic_control(
            treated_unit=series,
            donor_pool=donor_series_list,
            pre_period=pre_period
        )
        
        # Estimate effect
        post_end = campaign_start_date + timedelta(days=post_period_days)
        post_period = (campaign_start_date, post_end)
        
        synthetic_control_result = sc_builder.estimate_effect(
            treated=series,
            synthetic=synthetic_control,
            post_period=post_period
        )
    
    # Step 3: Duration analysis
    duration_analysis_result = duration_analysis(
        time_series=time_series,
        campaign_start_date=campaign_start_date,
        itsa_model=itsa_model
    )
    
    # Step 4: Permutation test for robustness
    permutation_test_result = _permutation_test_campaign(
        series=series,
        campaign_start_date=campaign_start_date,
        observed_effect=itsa_effect.effect_size,
        n_permutations=1000
    )
    
    # Step 5: Generate comprehensive campaign report
    campaign_report = _generate_campaign_report(
        time_series=time_series,
        campaign_start_date=campaign_start_date,
        campaign_end_date=campaign_end_date,
        itsa_result=itsa_effect,
        synthetic_control_result=synthetic_control_result,
        duration_analysis=duration_analysis_result,
        permutation_test=permutation_test_result
    )
    
    return {
        'itsa_result': itsa_effect,
        'synthetic_control_result': synthetic_control_result,
        'duration_analysis': duration_analysis_result,
        'permutation_test': permutation_test_result,
        'campaign_report': campaign_report
    }


def duration_analysis(
    time_series: TimeSeriesData,
    campaign_start_date: date,
    itsa_model: Any
) -> Dict[str, Any]:
    """Analyze campaign effects across different time windows.
    
    Distinguishes between immediate (0-7 days), short-term (8-30 days),
    and long-term (30+ days) effects with separate effect size estimates.
    
    Args:
        time_series: Time series data
        campaign_start_date: Campaign start date
        itsa_model: Fitted ITSA model
    
    Returns:
        Dictionary with effect sizes for each time window
    """
    # Convert to pandas Series
    series = pd.Series(
        time_series.values.values,
        index=pd.to_datetime(time_series.date.values)
    )
    
    campaign_ts = pd.Timestamp(campaign_start_date)
    post_data = series[series.index >= campaign_ts]
    
    if len(post_data) == 0:
        raise ValueError("No post-campaign data available")
    
    # Initialize analyzer
    itsa_analyzer = InterruptedTimeSeriesAnalyzer()
    
    # Define time windows
    windows = {
        'immediate': (0, 7),      # 0-7 days
        'short_term': (8, 30),    # 8-30 days
        'long_term': (31, 90)     # 31-90 days
    }
    
    effects_by_window = {}
    
    for window_name, (start_day, end_day) in windows.items():
        # Get data for this window
        window_start = campaign_ts + timedelta(days=start_day)
        window_end = campaign_ts + timedelta(days=end_day)
        
        window_data = series[
            (series.index >= window_start) & (series.index <= window_end)
        ]
        
        if len(window_data) == 0:
            effects_by_window[window_name] = {
                'effect_size': 0.0,
                'confidence_interval': (0.0, 0.0),
                'p_value': 1.0,
                'num_observations': 0,
                'available': False
            }
            continue
        
        # Construct counterfactual for this window
        counterfactual = itsa_analyzer.construct_counterfactual(
            itsa_model,
            (window_start.date(), window_end.date())
        )
        
        # Calculate effect
        treatment_effects = window_data.values - counterfactual.values
        effect_size = np.mean(treatment_effects)
        
        # Calculate confidence interval
        se = stats.sem(treatment_effects)
        ci_lower = effect_size - 1.96 * se
        ci_upper = effect_size + 1.96 * se
        
        # Calculate p-value
        if len(treatment_effects) > 1:
            t_stat, p_value = stats.ttest_1samp(treatment_effects, 0)
        else:
            p_value = 1.0
        
        effects_by_window[window_name] = {
            'effect_size': float(effect_size),
            'confidence_interval': (float(ci_lower), float(ci_upper)),
            'p_value': float(p_value),
            'num_observations': len(window_data),
            'available': True,
            'window_days': (start_day, end_day)
        }
    
    return {
        'effects_by_window': effects_by_window,
        'summary': _summarize_duration_effects(effects_by_window)
    }


def cross_campaign_comparison(
    campaign_results: List[Dict[str, Any]],
    campaign_names: List[str]
) -> Dict[str, Any]:
    """Compare effect sizes across multiple campaigns using meta-analysis.
    
    Tests whether effect sizes differ significantly across campaigns using
    ANOVA or hierarchical modeling.
    
    Args:
        campaign_results: List of campaign evaluation results
        campaign_names: Names of campaigns for labeling
    
    Returns:
        Dictionary with comparison test results and meta-analysis
    
    Raises:
        ValueError: If fewer than 2 campaigns provided
    """
    if len(campaign_results) < 2:
        raise ValueError("Need at least 2 campaigns for comparison")
    
    if len(campaign_names) != len(campaign_results):
        raise ValueError("Number of campaign names must match number of results")
    
    # Extract effect sizes and standard errors
    effect_sizes = []
    standard_errors = []
    
    for result in campaign_results:
        # Get ITSA result
        itsa_result = result.get('itsa_result')
        if itsa_result is None:
            raise ValueError("Campaign result missing itsa_result")
        
        effect_size = itsa_result.effect_size
        ci_lower, ci_upper = itsa_result.confidence_interval
        
        # Calculate standard error from CI
        se = (ci_upper - ci_lower) / (2 * 1.96)
        
        effect_sizes.append(effect_size)
        standard_errors.append(se)
    
    effect_sizes = np.array(effect_sizes)
    standard_errors = np.array(standard_errors)
    
    # Perform ANOVA-like test using weighted means
    # Weight by inverse variance
    weights = 1 / (standard_errors ** 2)
    weights = weights / np.sum(weights)  # Normalize
    
    # Weighted mean effect
    pooled_effect = np.sum(weights * effect_sizes)
    
    # Test heterogeneity using Q-statistic
    Q = np.sum(weights * (effect_sizes - pooled_effect) ** 2)
    df = len(effect_sizes) - 1
    p_value_heterogeneity = 1 - stats.chi2.cdf(Q, df) if df > 0 else 1.0
    
    # I-squared statistic (proportion of variance due to heterogeneity)
    I_squared = max(0, (Q - df) / Q) if Q > 0 else 0.0
    
    # Perform pairwise comparisons
    tester = HypothesisTester()
    pairwise_comparisons = {}
    
    for i in range(len(campaign_names)):
        for j in range(i + 1, len(campaign_names)):
            # Z-test for difference in effects
            diff = effect_sizes[i] - effect_sizes[j]
            se_diff = np.sqrt(standard_errors[i]**2 + standard_errors[j]**2)
            
            if se_diff > 0:
                z_stat = diff / se_diff
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = 0.0
                p_value = 1.0
            
            comparison_key = f"{campaign_names[i]}_vs_{campaign_names[j]}"
            pairwise_comparisons[comparison_key] = {
                'difference': float(diff),
                'se_difference': float(se_diff),
                'z_statistic': float(z_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
    
    # Generate comparison report
    comparison_report = {
        'num_campaigns': len(campaign_results),
        'campaign_names': campaign_names,
        'individual_effects': [
            {
                'campaign': campaign_names[i],
                'effect_size': float(effect_sizes[i]),
                'standard_error': float(standard_errors[i])
            }
            for i in range(len(campaign_names))
        ],
        'meta_analysis': {
            'pooled_effect': float(pooled_effect),
            'Q_statistic': float(Q),
            'p_value_heterogeneity': float(p_value_heterogeneity),
            'I_squared': float(I_squared),
            'heterogeneity_interpretation': (
                'High heterogeneity' if I_squared > 0.75 else
                'Moderate heterogeneity' if I_squared > 0.50 else
                'Low heterogeneity'
            )
        },
        'pairwise_comparisons': pairwise_comparisons
    }
    
    return comparison_report


# Helper functions

def _permutation_test_campaign(
    series: Series,
    campaign_start_date: date,
    observed_effect: float,
    n_permutations: int = 1000
) -> TestResult:
    """Perform permutation test for campaign effect significance."""
    campaign_ts = pd.Timestamp(campaign_start_date)
    
    # Get pre and post data
    pre_data = series[series.index < campaign_ts]
    post_data = series[series.index >= campaign_ts]
    
    # Combine data
    combined = pd.concat([pre_data, post_data])
    n_post = len(post_data)
    
    # Perform permutations
    permuted_effects = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_permutations):
        # Randomly assign observations to "post" period
        shuffled_indices = rng.permutation(len(combined))
        pseudo_post_indices = shuffled_indices[:n_post]
        pseudo_pre_indices = shuffled_indices[n_post:]
        
        pseudo_post = combined.iloc[pseudo_post_indices]
        pseudo_pre = combined.iloc[pseudo_pre_indices]
        
        # Calculate pseudo effect
        pseudo_effect = pseudo_post.mean() - pseudo_pre.mean()
        permuted_effects.append(pseudo_effect)
    
    permuted_effects = np.array(permuted_effects)
    
    # Calculate p-value
    p_value = np.mean(np.abs(permuted_effects) >= np.abs(observed_effect))
    
    # Calculate confidence interval from permutation distribution
    ci_lower = np.percentile(permuted_effects, 2.5)
    ci_upper = np.percentile(permuted_effects, 97.5)
    
    interpretation = (
        f"Permutation test ({n_permutations} permutations): "
        f"Observed effect ({observed_effect:.2f}) is "
        f"{'significant' if p_value < 0.05 else 'not significant'} "
        f"(p={p_value:.4f})."
    )
    
    return TestResult(
        test_name='Campaign Permutation Test',
        statistic=float(observed_effect),
        p_value=float(p_value),
        effect_size=float(observed_effect / np.std(permuted_effects)) if np.std(permuted_effects) > 0 else 0.0,
        confidence_interval=(float(ci_lower), float(ci_upper)),
        is_significant=bool(p_value < 0.05),
        alpha=0.05,
        interpretation=interpretation
    )


def _summarize_duration_effects(
    effects_by_window: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Summarize duration analysis results."""
    summary = {
        'immediate_effect_significant': False,
        'short_term_effect_significant': False,
        'long_term_effect_significant': False,
        'effect_persistence': 'none'
    }
    
    # Check significance for each window
    for window_name in ['immediate', 'short_term', 'long_term']:
        if window_name in effects_by_window:
            window_data = effects_by_window[window_name]
            if window_data.get('available', False):
                is_sig = window_data['p_value'] < 0.05
                summary[f'{window_name}_effect_significant'] = is_sig
    
    # Determine persistence
    if summary['long_term_effect_significant']:
        summary['effect_persistence'] = 'long-term'
    elif summary['short_term_effect_significant']:
        summary['effect_persistence'] = 'short-term'
    elif summary['immediate_effect_significant']:
        summary['effect_persistence'] = 'immediate-only'
    
    return summary


def _generate_campaign_report(
    time_series: TimeSeriesData,
    campaign_start_date: date,
    campaign_end_date: Optional[date],
    itsa_result: CausalEffect,
    synthetic_control_result: Optional[CausalEffect],
    duration_analysis: Dict[str, Any],
    permutation_test: TestResult
) -> Dict[str, Any]:
    """Generate comprehensive campaign evaluation report."""
    report = {
        'campaign_info': {
            'metric_type': time_series.metric_type,
            'platform': time_series.platform,
            'start_date': campaign_start_date,
            'end_date': campaign_end_date
        },
        'overall_effect': {
            'method': 'ITSA',
            'effect_size': itsa_result.effect_size,
            'percentage_effect': itsa_result.percentage_effect(),
            'confidence_interval': itsa_result.confidence_interval,
            'p_value': itsa_result.p_value,
            'significant': itsa_result.p_value < 0.05
        },
        'duration_effects': duration_analysis,
        'robustness_checks': {
            'permutation_test': permutation_test.to_dict()
        },
        'recommendations': []
    }
    
    # Add synthetic control results if available
    if synthetic_control_result:
        report['synthetic_control_validation'] = {
            'effect_size': synthetic_control_result.effect_size,
            'percentage_effect': synthetic_control_result.percentage_effect(),
            'confidence_interval': synthetic_control_result.confidence_interval,
            'p_value': synthetic_control_result.p_value
        }
    
    # Generate recommendations
    if itsa_result.p_value < 0.05:
        if itsa_result.effect_size > 0:
            report['recommendations'].append(
                f"Campaign had a significant POSITIVE effect "
                f"({itsa_result.percentage_effect():.1f}% increase). "
                "Consider replicating this campaign approach."
            )
        else:
            report['recommendations'].append(
                f"Campaign had a significant NEGATIVE effect "
                f"({itsa_result.percentage_effect():.1f}% decrease). "
                "Review campaign strategy and avoid similar approaches."
            )
    else:
        report['recommendations'].append(
            "Campaign effect was not statistically significant. "
            "Consider alternative strategies or longer campaign duration."
        )
    
    # Add duration-specific recommendations
    duration_summary = duration_analysis.get('summary', {})
    persistence = duration_summary.get('effect_persistence', 'none')
    
    if persistence == 'immediate-only':
        report['recommendations'].append(
            "Campaign effect was immediate but did not persist. "
            "Consider strategies to sustain engagement beyond initial impact."
        )
    elif persistence == 'long-term':
        report['recommendations'].append(
            "Campaign effect persisted long-term. This indicates a successful "
            "intervention with lasting impact."
        )
    
    return report
