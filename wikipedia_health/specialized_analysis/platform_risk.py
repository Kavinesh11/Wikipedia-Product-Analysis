"""Platform risk assessment module.

This module provides functions for assessing platform dependency risks,
including HHI calculation, threshold testing, and scenario analysis.
"""

from typing import Dict, List, Any, Tuple
from datetime import date
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy import stats

from wikipedia_health.models.data_models import TimeSeriesData, TestResult
from wikipedia_health.statistical_validation.hypothesis_tester import HypothesisTester
from wikipedia_health.statistical_validation.confidence_interval import ConfidenceIntervalCalculator


def assess_platform_risk(
    platform_data: Dict[str, TimeSeriesData],
    mobile_threshold: float = 0.70,
    high_concentration_hhi: float = 2500.0
) -> Dict[str, Any]:
    """Assess platform dependency risks with measurable thresholds.
    
    Calculates platform mix proportions, HHI (Herfindahl-Hirschman Index),
    CAGR for each platform, and performs threshold testing for mobile dependency.
    
    Args:
        platform_data: Dictionary mapping platform names to TimeSeriesData
                      Expected keys: 'desktop', 'mobile-web', 'mobile-app'
        mobile_threshold: Threshold for mobile dependency risk (default 0.70)
        high_concentration_hhi: HHI threshold for high concentration (default 2500)
    
    Returns:
        Dictionary containing:
            - platform_mix: Current platform proportions with confidence intervals
            - cagr: Compound annual growth rates for each platform
            - hhi: Herfindahl-Hirschman Index with risk classification
            - threshold_test: Test results for mobile dependency threshold
            - volatility: Coefficient of variation for each platform
            - scenario_analysis: Impact analysis for various decline scenarios
            - risk_report: Comprehensive risk assessment report
    
    Raises:
        ValueError: If platform_data is empty or missing required platforms
    """
    required_platforms = {'desktop', 'mobile-web', 'mobile-app'}
    if not platform_data:
        raise ValueError("platform_data cannot be empty")
    
    # Validate platforms
    available_platforms = set(platform_data.keys())
    if not required_platforms.issubset(available_platforms):
        missing = required_platforms - available_platforms
        raise ValueError(f"Missing required platforms: {missing}")
    
    # Initialize components
    ci_calculator = ConfidenceIntervalCalculator()
    tester = HypothesisTester()
    
    # Step 1: Calculate current platform mix with confidence intervals
    platform_mix = _calculate_platform_mix(platform_data, ci_calculator)
    
    # Step 2: Calculate CAGR for each platform
    cagr_results = _calculate_platform_cagr(platform_data, ci_calculator)
    
    # Step 3: Calculate HHI (Herfindahl-Hirschman Index)
    hhi_result = _calculate_hhi(platform_mix, high_concentration_hhi)
    
    # Step 4: Test mobile dependency threshold
    threshold_test = _test_mobile_threshold(
        platform_mix,
        mobile_threshold,
        ci_calculator
    )
    
    # Step 5: Calculate platform volatility (coefficient of variation)
    volatility = _calculate_platform_volatility(platform_data, tester)
    
    # Step 6: Perform scenario analysis
    scenario_analysis = _perform_scenario_analysis(
        platform_mix,
        decline_scenarios=[0.10, 0.20, 0.30]
    )
    
    # Step 7: Generate comprehensive risk report
    risk_report = _generate_risk_report(
        platform_mix=platform_mix,
        cagr=cagr_results,
        hhi=hhi_result,
        threshold_test=threshold_test,
        volatility=volatility,
        scenario_analysis=scenario_analysis
    )
    
    return {
        'platform_mix': platform_mix,
        'cagr': cagr_results,
        'hhi': hhi_result,
        'threshold_test': threshold_test,
        'volatility': volatility,
        'scenario_analysis': scenario_analysis,
        'risk_report': risk_report
    }


def threshold_testing(
    platform_proportions: Dict[str, float],
    threshold: float,
    confidence_intervals: Dict[str, Tuple[float, float]]
) -> TestResult:
    """Test whether platform dependency exceeds threshold.
    
    Args:
        platform_proportions: Dictionary of platform proportions
        threshold: Threshold value to test against
        confidence_intervals: Confidence intervals for each platform
    
    Returns:
        TestResult indicating whether threshold is exceeded
    """
    # Calculate total mobile proportion
    mobile_platforms = ['mobile-web', 'mobile-app']
    mobile_proportion = sum(
        platform_proportions.get(p, 0.0) for p in mobile_platforms
    )
    
    # Calculate combined confidence interval for mobile
    mobile_ci_lower = sum(
        confidence_intervals.get(p, (0, 0))[0] for p in mobile_platforms
    )
    mobile_ci_upper = sum(
        confidence_intervals.get(p, (0, 0))[1] for p in mobile_platforms
    )
    
    # Test if proportion exceeds threshold
    exceeds_threshold = mobile_proportion > threshold
    
    # Calculate probability of exceeding threshold using normal approximation
    # Assume normal distribution with mean = mobile_proportion
    se = (mobile_ci_upper - mobile_ci_lower) / (2 * 1.96)  # Back-calculate SE
    z_score = (mobile_proportion - threshold) / se if se > 0 else 0.0
    p_value = 1 - stats.norm.cdf(z_score) if z_score > 0 else stats.norm.cdf(z_score)
    
    # Effect size: distance from threshold in standard errors
    effect_size = abs(z_score)
    
    # Interpretation
    interpretation = (
        f"Mobile dependency test: Current mobile proportion is {mobile_proportion:.2%} "
        f"(95% CI: [{mobile_ci_lower:.2%}, {mobile_ci_upper:.2%}]). "
        f"Threshold: {threshold:.2%}. "
        f"{'EXCEEDS' if exceeds_threshold else 'DOES NOT EXCEED'} threshold "
        f"(p={p_value:.4f})."
    )
    
    return TestResult(
        test_name='Mobile Dependency Threshold Test',
        statistic=float(mobile_proportion),
        p_value=float(p_value),
        effect_size=float(effect_size),
        confidence_interval=(float(mobile_ci_lower), float(mobile_ci_upper)),
        is_significant=exceeds_threshold,
        alpha=0.05,
        interpretation=interpretation
    )


def scenario_analysis(
    platform_mix: Dict[str, float],
    dominant_platform: str,
    decline_percentages: List[float]
) -> Dict[str, Any]:
    """Perform scenario analysis for platform decline.
    
    Args:
        platform_mix: Current platform proportions
        dominant_platform: Name of dominant platform
        decline_percentages: List of decline percentages to analyze (e.g., [0.10, 0.20, 0.30])
    
    Returns:
        Dictionary with scenario analysis results
    """
    scenarios = {}
    
    current_value = platform_mix.get(dominant_platform, 0.0)
    
    for decline_pct in decline_percentages:
        declined_value = current_value * (1 - decline_pct)
        absolute_loss = current_value - declined_value
        
        scenarios[f"{decline_pct:.0%}_decline"] = {
            'current_proportion': current_value,
            'declined_proportion': declined_value,
            'absolute_loss': absolute_loss,
            'percentage_decline': decline_pct,
            'impact_description': (
                f"A {decline_pct:.0%} decline in {dominant_platform} would reduce "
                f"its proportion from {current_value:.2%} to {declined_value:.2%}, "
                f"representing a loss of {absolute_loss:.2%} of total traffic."
            )
        }
    
    return scenarios


def _calculate_platform_mix(
    platform_data: Dict[str, TimeSeriesData],
    ci_calculator: ConfidenceIntervalCalculator
) -> Dict[str, Dict[str, Any]]:
    """Calculate platform mix proportions with confidence intervals.
    
    Args:
        platform_data: Dictionary of platform time series data
        ci_calculator: Confidence interval calculator
    
    Returns:
        Dictionary with platform proportions and confidence intervals
    """
    # Calculate total traffic for most recent period
    recent_totals = {}
    for platform, ts_data in platform_data.items():
        # Use last 30 days average
        recent_values = ts_data.values.iloc[-30:]
        recent_totals[platform] = recent_values.mean()
    
    total_traffic = sum(recent_totals.values())
    
    # Calculate proportions and confidence intervals
    platform_mix = {}
    for platform, traffic in recent_totals.items():
        proportion = traffic / total_traffic if total_traffic > 0 else 0.0
        
        # Calculate confidence interval using bootstrap
        ts_data = platform_data[platform]
        recent_values = ts_data.values.iloc[-30:]
        
        # Bootstrap CI for proportion
        def proportion_stat(data):
            return data.mean() / total_traffic if total_traffic > 0 else 0.0
        
        ci_lower, ci_upper = ci_calculator.bootstrap_ci(
            recent_values,
            proportion_stat,
            confidence_level=0.95,
            n_bootstrap=1000
        )
        
        platform_mix[platform] = {
            'proportion': proportion,
            'confidence_interval': (ci_lower, ci_upper),
            'recent_average_traffic': traffic
        }
    
    return platform_mix


def _calculate_platform_cagr(
    platform_data: Dict[str, TimeSeriesData],
    ci_calculator: ConfidenceIntervalCalculator
) -> Dict[str, Dict[str, Any]]:
    """Calculate compound annual growth rate for each platform.
    
    Args:
        platform_data: Dictionary of platform time series data
        ci_calculator: Confidence interval calculator
    
    Returns:
        Dictionary with CAGR and confidence intervals for each platform
    """
    cagr_results = {}
    
    for platform, ts_data in platform_data.items():
        values = ts_data.values
        dates = pd.to_datetime(ts_data.date)
        
        # Calculate time span in years
        time_span_days = (dates.iloc[-1] - dates.iloc[0]).days
        time_span_years = time_span_days / 365.25
        
        if time_span_years < 0.1:  # Less than ~1 month
            cagr_results[platform] = {
                'cagr': 0.0,
                'confidence_interval': (0.0, 0.0),
                'time_span_years': time_span_years
            }
            continue
        
        # Calculate CAGR
        start_value = values.iloc[:30].mean()  # First 30 days average
        end_value = values.iloc[-30:].mean()   # Last 30 days average
        
        if start_value > 0:
            cagr = (np.power(end_value / start_value, 1 / time_span_years) - 1)
        else:
            cagr = 0.0
        
        # Bootstrap confidence interval for CAGR
        def cagr_stat(data):
            if len(data) < 2:
                return 0.0
            # Convert to pandas Series if it's a numpy array
            if isinstance(data, np.ndarray):
                data = pd.Series(data)
            start = data.iloc[:min(30, len(data)//2)].mean()
            end = data.iloc[-min(30, len(data)//2):].mean()
            if start > 0:
                return (np.power(end / start, 1 / time_span_years) - 1)
            return 0.0
        
        ci_lower, ci_upper = ci_calculator.bootstrap_ci(
            values,
            cagr_stat,
            confidence_level=0.95,
            n_bootstrap=1000
        )
        
        cagr_results[platform] = {
            'cagr': cagr,
            'confidence_interval': (ci_lower, ci_upper),
            'time_span_years': time_span_years,
            'start_value': start_value,
            'end_value': end_value
        }
    
    return cagr_results


def _calculate_hhi(
    platform_mix: Dict[str, Dict[str, Any]],
    high_concentration_threshold: float
) -> Dict[str, Any]:
    """Calculate Herfindahl-Hirschman Index for platform concentration.
    
    Args:
        platform_mix: Platform proportions
        high_concentration_threshold: HHI threshold for high concentration
    
    Returns:
        Dictionary with HHI score and risk classification
    """
    # Calculate HHI: sum of squared market shares (in percentage points)
    hhi = sum(
        (data['proportion'] * 100) ** 2
        for data in platform_mix.values()
    )
    
    # Classify concentration risk
    if hhi < 1500:
        risk_level = 'low'
        description = 'Unconcentrated market - low platform dependency risk'
    elif hhi < 2500:
        risk_level = 'moderate'
        description = 'Moderately concentrated market - moderate platform dependency risk'
    else:
        risk_level = 'high'
        description = 'Highly concentrated market - high platform dependency risk'
    
    return {
        'hhi': hhi,
        'risk_level': risk_level,
        'description': description,
        'threshold': high_concentration_threshold,
        'exceeds_threshold': hhi > high_concentration_threshold
    }


def _test_mobile_threshold(
    platform_mix: Dict[str, Dict[str, Any]],
    threshold: float,
    ci_calculator: ConfidenceIntervalCalculator
) -> TestResult:
    """Test whether mobile dependency exceeds threshold.
    
    Args:
        platform_mix: Platform proportions with confidence intervals
        threshold: Mobile dependency threshold
        ci_calculator: Confidence interval calculator
    
    Returns:
        TestResult for threshold test
    """
    # Extract proportions and CIs
    proportions = {
        platform: data['proportion']
        for platform, data in platform_mix.items()
    }
    
    confidence_intervals = {
        platform: data['confidence_interval']
        for platform, data in platform_mix.items()
    }
    
    return threshold_testing(proportions, threshold, confidence_intervals)


def _calculate_platform_volatility(
    platform_data: Dict[str, TimeSeriesData],
    tester: HypothesisTester
) -> Dict[str, Any]:
    """Calculate coefficient of variation for each platform.
    
    Args:
        platform_data: Dictionary of platform time series data
        tester: Hypothesis tester for comparing volatilities
    
    Returns:
        Dictionary with volatility metrics and comparison tests
    """
    volatility = {}
    cv_values = []
    platform_names = []
    
    for platform, ts_data in platform_data.items():
        values = ts_data.values
        
        # Calculate coefficient of variation
        mean_val = values.mean()
        std_val = values.std()
        cv = (std_val / mean_val) if mean_val > 0 else 0.0
        
        volatility[platform] = {
            'coefficient_of_variation': cv,
            'mean': mean_val,
            'std': std_val
        }
        
        cv_values.append(cv)
        platform_names.append(platform)
    
    # Test if volatilities differ significantly
    # Use ANOVA on absolute deviations from mean as proxy for variance comparison
    groups = [
        np.abs(platform_data[p].values - platform_data[p].values.mean())
        for p in platform_names
    ]
    
    volatility_test = tester.anova(groups)
    
    return {
        'by_platform': volatility,
        'comparison_test': volatility_test.to_dict()
    }


def _perform_scenario_analysis(
    platform_mix: Dict[str, Dict[str, Any]],
    decline_scenarios: List[float]
) -> Dict[str, Any]:
    """Perform scenario analysis for platform declines.
    
    Args:
        platform_mix: Current platform proportions
        decline_scenarios: List of decline percentages
    
    Returns:
        Dictionary with scenario analysis results
    """
    # Identify dominant platform
    dominant_platform = max(
        platform_mix.items(),
        key=lambda x: x[1]['proportion']
    )[0]
    
    # Extract proportions
    proportions = {
        platform: data['proportion']
        for platform, data in platform_mix.items()
    }
    
    # Run scenario analysis
    scenarios = scenario_analysis(
        proportions,
        dominant_platform,
        decline_scenarios
    )
    
    return {
        'dominant_platform': dominant_platform,
        'scenarios': scenarios
    }


def _generate_risk_report(
    platform_mix: Dict[str, Dict[str, Any]],
    cagr: Dict[str, Dict[str, Any]],
    hhi: Dict[str, Any],
    threshold_test: TestResult,
    volatility: Dict[str, Any],
    scenario_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate comprehensive platform risk assessment report.
    
    Args:
        platform_mix: Platform proportions
        cagr: Growth rate analysis
        hhi: HHI analysis
        threshold_test: Threshold test results
        volatility: Volatility analysis
        scenario_analysis: Scenario analysis results
    
    Returns:
        Comprehensive risk report dictionary
    """
    report = {
        'executive_summary': {
            'overall_risk_level': hhi['risk_level'],
            'hhi_score': hhi['hhi'],
            'mobile_dependency_exceeds_threshold': threshold_test.is_significant,
            'dominant_platform': scenario_analysis['dominant_platform']
        },
        'current_platform_mix': {},
        'growth_trends': {},
        'concentration_risk': hhi,
        'threshold_analysis': threshold_test.to_dict(),
        'volatility_analysis': volatility,
        'scenario_impacts': scenario_analysis['scenarios'],
        'recommendations': []
    }
    
    # Add platform mix details
    for platform, data in platform_mix.items():
        report['current_platform_mix'][platform] = {
            'proportion': f"{data['proportion']:.2%}",
            'confidence_interval': (
                f"{data['confidence_interval'][0]:.2%}",
                f"{data['confidence_interval'][1]:.2%}"
            )
        }
    
    # Add growth trends
    for platform, data in cagr.items():
        report['growth_trends'][platform] = {
            'cagr': f"{data['cagr']:.2%}",
            'confidence_interval': (
                f"{data['confidence_interval'][0]:.2%}",
                f"{data['confidence_interval'][1]:.2%}"
            )
        }
    
    # Generate recommendations
    if hhi['risk_level'] == 'high':
        report['recommendations'].append(
            "HIGH RISK: Platform concentration is high. Diversify traffic sources "
            "and reduce dependency on dominant platform."
        )
    
    if threshold_test.is_significant:
        report['recommendations'].append(
            "Mobile dependency exceeds 70% threshold. Monitor mobile platform "
            "stability and develop contingency plans for mobile disruptions."
        )
    
    # Check for declining platforms
    for platform, data in cagr.items():
        if data['cagr'] < -0.05:  # Declining more than 5% per year
            report['recommendations'].append(
                f"Platform '{platform}' is declining at {data['cagr']:.1%} CAGR. "
                "Investigate causes and consider intervention strategies."
            )
    
    return report
