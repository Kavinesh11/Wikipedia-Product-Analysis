#!/usr/bin/env python3
"""Custom analysis example.

This script demonstrates advanced customization:
1. Custom statistical tests
2. Custom causal inference methods
3. Custom visualization
4. Integration with external data

Usage:
    python examples/scripts/custom_analysis.py
"""

from datetime import date
import pandas as pd
import numpy as np

from wikipedia_health.config import load_config
from wikipedia_health.data_acquisition import WikimediaAPIClient
from wikipedia_health.statistical_validation import HypothesisTester, EffectSizeCalculator
from wikipedia_health.time_series import TimeSeriesDecomposer, ChangepointDetector
from wikipedia_health.causal_inference import InterruptedTimeSeriesAnalyzer
from wikipedia_health.visualization import plot_trend_with_confidence_bands


def custom_statistical_test(data: pd.DataFrame) -> dict:
    """Perform custom statistical analysis.
    
    This example shows how to use individual components for custom analysis.
    
    Args:
        data: DataFrame with time series data.
        
    Returns:
        Dictionary with test results.
    """
    print("Running custom statistical tests...")
    
    # Split data into two periods
    midpoint = len(data) // 2
    period1 = data.iloc[:midpoint]['pageviews']
    period2 = data.iloc[midpoint:]['pageviews']
    
    # Initialize tester
    tester = HypothesisTester()
    
    # Perform t-test
    t_test_result = tester.t_test(period1, period2, alternative='two-sided')
    print(f"  T-test: t={t_test_result.statistic:.3f}, p={t_test_result.p_value:.4f}")
    
    # Perform Mann-Whitney U test (non-parametric alternative)
    mw_result = tester.mann_whitney(period1, period2)
    print(f"  Mann-Whitney U: U={mw_result.statistic:.3f}, p={mw_result.p_value:.4f}")
    
    # Calculate effect size
    effect_calc = EffectSizeCalculator()
    cohens_d = effect_calc.cohens_d(period1, period2)
    print(f"  Cohen's d: {cohens_d:.3f}")
    
    return {
        't_test': t_test_result,
        'mann_whitney': mw_result,
        'cohens_d': cohens_d
    }


def custom_changepoint_analysis(data: pd.DataFrame) -> list:
    """Perform custom changepoint detection.
    
    This example shows how to use multiple changepoint detection methods
    and combine their results.
    
    Args:
        data: DataFrame with time series data.
        
    Returns:
        List of detected changepoints.
    """
    print("Running custom changepoint detection...")
    
    detector = ChangepointDetector()
    series = data['pageviews']
    
    # Try multiple methods
    pelt_changepoints = detector.detect_pelt(series, penalty=None, min_size=30)
    print(f"  PELT detected {len(pelt_changepoints)} changepoints")
    
    binseg_changepoints = detector.detect_binary_segmentation(series, n_changepoints=5)
    print(f"  Binary Segmentation detected {len(binseg_changepoints)} changepoints")
    
    # Find consensus changepoints (detected by multiple methods)
    consensus = []
    for pelt_cp in pelt_changepoints:
        for binseg_cp in binseg_changepoints:
            # If changepoints are within 7 days of each other
            if abs((pelt_cp.date - binseg_cp.date).days) <= 7:
                # Test significance
                is_sig, p_value = detector.test_significance(series, pelt_cp)
                if is_sig:
                    consensus.append(pelt_cp)
                    print(f"  Consensus changepoint: {pelt_cp.date} (p={p_value:.4f})")
                break
    
    return consensus


def custom_causal_analysis(
    data: pd.DataFrame,
    intervention_date: date
) -> dict:
    """Perform custom causal inference analysis.
    
    This example shows how to use causal inference components with
    custom pre/post periods and validation.
    
    Args:
        data: DataFrame with time series data.
        intervention_date: Date of intervention.
        
    Returns:
        Dictionary with causal effect estimates.
    """
    print(f"Running custom causal analysis for intervention on {intervention_date}...")
    
    # Initialize analyzer
    analyzer = InterruptedTimeSeriesAnalyzer()
    
    # Fit model with custom pre-period length
    series = data.set_index('date')['pageviews']
    model = analyzer.fit(series, intervention_date, pre_period_length=120)
    
    # Estimate effect with custom post-period
    effect = analyzer.estimate_effect(model, post_period_length=60)
    print(f"  Effect size: {effect.effect_size:+.2%}")
    print(f"  95% CI: [{effect.confidence_interval[0]:+.2%}, {effect.confidence_interval[1]:+.2%}]")
    print(f"  P-value: {effect.p_value:.4f}")
    
    # Test parallel trends assumption
    parallel_trends_test = analyzer.test_parallel_trends(model)
    print(f"  Parallel trends test: p={parallel_trends_test.p_value:.4f}")
    
    if parallel_trends_test.p_value < 0.05:
        print("  Warning: Parallel trends assumption may be violated")
    
    return {
        'effect': effect,
        'model': model,
        'parallel_trends_test': parallel_trends_test
    }


def custom_visualization(data: pd.DataFrame, results: dict, output_file: str):
    """Create custom visualization.
    
    This example shows how to create custom plots with statistical overlays.
    
    Args:
        data: DataFrame with time series data.
        results: Analysis results.
        output_file: Path to save plot.
    """
    print(f"Creating custom visualization...")
    
    # Create plot with confidence bands
    fig = plot_trend_with_confidence_bands(
        data=data,
        date_column='date',
        value_column='pageviews',
        confidence_level=0.95,
        title='Wikipedia Pageviews with Custom Analysis',
        show_changepoints=True,
        changepoints=results.get('changepoints', [])
    )
    
    # Save plot
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Plot saved to {output_file}")


def integrate_external_data(
    wikipedia_data: pd.DataFrame,
    external_data_path: str
) -> pd.DataFrame:
    """Integrate external data with Wikipedia data.
    
    This example shows how to merge external data sources for
    enhanced analysis.
    
    Args:
        wikipedia_data: Wikipedia pageview data.
        external_data_path: Path to external data CSV.
        
    Returns:
        Merged DataFrame.
    """
    print("Integrating external data...")
    
    # Load external data (e.g., Google Trends, news mentions, etc.)
    try:
        external_data = pd.read_csv(external_data_path, parse_dates=['date'])
        
        # Merge with Wikipedia data
        merged = pd.merge(
            wikipedia_data,
            external_data,
            on='date',
            how='left'
        )
        
        print(f"  Merged {len(merged)} records")
        print(f"  Columns: {', '.join(merged.columns)}")
        
        return merged
    except FileNotFoundError:
        print(f"  External data file not found: {external_data_path}")
        print(f"  Continuing with Wikipedia data only")
        return wikipedia_data


def main():
    """Run custom analysis."""
    print("Custom Wikipedia Health Analysis")
    print("=" * 70)
    print()
    
    # Configuration
    start_date = date(2020, 1, 1)
    end_date = date(2023, 12, 31)
    intervention_date = date(2022, 11, 30)  # Example: ChatGPT launch
    
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    print("✓ Configuration loaded")
    print()
    
    # Fetch data
    print("Fetching data...")
    client = WikimediaAPIClient(config.api)
    data = client.fetch_pageviews(
        start_date=start_date,
        end_date=end_date,
        platforms=['all'],
        agent_type='user'
    )
    print(f"✓ Fetched {len(data)} records")
    print()
    
    # Custom statistical tests
    stat_results = custom_statistical_test(data)
    print()
    
    # Custom changepoint detection
    changepoints = custom_changepoint_analysis(data)
    print()
    
    # Custom causal analysis
    causal_results = custom_causal_analysis(data, intervention_date)
    print()
    
    # Integrate external data (if available)
    merged_data = integrate_external_data(data, 'external_data.csv')
    print()
    
    # Custom visualization
    results = {
        'statistical_tests': stat_results,
        'changepoints': changepoints,
        'causal_effect': causal_results
    }
    custom_visualization(data, results, 'custom_analysis_plot.png')
    print()
    
    print("=" * 70)
    print("Custom analysis complete!")
    print()
    print("Key Findings:")
    print(f"- Detected {len(changepoints)} significant structural breaks")
    print(f"- Intervention effect: {causal_results['effect'].effect_size:+.2%}")
    print(f"- Effect significance: p={causal_results['effect'].p_value:.4f}")


if __name__ == '__main__':
    main()
