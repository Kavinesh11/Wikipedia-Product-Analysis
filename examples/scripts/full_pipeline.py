#!/usr/bin/env python3
"""Full analysis pipeline example.

This script demonstrates a complete analysis workflow:
1. Data acquisition from Wikimedia APIs
2. Data validation and quality checks
3. All analysis types (trends, platforms, seasonality, etc.)
4. Cross-validation and robustness checks
5. Report generation

Usage:
    python examples/scripts/full_pipeline.py --start-date 2020-01-01 --end-date 2023-12-31
    python examples/scripts/full_pipeline.py --config custom_config.yaml --output-dir results/
"""

import argparse
import json
from datetime import date, datetime
from pathlib import Path

from wikipedia_health.config import load_config
from wikipedia_health.analysis_system import AnalysisSystem


def parse_date_list(date_str: str) -> list:
    """Parse comma-separated date:label pairs.
    
    Args:
        date_str: String like "2022-11-30:ChatGPT,2024-05-14:AI Overviews"
        
    Returns:
        List of (date, label) tuples.
    """
    if not date_str:
        return []
    
    result = []
    for pair in date_str.split(','):
        date_part, label = pair.split(':')
        result.append((date.fromisoformat(date_part), label))
    return result


def main():
    """Run full analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Run complete Wikipedia health analysis pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--platforms',
        nargs='+',
        default=['desktop', 'mobile-web', 'mobile-app'],
        help='Platforms to analyze'
    )
    parser.add_argument(
        '--external-events',
        type=str,
        default='',
        help='External events (format: YYYY-MM-DD:label,YYYY-MM-DD:label,...)'
    )
    parser.add_argument(
        '--campaign-dates',
        type=str,
        default='',
        help='Campaign dates (format: YYYY-MM-DD:label,YYYY-MM-DD:label,...)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    external_events = parse_date_list(args.external_events)
    campaign_dates = parse_date_list(args.campaign_dates)
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('output') / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Wikipedia Product Health Analysis - Full Pipeline")
    print(f"=" * 70)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Platforms: {', '.join(args.platforms)}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load configuration
    print("Step 1: Loading configuration...")
    config = load_config(args.config)
    print(f"✓ Configuration loaded")
    print()
    
    # Initialize analysis system
    print("Step 2: Initializing analysis system...")
    system = AnalysisSystem(config=config)
    print(f"✓ System initialized")
    print()
    
    # Run full analysis
    print("Step 3: Running full analysis...")
    print("This will take several minutes...")
    print()
    
    results = system.run_full_analysis(
        start_date=start_date,
        end_date=end_date,
        platforms=args.platforms,
        external_events=external_events if external_events else None,
        campaign_dates=campaign_dates if campaign_dates else None
    )
    
    print(f"✓ Analysis complete")
    print()
    
    # Display summary
    print("Analysis Summary")
    print("=" * 70)
    print()
    
    # Trends
    if 'trends' in results:
        print(f"Long-Term Trends:")
        print(f"  Changepoints detected: {len(results['trends'].get('changepoints', []))}")
        print(f"  Findings: {len(results['trends'].get('findings', []))}")
        print()
    
    # Platforms
    if 'platforms' in results:
        print(f"Platform Analysis:")
        platform_mix = results['platforms'].get('platform_mix', {})
        for platform, proportion in platform_mix.items():
            print(f"  {platform}: {proportion:.1%}")
        print()
    
    # Seasonality
    if 'seasonality' in results:
        print(f"Seasonality:")
        seasonal_strength = results['seasonality'].get('seasonal_strength', 0)
        print(f"  Seasonal strength: {seasonal_strength:.2f}")
        print()
    
    # Campaigns
    if 'campaigns' in results and campaign_dates:
        print(f"Campaign Effects:")
        for campaign, effect in results['campaigns'].get('campaign_effects', {}).items():
            print(f"  {campaign}: {effect.effect_size:+.2%} (p={effect.p_value:.4f})")
        print()
    
    # Events
    if 'events' in results and external_events:
        print(f"External Event Impacts:")
        for event, impact in results['events'].get('event_impacts', {}).items():
            print(f"  {event}: CAR={impact.get('car', 0):+.2%}")
        print()
    
    # Forecasts
    if 'forecasts' in results:
        print(f"Forecasts:")
        forecast = results['forecasts'].get('ensemble_forecast')
        if forecast:
            print(f"  Mean forecast: {forecast.point_forecast.mean():.0f} daily pageviews")
            print(f"  95% CI: [{forecast.lower_bound.mean():.0f}, {forecast.upper_bound.mean():.0f}]")
        print()
    
    # Overall findings
    all_findings = results.get('findings', [])
    print(f"Total Findings: {len(all_findings)}")
    print("-" * 70)
    for i, finding in enumerate(all_findings[:10], 1):  # Show first 10
        print(f"{i}. {finding.description}")
        print(f"   Confidence: {finding.confidence_level}")
        print()
    
    if len(all_findings) > 10:
        print(f"... and {len(all_findings) - 10} more findings")
        print()
    
    # Save results
    print("Step 4: Saving results...")
    
    # Save main results
    results_file = output_dir / 'full_analysis_results.json'
    with open(results_file, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {
            'metadata': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'platforms': args.platforms,
                'timestamp': datetime.now().isoformat()
            },
            'summary': {
                'total_findings': len(all_findings),
                'high_confidence_findings': sum(
                    1 for f in all_findings if f.confidence_level == 'high'
                )
            }
        }
        json.dump(json_results, f, indent=2)
    print(f"✓ Results saved to {results_file}")
    
    # Generate reports
    if hasattr(system, 'generate_reports'):
        print("Step 5: Generating reports...")
        system.generate_reports(results, output_dir)
        print(f"✓ Reports generated in {output_dir}")
    
    print()
    print("=" * 70)
    print("Pipeline complete!")
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
