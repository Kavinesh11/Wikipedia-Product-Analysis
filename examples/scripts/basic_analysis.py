#!/usr/bin/env python3
"""Basic analysis example.

This script demonstrates a simple workflow:
1. Load configuration
2. Fetch data
3. Run trend analysis
4. Save results

Usage:
    python examples/scripts/basic_analysis.py
    python examples/scripts/basic_analysis.py --config custom_config.yaml
"""

import argparse
import json
from datetime import date
from pathlib import Path

from wikipedia_health.config import load_config
from wikipedia_health.analysis_system import AnalysisSystem


def main():
    """Run basic analysis."""
    parser = argparse.ArgumentParser(description='Run basic Wikipedia health analysis')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2023-12-31',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    
    print(f"Wikipedia Product Health Analysis")
    print(f"=" * 60)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Configuration: {args.config}")
    print()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    print(f"✓ Configuration loaded")
    print()
    
    # Initialize analysis system
    print("Initializing analysis system...")
    system = AnalysisSystem(config=config)
    print(f"✓ System initialized")
    print()
    
    # Run trend analysis
    print("Running trend analysis...")
    print("This may take a few minutes...")
    results = system.analyze_long_term_trends(
        start_date=start_date,
        end_date=end_date,
        platforms=['desktop', 'mobile-web', 'mobile-app']
    )
    print(f"✓ Analysis complete")
    print()
    
    # Display findings
    print(f"Findings ({len(results['findings'])}):")
    print("-" * 60)
    for i, finding in enumerate(results['findings'], 1):
        print(f"{i}. {finding.description}")
        print(f"   Confidence: {finding.confidence_level}")
        print(f"   Evidence: {len(finding.evidence)} statistical tests")
        print()
    
    # Display changepoints
    if 'changepoints' in results:
        print(f"Structural Breaks ({len(results['changepoints'])}):")
        print("-" * 60)
        for cp in results['changepoints']:
            print(f"- {cp.date}: {cp.direction} ({cp.magnitude:+.2%})")
            print(f"  Confidence: {cp.confidence:.2%}")
            print()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'basic_analysis_results.json'
    print(f"Saving results to {output_file}...")
    
    # Convert results to JSON-serializable format
    results_json = {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'findings': [
            {
                'description': f.description,
                'confidence_level': f.confidence_level,
                'evidence_count': len(f.evidence)
            }
            for f in results['findings']
        ],
        'changepoints': [
            {
                'date': cp.date.isoformat(),
                'direction': cp.direction,
                'magnitude': cp.magnitude,
                'confidence': cp.confidence
            }
            for cp in results.get('changepoints', [])
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"✓ Results saved")
    print()
    print("Analysis complete!")


if __name__ == '__main__':
    main()
