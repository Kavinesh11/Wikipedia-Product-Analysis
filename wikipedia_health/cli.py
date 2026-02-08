"""Command-line interface for Wikipedia Product Health Analysis.

This module provides a CLI for running various types of analyses on Wikipedia
product health data, with support for different output formats and configuration options.
"""

import argparse
import sys
import logging
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

from wikipedia_health.analysis_system import AnalysisSystem
from wikipedia_health.config import load_config
from wikipedia_health.visualization import export_report_to_file


logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> date:
    """Parse date string in YYYY-MM-DD format.
    
    Args:
        date_str: Date string
    
    Returns:
        date object
    
    Raises:
        ValueError: If date format is invalid
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


def parse_campaign_dates(campaign_str: str) -> List[Tuple[date, str]]:
    """Parse campaign dates string.
    
    Format: "YYYY-MM-DD:campaign_name,YYYY-MM-DD:campaign_name,..."
    
    Args:
        campaign_str: Campaign dates string
    
    Returns:
        List of (date, campaign_name) tuples
    """
    campaigns = []
    for item in campaign_str.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            date_str, name = item.split(':', 1)
            campaigns.append((parse_date(date_str), name.strip()))
        except ValueError:
            logger.warning(f"Invalid campaign format: {item}. Expected YYYY-MM-DD:name")
    return campaigns


def parse_event_dates(event_str: str) -> List[Tuple[date, str, str]]:
    """Parse event dates string.
    
    Format: "YYYY-MM-DD:event_name:category,YYYY-MM-DD:event_name:category,..."
    Categories: political, natural_disaster, celebrity, scientific
    
    Args:
        event_str: Event dates string
    
    Returns:
        List of (date, event_name, category) tuples
    """
    events = []
    valid_categories = ['political', 'natural_disaster', 'celebrity', 'scientific']
    
    for item in event_str.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            parts = item.split(':', 2)
            if len(parts) != 3:
                logger.warning(f"Invalid event format: {item}. Expected YYYY-MM-DD:name:category")
                continue
            
            date_str, name, category = parts
            category = category.strip().lower()
            
            if category not in valid_categories:
                logger.warning(f"Invalid event category: {category}. Must be one of {valid_categories}")
                continue
            
            events.append((parse_date(date_str), name.strip(), category))
        except ValueError as e:
            logger.warning(f"Invalid event format: {item}. Error: {e}")
    
    return events


def parse_external_events(external_str: str) -> List[Tuple[date, str]]:
    """Parse external events string for trend analysis.
    
    Format: "YYYY-MM-DD:event_name,YYYY-MM-DD:event_name,..."
    
    Args:
        external_str: External events string
    
    Returns:
        List of (date, event_name) tuples
    """
    events = []
    for item in external_str.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            date_str, name = item.split(':', 1)
            events.append((parse_date(date_str), name.strip()))
        except ValueError:
            logger.warning(f"Invalid external event format: {item}. Expected YYYY-MM-DD:name")
    return events


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.
    
    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.
    
    Returns:
        ArgumentParser object
    """
    parser = argparse.ArgumentParser(
        description='Wikipedia Product Health Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis
  %(prog)s full --start-date 2020-01-01 --end-date 2023-12-31
  
  # Analyze trends with external events
  %(prog)s trends --start-date 2020-01-01 --end-date 2023-12-31 \\
    --external-events "2022-11-30:ChatGPT Launch,2024-05-14:Google AI Overviews"
  
  # Analyze platform dependency
  %(prog)s platforms --start-date 2020-01-01 --end-date 2023-12-31 \\
    --platforms desktop mobile-web mobile-app
  
  # Analyze campaigns
  %(prog)s campaigns --start-date 2020-01-01 --end-date 2023-12-31 \\
    --campaign-dates "2021-06-15:Summer Campaign,2022-12-01:Winter Campaign"
  
  # Generate forecasts
  %(prog)s forecasts --start-date 2020-01-01 --end-date 2023-12-31 \\
    --horizon 90 --output-format json
        """
    )
    
    # Global options
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for results (default: output/TIMESTAMP)'
    )
    parser.add_argument(
        '--output-format',
        choices=['json', 'html', 'pdf'],
        default='html',
        help='Output format for reports (default: html)'
    )
    
    # Create subparsers for different analysis types
    subparsers = parser.add_subparsers(dest='command', help='Analysis command')
    
    # Common arguments for all analysis commands
    def add_common_args(subparser):
        """Add common arguments to a subparser."""
        subparser.add_argument(
            '--start-date',
            type=str,
            required=True,
            help='Analysis start date (YYYY-MM-DD)'
        )
        subparser.add_argument(
            '--end-date',
            type=str,
            required=True,
            help='Analysis end date (YYYY-MM-DD)'
        )
        subparser.add_argument(
            '--platforms',
            nargs='+',
            choices=['desktop', 'mobile-web', 'mobile-app', 'all'],
            help='Platforms to analyze (default: all)'
        )
        subparser.add_argument(
            '--significance-level',
            type=float,
            help='Statistical significance level (default: 0.05)'
        )
    
    # Full analysis command
    full_parser = subparsers.add_parser(
        'full',
        help='Run complete analysis pipeline'
    )
    add_common_args(full_parser)
    full_parser.add_argument(
        '--analysis-types',
        nargs='+',
        choices=['trends', 'platforms', 'seasonality', 'campaigns', 'events', 'forecasts'],
        help='Types of analyses to run (default: all)'
    )
    
    # Trends analysis command
    trends_parser = subparsers.add_parser(
        'trends',
        help='Analyze long-term trends and structural shifts'
    )
    add_common_args(trends_parser)
    trends_parser.add_argument(
        '--external-events',
        type=str,
        help='External events for temporal alignment (format: YYYY-MM-DD:name,YYYY-MM-DD:name,...)'
    )
    
    # Platform analysis command
    platforms_parser = subparsers.add_parser(
        'platforms',
        help='Analyze platform dependency and risks'
    )
    add_common_args(platforms_parser)
    platforms_parser.add_argument(
        '--mobile-threshold',
        type=float,
        default=0.70,
        help='Mobile dependency threshold (default: 0.70)'
    )
    
    # Seasonality analysis command
    seasonality_parser = subparsers.add_parser(
        'seasonality',
        help='Analyze seasonal patterns and temporal effects'
    )
    add_common_args(seasonality_parser)
    seasonality_parser.add_argument(
        '--seasonal-period',
        type=int,
        default=7,
        help='Seasonal period in days (default: 7)'
    )
    
    # Campaign analysis command
    campaigns_parser = subparsers.add_parser(
        'campaigns',
        help='Analyze campaign effectiveness'
    )
    add_common_args(campaigns_parser)
    campaigns_parser.add_argument(
        '--campaign-dates',
        type=str,
        required=True,
        help='Campaign dates (format: YYYY-MM-DD:name,YYYY-MM-DD:name,...)'
    )
    
    # Events analysis command
    events_parser = subparsers.add_parser(
        'events',
        help='Analyze external event impacts'
    )
    add_common_args(events_parser)
    events_parser.add_argument(
        '--event-dates',
        type=str,
        required=True,
        help='Event dates (format: YYYY-MM-DD:name:category,YYYY-MM-DD:name:category,...)'
    )
    
    # Forecasts command
    forecasts_parser = subparsers.add_parser(
        'forecasts',
        help='Generate traffic forecasts'
    )
    add_common_args(forecasts_parser)
    forecasts_parser.add_argument(
        '--horizon',
        type=int,
        default=90,
        help='Forecast horizon in days (default: 90)'
    )
    forecasts_parser.add_argument(
        '--methods',
        nargs='+',
        choices=['arima', 'prophet', 'exponential_smoothing'],
        help='Forecasting methods to use (default: all)'
    )
    
    return parser


def run_full_analysis(args: argparse.Namespace, system: AnalysisSystem) -> int:
    """Run full analysis pipeline.
    
    Args:
        args: Parsed command-line arguments
        system: AnalysisSystem instance
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
        
        logger.info(f"Running full analysis from {start_date} to {end_date}")
        
        results = system.run_full_analysis(
            start_date=start_date,
            end_date=end_date,
            platforms=args.platforms,
            output_dir=args.output_dir,
            analysis_types=args.analysis_types
        )
        
        logger.info("Full analysis completed successfully")
        logger.info(f"Results saved to {args.output_dir or 'output/TIMESTAMP'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Full analysis failed: {e}", exc_info=True)
        return 1


def run_trends_analysis(args: argparse.Namespace, system: AnalysisSystem) -> int:
    """Run trends analysis.
    
    Args:
        args: Parsed command-line arguments
        system: AnalysisSystem instance
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
        
        logger.info(f"Running trends analysis from {start_date} to {end_date}")
        
        # Acquire data
        data = system._acquire_data(start_date, end_date, args.platforms or system.config.validation.platforms)
        
        # Parse external events if provided
        external_events = None
        if args.external_events:
            external_events = parse_external_events(args.external_events)
            logger.info(f"Analyzing alignment with {len(external_events)} external events")
        
        # Run trends analysis
        results = system.analyze_long_term_trends(data, external_events=external_events)
        
        # Save results
        output_dir = args.output_dir or Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from wikipedia_health.reproducibility import save_results
        save_results(results, output_dir / "trends_results.json")
        
        logger.info("Trends analysis completed successfully")
        logger.info(f"Results saved to {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Trends analysis failed: {e}", exc_info=True)
        return 1


def run_platforms_analysis(args: argparse.Namespace, system: AnalysisSystem) -> int:
    """Run platform dependency analysis.
    
    Args:
        args: Parsed command-line arguments
        system: AnalysisSystem instance
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
        
        logger.info(f"Running platform analysis from {start_date} to {end_date}")
        
        # Acquire data
        data = system._acquire_data(start_date, end_date, args.platforms or system.config.validation.platforms)
        
        # Run platform analysis
        results = system.analyze_platform_dependency(data)
        
        # Save results
        output_dir = args.output_dir or Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from wikipedia_health.reproducibility import save_results
        save_results(results, output_dir / "platforms_results.json")
        
        logger.info("Platform analysis completed successfully")
        logger.info(f"Results saved to {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Platform analysis failed: {e}", exc_info=True)
        return 1


def run_seasonality_analysis(args: argparse.Namespace, system: AnalysisSystem) -> int:
    """Run seasonality analysis.
    
    Args:
        args: Parsed command-line arguments
        system: AnalysisSystem instance
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
        
        logger.info(f"Running seasonality analysis from {start_date} to {end_date}")
        
        # Acquire data
        data = system._acquire_data(start_date, end_date, args.platforms or system.config.validation.platforms)
        
        # Run seasonality analysis
        results = system.analyze_seasonality(data)
        
        # Save results
        output_dir = args.output_dir or Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from wikipedia_health.reproducibility import save_results
        save_results(results, output_dir / "seasonality_results.json")
        
        logger.info("Seasonality analysis completed successfully")
        logger.info(f"Results saved to {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Seasonality analysis failed: {e}", exc_info=True)
        return 1


def run_campaigns_analysis(args: argparse.Namespace, system: AnalysisSystem) -> int:
    """Run campaign effectiveness analysis.
    
    Args:
        args: Parsed command-line arguments
        system: AnalysisSystem instance
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
        
        logger.info(f"Running campaign analysis from {start_date} to {end_date}")
        
        # Parse campaign dates
        campaign_dates = parse_campaign_dates(args.campaign_dates)
        if not campaign_dates:
            logger.error("No valid campaign dates provided")
            return 1
        
        logger.info(f"Analyzing {len(campaign_dates)} campaigns")
        
        # Acquire data
        data = system._acquire_data(start_date, end_date, args.platforms or system.config.validation.platforms)
        
        # Run campaign analysis
        results = system.analyze_campaigns(data, campaign_dates=campaign_dates)
        
        # Save results
        output_dir = args.output_dir or Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from wikipedia_health.reproducibility import save_results
        save_results(results, output_dir / "campaigns_results.json")
        
        logger.info("Campaign analysis completed successfully")
        logger.info(f"Results saved to {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Campaign analysis failed: {e}", exc_info=True)
        return 1


def run_events_analysis(args: argparse.Namespace, system: AnalysisSystem) -> int:
    """Run external events analysis.
    
    Args:
        args: Parsed command-line arguments
        system: AnalysisSystem instance
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
        
        logger.info(f"Running events analysis from {start_date} to {end_date}")
        
        # Parse event dates
        event_dates = parse_event_dates(args.event_dates)
        if not event_dates:
            logger.error("No valid event dates provided")
            return 1
        
        logger.info(f"Analyzing {len(event_dates)} events")
        
        # Acquire data
        data = system._acquire_data(start_date, end_date, args.platforms or system.config.validation.platforms)
        
        # Run events analysis
        results = system.analyze_external_shocks(data, event_dates=event_dates)
        
        # Save results
        output_dir = args.output_dir or Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from wikipedia_health.reproducibility import save_results
        save_results(results, output_dir / "events_results.json")
        
        logger.info("Events analysis completed successfully")
        logger.info(f"Results saved to {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Events analysis failed: {e}", exc_info=True)
        return 1


def run_forecasts(args: argparse.Namespace, system: AnalysisSystem) -> int:
    """Run forecast generation.
    
    Args:
        args: Parsed command-line arguments
        system: AnalysisSystem instance
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
        
        logger.info(f"Generating forecasts from {start_date} to {end_date}")
        logger.info(f"Forecast horizon: {args.horizon} days")
        
        # Acquire data
        data = system._acquire_data(start_date, end_date, args.platforms or system.config.validation.platforms)
        
        # Run forecast generation
        results = system.generate_forecasts(data, horizon=args.horizon)
        
        # Save results
        output_dir = args.output_dir or Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from wikipedia_health.reproducibility import save_results
        save_results(results, output_dir / "forecasts_results.json")
        
        logger.info("Forecast generation completed successfully")
        logger.info(f"Results saved to {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}", exc_info=True)
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point.
    
    Args:
        argv: Command-line arguments (default: sys.argv[1:])
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override configuration with command-line arguments
        if hasattr(args, 'significance_level') and args.significance_level:
            config.statistical.significance_level = args.significance_level
        
        if hasattr(args, 'seasonal_period') and args.seasonal_period:
            config.time_series.seasonal_period = args.seasonal_period
        
        # Initialize analysis system
        system = AnalysisSystem(config=config)
        
        # Route to appropriate command handler
        if args.command == 'full':
            return run_full_analysis(args, system)
        elif args.command == 'trends':
            return run_trends_analysis(args, system)
        elif args.command == 'platforms':
            return run_platforms_analysis(args, system)
        elif args.command == 'seasonality':
            return run_seasonality_analysis(args, system)
        elif args.command == 'campaigns':
            return run_campaigns_analysis(args, system)
        elif args.command == 'events':
            return run_events_analysis(args, system)
        elif args.command == 'forecasts':
            return run_forecasts(args, system)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        logger.error(f"CLI execution failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
