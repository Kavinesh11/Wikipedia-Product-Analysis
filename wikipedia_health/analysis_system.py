"""Main Analysis System Orchestrator.

This module provides the Analysis_System class that orchestrates all components
of the Wikipedia Product Health Analysis system, providing a unified interface
for running complete analyses with error handling, progress tracking, and logging.
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from wikipedia_health.config import Config, load_config
from wikipedia_health.data_acquisition import (
    WikimediaAPIClient,
    DataValidator,
    save_timeseries_data,
    load_timeseries_data
)
from wikipedia_health.models import (
    TimeSeriesData,
    Finding,
    ValidationReport
)
from wikipedia_health.time_series import (
    TimeSeriesDecomposer,
    ChangepointDetector,
    Forecaster
)
from wikipedia_health.statistical_validation import (
    HypothesisTester,
    ConfidenceIntervalCalculator,
    EffectSizeCalculator
)
from wikipedia_health.causal_inference import (
    InterruptedTimeSeriesAnalyzer,
    DifferenceInDifferencesAnalyzer,
    EventStudyAnalyzer,
    SyntheticControlBuilder
)
from wikipedia_health.evidence_framework import (
    CrossValidator,
    RobustnessChecker
)
from wikipedia_health.multi_dimensional_analysis import MultiDimensionalAnalyzer
from wikipedia_health.reproducibility import (
    AnalysisLogger,
    save_results,
    load_results,
    run_pipeline
)
from wikipedia_health.visualization import (
    generate_finding_report,
    generate_evidence_report,
    export_report_to_file
)


logger = logging.getLogger(__name__)


@dataclass
class AnalysisProgress:
    """Tracks progress of analysis execution."""
    
    total_steps: int
    completed_steps: int = 0
    current_step: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100
    
    def update(self, step_name: str) -> None:
        """Update progress with new step."""
        self.current_step = step_name
        self.completed_steps += 1
        logger.info(f"Progress: {self.progress_percentage:.1f}% - {step_name}")
    
    def add_error(self, error: str) -> None:
        """Add error to tracking."""
        self.errors.append(error)
        logger.error(error)
    
    def add_warning(self, warning: str) -> None:
        """Add warning to tracking."""
        self.warnings.append(warning)
        logger.warning(warning)


class AnalysisSystem:
    """Main orchestrator for Wikipedia Product Health Analysis.
    
    This class provides a unified interface for running complete analyses,
    coordinating all components, handling errors, and tracking progress.
    
    Attributes:
        config: System configuration
        api_client: Wikimedia API client
        data_validator: Data validation component
        decomposer: Time series decomposer
        changepoint_detector: Changepoint detection component
        forecaster: Forecasting component
        hypothesis_tester: Statistical hypothesis testing
        ci_calculator: Confidence interval calculator
        effect_size_calculator: Effect size calculator
        itsa_analyzer: Interrupted time series analyzer
        did_analyzer: Difference-in-differences analyzer
        event_study_analyzer: Event study analyzer
        synthetic_control_builder: Synthetic control builder
        cross_validator: Cross-validation component
        robustness_checker: Robustness checking component
        multi_dim_analyzer: Multi-dimensional analyzer
        analysis_logger: Analysis logging component
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        config_path: Optional[Path] = None
    ):
        """Initialize Analysis System.
        
        Args:
            config: Configuration object (if None, loads from config_path or defaults)
            config_path: Path to configuration file
        """
        # Load configuration
        if config is None:
            self.config = load_config(config_path)
        else:
            self.config = config
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Analysis System initialized successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all analysis components."""
        try:
            # Data acquisition
            self.api_client = WikimediaAPIClient(config=self.config)
            self.data_validator = DataValidator(config=self.config)
            
            # Time series analysis
            self.decomposer = TimeSeriesDecomposer()
            self.changepoint_detector = ChangepointDetector()
            self.forecaster = Forecaster()
            
            # Statistical validation
            self.hypothesis_tester = HypothesisTester()
            self.ci_calculator = ConfidenceIntervalCalculator()
            self.effect_size_calculator = EffectSizeCalculator()
            
            # Causal inference
            self.itsa_analyzer = InterruptedTimeSeriesAnalyzer()
            self.did_analyzer = DifferenceInDifferencesAnalyzer()
            self.event_study_analyzer = EventStudyAnalyzer()
            self.synthetic_control_builder = SyntheticControlBuilder()
            
            # Evidence framework
            self.cross_validator = CrossValidator()
            self.robustness_checker = RobustnessChecker()
            
            # Multi-dimensional analysis
            self.multi_dim_analyzer = MultiDimensionalAnalyzer()
            
            # Reproducibility
            self.analysis_logger = AnalysisLogger()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run_full_analysis(
        self,
        start_date: date,
        end_date: date,
        platforms: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        analysis_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run complete analysis pipeline.
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            platforms: List of platforms to analyze (default: all)
            output_dir: Directory for output files
            analysis_types: Types of analyses to run (default: all)
                Options: 'trends', 'platforms', 'seasonality', 'campaigns',
                        'events', 'forecasts'
        
        Returns:
            Dictionary containing all analysis results and metadata
        """
        # Set defaults
        if platforms is None:
            platforms = self.config.validation.platforms
        if output_dir is None:
            output_dir = Path("output") / datetime.now().strftime("%Y%m%d_%H%M%S")
        if analysis_types is None:
            analysis_types = ['trends', 'platforms', 'seasonality', 'campaigns', 
                            'events', 'forecasts']
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress tracking
        progress = AnalysisProgress(total_steps=len(analysis_types) + 2)
        
        # Start analysis logging
        self.analysis_logger.start_analysis(
            start_date=start_date,
            end_date=end_date,
            platforms=platforms,
            analysis_types=analysis_types
        )
        
        results = {
            'metadata': {
                'start_date': start_date,
                'end_date': end_date,
                'platforms': platforms,
                'analysis_types': analysis_types,
                'timestamp': datetime.now().isoformat()
            },
            'data': {},
            'findings': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: Data acquisition
            progress.update("Acquiring data from Wikimedia APIs")
            data = self._acquire_data(start_date, end_date, platforms)
            results['data'] = data
            
            # Step 2: Data validation
            progress.update("Validating data quality")
            validation_report = self._validate_data(data)
            results['validation'] = validation_report
            
            if not validation_report.is_valid:
                progress.add_warning(
                    f"Data validation issues detected: {validation_report.recommendations}"
                )
            
            # Run requested analyses
            if 'trends' in analysis_types:
                progress.update("Analyzing long-term trends")
                try:
                    trends_results = self.analyze_long_term_trends(data)
                    results['trends'] = trends_results
                except Exception as e:
                    error_msg = f"Trends analysis failed: {e}"
                    progress.add_error(error_msg)
                    results['errors'].append(error_msg)
            
            if 'platforms' in analysis_types:
                progress.update("Analyzing platform dependency")
                try:
                    platform_results = self.analyze_platform_dependency(data)
                    results['platforms'] = platform_results
                except Exception as e:
                    error_msg = f"Platform analysis failed: {e}"
                    progress.add_error(error_msg)
                    results['errors'].append(error_msg)
            
            if 'seasonality' in analysis_types:
                progress.update("Analyzing seasonality patterns")
                try:
                    seasonality_results = self.analyze_seasonality(data)
                    results['seasonality'] = seasonality_results
                except Exception as e:
                    error_msg = f"Seasonality analysis failed: {e}"
                    progress.add_error(error_msg)
                    results['errors'].append(error_msg)
            
            if 'campaigns' in analysis_types:
                progress.update("Analyzing campaign effectiveness")
                try:
                    campaign_results = self.analyze_campaigns(data)
                    results['campaigns'] = campaign_results
                except Exception as e:
                    error_msg = f"Campaign analysis failed: {e}"
                    progress.add_error(error_msg)
                    results['errors'].append(error_msg)
            
            if 'events' in analysis_types:
                progress.update("Analyzing external event impacts")
                try:
                    events_results = self.analyze_external_shocks(data)
                    results['events'] = events_results
                except Exception as e:
                    error_msg = f"Events analysis failed: {e}"
                    progress.add_error(error_msg)
                    results['errors'].append(error_msg)
            
            if 'forecasts' in analysis_types:
                progress.update("Generating forecasts")
                try:
                    forecast_results = self.generate_forecasts(data)
                    results['forecasts'] = forecast_results
                except Exception as e:
                    error_msg = f"Forecast generation failed: {e}"
                    progress.add_error(error_msg)
                    results['errors'].append(error_msg)
            
            # Collect all findings
            results['findings'] = self._collect_findings(results)
            
            # Generate reports
            self._generate_reports(results, output_dir)
            
            # Save results
            save_results(results, output_dir / "results.json")
            
            # Log completion
            self.analysis_logger.complete_analysis(
                results=results,
                errors=progress.errors,
                warnings=progress.warnings
            )
            
            logger.info(f"Analysis completed successfully. Results saved to {output_dir}")
            
        except Exception as e:
            error_msg = f"Analysis pipeline failed: {e}"
            logger.error(error_msg)
            progress.add_error(error_msg)
            results['errors'].append(error_msg)
            raise
        
        finally:
            results['progress'] = {
                'completed_steps': progress.completed_steps,
                'total_steps': progress.total_steps,
                'progress_percentage': progress.progress_percentage,
                'errors': progress.errors,
                'warnings': progress.warnings
            }
        
        return results
    
    def _acquire_data(
        self,
        start_date: date,
        end_date: date,
        platforms: List[str]
    ) -> Dict[str, TimeSeriesData]:
        """Acquire data from Wikimedia APIs.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            platforms: List of platforms
        
        Returns:
            Dictionary of TimeSeriesData objects
        """
        data = {}
        
        try:
            # Fetch pageviews
            pageviews_df = self.api_client.fetch_pageviews(
                start_date=start_date,
                end_date=end_date,
                platforms=platforms,
                agent_type='user'
            )
            data['pageviews'] = TimeSeriesData.from_dataframe(
                pageviews_df,
                metric_type='pageviews'
            )
            
            # Fetch editor counts
            editors_df = self.api_client.fetch_editor_counts(
                start_date=start_date,
                end_date=end_date
            )
            data['editors'] = TimeSeriesData.from_dataframe(
                editors_df,
                metric_type='editors'
            )
            
            # Fetch edit volumes
            edits_df = self.api_client.fetch_edit_volumes(
                start_date=start_date,
                end_date=end_date
            )
            data['edits'] = TimeSeriesData.from_dataframe(
                edits_df,
                metric_type='edits'
            )
            
            logger.info("Data acquisition completed successfully")
            
        except Exception as e:
            logger.error(f"Data acquisition failed: {e}")
            raise
        
        return data
    
    def _validate_data(
        self,
        data: Dict[str, TimeSeriesData]
    ) -> ValidationReport:
        """Validate acquired data.
        
        Args:
            data: Dictionary of TimeSeriesData objects
        
        Returns:
            ValidationReport
        """
        try:
            # Validate each data source
            reports = []
            for source_name, ts_data in data.items():
                df = ts_data.to_dataframe()
                report = self.data_validator.check_completeness(
                    df,
                    (ts_data.date.min(), ts_data.date.max())
                )
                reports.append(report)
            
            # Combine reports
            combined_report = self._combine_validation_reports(reports)
            
            logger.info(f"Data validation completed. Valid: {combined_report.is_valid}")
            
            return combined_report
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise
    
    def _combine_validation_reports(
        self,
        reports: List[ValidationReport]
    ) -> ValidationReport:
        """Combine multiple validation reports.
        
        Args:
            reports: List of ValidationReport objects
        
        Returns:
            Combined ValidationReport
        """
        # Simple combination - all must be valid
        is_valid = all(r.is_valid for r in reports)
        completeness_score = sum(r.completeness_score for r in reports) / len(reports)
        
        all_missing_dates = []
        all_anomalies = []
        all_recommendations = []
        
        for report in reports:
            all_missing_dates.extend(report.missing_dates)
            all_anomalies.extend(report.anomalies)
            all_recommendations.extend(report.recommendations)
        
        return ValidationReport(
            is_valid=is_valid,
            completeness_score=completeness_score,
            missing_dates=list(set(all_missing_dates)),
            anomalies=all_anomalies,
            quality_metrics={},
            recommendations=list(set(all_recommendations))
        )
    
    def _collect_findings(self, results: Dict[str, Any]) -> List[Finding]:
        """Collect findings from all analyses.
        
        Args:
            results: Analysis results dictionary
        
        Returns:
            List of Finding objects
        """
        findings = []
        
        # Extract findings from each analysis type
        for analysis_type in ['trends', 'platforms', 'seasonality', 'campaigns', 
                             'events', 'forecasts']:
            if analysis_type in results and 'findings' in results[analysis_type]:
                findings.extend(results[analysis_type]['findings'])
        
        return findings
    
    def _generate_reports(
        self,
        results: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Generate analysis reports.
        
        Args:
            results: Analysis results
            output_dir: Output directory
        """
        try:
            # Generate finding report
            if results['findings']:
                finding_report = generate_finding_report(results['findings'])
                export_report_to_file(
                    finding_report,
                    output_dir / "findings_report.html"
                )
            
            # Generate evidence report
            evidence_report = generate_evidence_report(results)
            export_report_to_file(
                evidence_report,
                output_dir / "evidence_report.html"
            )
            
            logger.info(f"Reports generated in {output_dir}")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            # Don't raise - reports are nice-to-have
    
    # Analysis workflow functions (Team-specific analyses)
    def analyze_long_term_trends(
        self,
        data: Dict[str, TimeSeriesData],
        external_events: Optional[List[Tuple[date, str]]] = None
    ) -> Dict[str, Any]:
        """Analyze long-term trends (Team 1: structural shifts, AI impact).
        
        Performs structural shift detection, temporal alignment testing with
        external events (e.g., ChatGPT launch, Google AI Overviews), and
        pre-post comparison analysis.
        
        Args:
            data: Dictionary of TimeSeriesData objects
            external_events: List of (date, event_name) tuples for alignment testing
        
        Returns:
            Analysis results dictionary containing:
                - structural_shifts: Detected structural shifts
                - temporal_alignment: Alignment test results with external events
                - pre_post_comparisons: Statistical comparisons of periods
                - findings: List of Finding objects
        """
        from wikipedia_health.specialized_analysis import (
            analyze_structural_shifts,
            temporal_alignment_test,
            pre_post_comparison
        )
        
        results = {
            'structural_shifts': {},
            'temporal_alignment': {},
            'pre_post_comparisons': {},
            'findings': []
        }
        
        # Analyze pageviews for structural shifts
        if 'pageviews' in data:
            pageviews = data['pageviews']
            
            # Detect structural shifts
            shift_results = analyze_structural_shifts(
                pageviews,
                min_segment_size=self.config.time_series.changepoint_min_size,
                significance_level=self.config.statistical.significance_level
            )
            results['structural_shifts']['pageviews'] = shift_results
            
            # Test temporal alignment with external events if provided
            if external_events and shift_results['consensus_changepoints']:
                for event_date, event_name in external_events:
                    alignment_result = temporal_alignment_test(
                        changepoints=shift_results['consensus_changepoints'],
                        external_event_date=event_date,
                        event_name=event_name
                    )
                    results['temporal_alignment'][event_name] = alignment_result
            
            # Perform pre-post comparisons for each detected shift
            for changepoint in shift_results['consensus_changepoints']:
                comparison = pre_post_comparison(
                    pageviews,
                    changepoint.date,
                    pre_days=90,
                    post_days=90
                )
                results['pre_post_comparisons'][changepoint.date.isoformat()] = comparison
        
        # Cross-validate with editor activity
        if 'editors' in data:
            editor_shifts = analyze_structural_shifts(
                data['editors'],
                min_segment_size=self.config.time_series.changepoint_min_size,
                significance_level=self.config.statistical.significance_level
            )
            results['structural_shifts']['editors'] = editor_shifts
        
        return results
    
    def analyze_platform_dependency(
        self,
        data: Dict[str, TimeSeriesData]
    ) -> Dict[str, Any]:
        """Analyze platform dependency (Team 2: mobile vs desktop).
        
        Assesses platform dependency risks, calculates HHI, tests mobile
        dependency thresholds, and performs scenario analysis.
        
        Args:
            data: Dictionary of TimeSeriesData objects
        
        Returns:
            Analysis results dictionary containing:
                - platform_risk: Risk assessment results
                - threshold_tests: Mobile dependency threshold tests
                - scenario_analysis: Impact scenarios for platform declines
                - findings: List of Finding objects
        """
        from wikipedia_health.specialized_analysis import (
            assess_platform_risk,
            threshold_testing,
            platform_scenario_analysis
        )
        
        results = {
            'platform_risk': {},
            'threshold_tests': {},
            'scenario_analysis': {},
            'findings': []
        }
        
        # Extract platform-specific data from pageviews
        if 'pageviews' in data:
            pageviews = data['pageviews']
            
            # Split by platform if data contains platform information
            platform_data = {}
            if hasattr(pageviews, 'platform') and pageviews.platform != 'all':
                # Data is already platform-specific
                platform_data[pageviews.platform] = pageviews
            else:
                # Assume data needs to be split by platform
                # This would require platform-specific filtering in the data
                logger.warning("Platform-specific data not available, using aggregate data")
                platform_data['all'] = pageviews
            
            # Assess platform risk if we have platform-specific data
            if len(platform_data) > 1:
                risk_assessment = assess_platform_risk(
                    platform_data,
                    mobile_threshold=0.70,
                    high_concentration_hhi=2500.0
                )
                results['platform_risk'] = risk_assessment
                
                # Perform threshold testing
                threshold_results = threshold_testing(
                    platform_data,
                    threshold=0.70
                )
                results['threshold_tests'] = threshold_results
                
                # Perform scenario analysis
                scenarios = platform_scenario_analysis(
                    platform_data,
                    decline_percentages=[0.10, 0.20, 0.30]
                )
                results['scenario_analysis'] = scenarios
        
        return results
    
    def analyze_seasonality(
        self,
        data: Dict[str, TimeSeriesData]
    ) -> Dict[str, Any]:
        """Analyze seasonality patterns (Team 3: temporal patterns).
        
        Performs seasonal decomposition, validates seasonality, analyzes
        day-of-week effects, models holiday impacts, and classifies
        utility vs leisure usage patterns.
        
        Args:
            data: Dictionary of TimeSeriesData objects
        
        Returns:
            Analysis results dictionary containing:
                - seasonality_analysis: Seasonal decomposition results
                - validation: Seasonality validation tests
                - day_of_week: Day-of-week effect analysis
                - holiday_effects: Holiday impact modeling
                - usage_classification: Utility vs leisure classification
                - findings: List of Finding objects
        """
        from wikipedia_health.specialized_analysis import (
            analyze_seasonality,
            validate_seasonality,
            day_of_week_analysis,
            holiday_effect_modeling,
            utility_vs_leisure_classification
        )
        
        results = {
            'seasonality_analysis': {},
            'validation': {},
            'day_of_week': {},
            'holiday_effects': {},
            'usage_classification': {},
            'findings': []
        }
        
        # Analyze pageviews seasonality
        if 'pageviews' in data:
            pageviews = data['pageviews']
            
            # Perform seasonal decomposition
            seasonality_results = analyze_seasonality(
                pageviews,
                period=self.config.time_series.seasonal_period,
                methods=['stl', 'x13']
            )
            results['seasonality_analysis']['pageviews'] = seasonality_results
            
            # Validate seasonality
            validation_results = validate_seasonality(
                pageviews,
                period=self.config.time_series.seasonal_period
            )
            results['validation']['pageviews'] = validation_results
            
            # Analyze day-of-week effects
            dow_results = day_of_week_analysis(pageviews)
            results['day_of_week']['pageviews'] = dow_results
            
            # Model holiday effects (if holiday data available)
            # For now, skip holiday modeling as it requires holiday dates
            # holiday_results = holiday_effect_modeling(pageviews, holiday_dates)
            # results['holiday_effects']['pageviews'] = holiday_results
        
        # Classify utility vs leisure usage using engagement ratios
        if 'pageviews' in data and 'editors' in data:
            classification = utility_vs_leisure_classification(
                pageviews=data['pageviews'],
                editors=data['editors']
            )
            results['usage_classification'] = classification
        
        return results
    
    def analyze_campaigns(
        self,
        data: Dict[str, TimeSeriesData],
        campaign_dates: Optional[List[Tuple[date, str]]] = None
    ) -> Dict[str, Any]:
        """Analyze campaign effectiveness (Team 4: campaign effectiveness).
        
        Evaluates campaign impacts using ITSA and synthetic controls,
        analyzes duration effects, and compares across campaigns.
        
        Args:
            data: Dictionary of TimeSeriesData objects
            campaign_dates: List of (date, campaign_name) tuples
        
        Returns:
            Analysis results dictionary containing:
                - campaign_evaluations: Individual campaign results
                - duration_analysis: Immediate/short/long-term effects
                - cross_campaign_comparison: Comparison across campaigns
                - findings: List of Finding objects
        """
        from wikipedia_health.specialized_analysis import (
            evaluate_campaign,
            duration_analysis,
            cross_campaign_comparison
        )
        
        results = {
            'campaign_evaluations': {},
            'duration_analysis': {},
            'cross_campaign_comparison': {},
            'findings': []
        }
        
        if campaign_dates is None or len(campaign_dates) == 0:
            logger.warning("No campaign dates provided, skipping campaign analysis")
            return results
        
        # Evaluate each campaign
        if 'pageviews' in data:
            pageviews = data['pageviews']
            campaign_results = []
            
            for campaign_date, campaign_name in campaign_dates:
                try:
                    # Evaluate campaign
                    evaluation = evaluate_campaign(
                        pageviews,
                        campaign_start_date=campaign_date,
                        pre_period_days=self.config.causal.pre_period_length,
                        post_period_days=self.config.causal.post_period_length
                    )
                    results['campaign_evaluations'][campaign_name] = evaluation
                    campaign_results.append((campaign_name, evaluation))
                    
                    # Analyze duration effects
                    duration_results = duration_analysis(
                        pageviews,
                        campaign_date,
                        immediate_days=7,
                        short_term_days=30,
                        long_term_days=90
                    )
                    results['duration_analysis'][campaign_name] = duration_results
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate campaign {campaign_name}: {e}")
            
            # Compare campaigns if multiple campaigns analyzed
            if len(campaign_results) > 1:
                comparison = cross_campaign_comparison(
                    campaigns=campaign_results
                )
                results['cross_campaign_comparison'] = comparison
        
        return results
    
    def analyze_external_shocks(
        self,
        data: Dict[str, TimeSeriesData],
        event_dates: Optional[List[Tuple[date, str, str]]] = None
    ) -> Dict[str, Any]:
        """Analyze external event impacts (Team 5: news sensitivity).
        
        Measures Wikipedia's response to external shocks using event study
        methodology and compares responses across event categories.
        
        Args:
            data: Dictionary of TimeSeriesData objects
            event_dates: List of (date, event_name, category) tuples
                Categories: 'political', 'natural_disaster', 'celebrity', 'scientific'
        
        Returns:
            Analysis results dictionary containing:
                - event_analyses: Individual event impact results
                - category_comparison: Comparison across event categories
                - findings: List of Finding objects
        """
        from wikipedia_health.specialized_analysis import (
            analyze_external_event,
            event_category_comparison
        )
        
        results = {
            'event_analyses': {},
            'category_comparison': {},
            'findings': []
        }
        
        if event_dates is None or len(event_dates) == 0:
            logger.warning("No event dates provided, skipping external event analysis")
            return results
        
        # Analyze each event
        if 'pageviews' in data:
            pageviews = data['pageviews']
            event_results = []
            
            for event_date, event_name, event_category in event_dates:
                try:
                    # Analyze event
                    event_analysis = analyze_external_event(
                        pageviews,
                        event_date=event_date,
                        event_name=event_name,
                        event_category=event_category,
                        baseline_window_days=self.config.causal.baseline_window,
                        post_window_days=self.config.causal.event_post_window
                    )
                    results['event_analyses'][event_name] = event_analysis
                    event_results.append((event_name, event_category, event_analysis))
                    
                except Exception as e:
                    logger.error(f"Failed to analyze event {event_name}: {e}")
            
            # Compare across event categories if multiple events analyzed
            if len(event_results) > 1:
                comparison = event_category_comparison(
                    events=event_results
                )
                results['category_comparison'] = comparison
        
        return results
    
    def generate_forecasts(
        self,
        data: Dict[str, TimeSeriesData],
        horizon: int = 90
    ) -> Dict[str, Any]:
        """Generate forecasts (Team 6: future projections).
        
        Generates ensemble forecasts using multiple methods, evaluates
        accuracy, and performs scenario analysis.
        
        Args:
            data: Dictionary of TimeSeriesData objects
            horizon: Forecast horizon in days
        
        Returns:
            Analysis results dictionary containing:
                - forecasts: Forecast results for each metric
                - accuracy_evaluation: Forecast accuracy metrics
                - scenario_analysis: Optimistic/baseline/pessimistic scenarios
                - findings: List of Finding objects
        """
        from wikipedia_health.specialized_analysis import (
            generate_forecast,
            evaluate_forecast_accuracy,
            forecast_scenario_analysis
        )
        
        results = {
            'forecasts': {},
            'accuracy_evaluation': {},
            'scenario_analysis': {},
            'findings': []
        }
        
        # Generate forecasts for each data source
        for source_name, ts_data in data.items():
            try:
                # Generate ensemble forecast
                forecast_results = generate_forecast(
                    ts_data,
                    horizon=horizon,
                    methods=self.config.time_series.forecast_methods,
                    confidence_levels=self.config.time_series.prediction_intervals
                )
                results['forecasts'][source_name] = forecast_results
                
                # Evaluate forecast accuracy on holdout data
                accuracy_results = evaluate_forecast_accuracy(
                    ts_data,
                    forecast_results,
                    holdout_percentage=self.config.time_series.holdout_percentage
                )
                results['accuracy_evaluation'][source_name] = accuracy_results
                
                # Perform scenario analysis
                scenarios = forecast_scenario_analysis(
                    forecast_results,
                    scenarios=['optimistic', 'baseline', 'pessimistic']
                )
                results['scenario_analysis'][source_name] = scenarios
                
            except Exception as e:
                logger.error(f"Failed to generate forecast for {source_name}: {e}")
        
        return results
