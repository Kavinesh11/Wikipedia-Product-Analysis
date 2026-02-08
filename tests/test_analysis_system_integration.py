"""Integration tests for Analysis_System orchestrator.

Tests end-to-end pipeline from data acquisition to report generation,
component interactions, data flow, and error propagation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from wikipedia_health import AnalysisSystem, AnalysisProgress
from wikipedia_health.config import Config
from wikipedia_health.models import TimeSeriesData, ValidationReport, Anomaly


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return Config()


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate synthetic data with trend and seasonality
    trend = np.linspace(1000, 1200, len(dates))
    seasonality = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    noise = np.random.normal(0, 20, len(dates))
    values = trend + seasonality + noise
    
    return TimeSeriesData(
        date=pd.Series(dates),
        values=pd.Series(values),
        platform='all',
        metric_type='pageviews',
        metadata={'source': 'test'}
    )


@pytest.fixture
def sample_data_dict(sample_time_series_data):
    """Create sample data dictionary."""
    # Create variations for different metrics
    pageviews = sample_time_series_data
    
    # Editors data (smaller scale)
    editors = TimeSeriesData(
        date=pageviews.date.copy(),
        values=pageviews.values / 100,
        platform='all',
        metric_type='editors',
        metadata={'source': 'test'}
    )
    
    # Edits data (medium scale)
    edits = TimeSeriesData(
        date=pageviews.date.copy(),
        values=pageviews.values / 10,
        platform='all',
        metric_type='edits',
        metadata={'source': 'test'}
    )
    
    return {
        'pageviews': pageviews,
        'editors': editors,
        'edits': edits
    }


class TestAnalysisSystemInitialization:
    """Test Analysis_System initialization."""
    
    def test_initialization_with_default_config(self):
        """Test initialization with default configuration."""
        system = AnalysisSystem()
        
        assert system.config is not None
        assert system.api_client is not None
        assert system.data_validator is not None
        assert system.decomposer is not None
        assert system.changepoint_detector is not None
        assert system.forecaster is not None
    
    def test_initialization_with_custom_config(self, mock_config):
        """Test initialization with custom configuration."""
        system = AnalysisSystem(config=mock_config)
        
        assert system.config == mock_config
        assert system.api_client is not None
    
    def test_initialization_with_config_file(self, temp_output_dir):
        """Test initialization with configuration file."""
        config_path = temp_output_dir / "config.yaml"
        config = Config()
        config.to_yaml(config_path)
        
        system = AnalysisSystem(config_path=config_path)
        
        assert system.config is not None


class TestAnalysisProgress:
    """Test AnalysisProgress tracking."""
    
    def test_progress_initialization(self):
        """Test progress initialization."""
        progress = AnalysisProgress(total_steps=5)
        
        assert progress.total_steps == 5
        assert progress.completed_steps == 0
        assert progress.progress_percentage == 0.0
    
    def test_progress_update(self):
        """Test progress update."""
        progress = AnalysisProgress(total_steps=5)
        
        progress.update("Step 1")
        assert progress.completed_steps == 1
        assert progress.current_step == "Step 1"
        assert progress.progress_percentage == 20.0
        
        progress.update("Step 2")
        assert progress.completed_steps == 2
        assert progress.progress_percentage == 40.0
    
    def test_error_tracking(self):
        """Test error tracking."""
        progress = AnalysisProgress(total_steps=5)
        
        progress.add_error("Error 1")
        progress.add_error("Error 2")
        
        assert len(progress.errors) == 2
        assert "Error 1" in progress.errors
    
    def test_warning_tracking(self):
        """Test warning tracking."""
        progress = AnalysisProgress(total_steps=5)
        
        progress.add_warning("Warning 1")
        
        assert len(progress.warnings) == 1
        assert "Warning 1" in progress.warnings


class TestDataAcquisitionIntegration:
    """Test data acquisition integration."""
    
    @patch('wikipedia_health.analysis_system.WikimediaAPIClient')
    def test_acquire_data_success(self, mock_client_class, sample_data_dict):
        """Test successful data acquisition."""
        # Mock API client
        mock_client = Mock()
        mock_client.fetch_pageviews.return_value = sample_data_dict['pageviews'].to_dataframe()
        mock_client.fetch_editor_counts.return_value = sample_data_dict['editors'].to_dataframe()
        mock_client.fetch_edit_volumes.return_value = sample_data_dict['edits'].to_dataframe()
        mock_client_class.return_value = mock_client
        
        system = AnalysisSystem()
        system.api_client = mock_client
        
        # Acquire data
        data = system._acquire_data(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            platforms=['desktop', 'mobile-web']
        )
        
        assert 'pageviews' in data
        assert 'editors' in data
        assert 'edits' in data
        assert mock_client.fetch_pageviews.called
        assert mock_client.fetch_editor_counts.called
        assert mock_client.fetch_edit_volumes.called
    
    @patch('wikipedia_health.analysis_system.WikimediaAPIClient')
    def test_acquire_data_api_failure(self, mock_client_class):
        """Test data acquisition with API failure."""
        # Mock API client to raise exception
        mock_client = Mock()
        mock_client.fetch_pageviews.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        system = AnalysisSystem()
        system.api_client = mock_client
        
        # Should raise exception
        with pytest.raises(Exception, match="API Error"):
            system._acquire_data(
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                platforms=['desktop']
            )


class TestDataValidationIntegration:
    """Test data validation integration."""
    
    def test_validate_data_success(self, sample_data_dict):
        """Test successful data validation."""
        system = AnalysisSystem()
        
        # Mock validator
        mock_report = ValidationReport(
            is_valid=True,
            completeness_score=1.0,
            missing_dates=[],
            anomalies=[],
            quality_metrics={},
            recommendations=[]
        )
        system.data_validator.check_completeness = Mock(return_value=mock_report)
        
        # Validate data
        report = system._validate_data(sample_data_dict)
        
        assert report.is_valid
        assert report.completeness_score > 0
    
    def test_validate_data_with_issues(self, sample_data_dict):
        """Test data validation with quality issues."""
        system = AnalysisSystem()
        
        # Mock validator with issues
        mock_report = ValidationReport(
            is_valid=False,
            completeness_score=0.8,
            missing_dates=[date(2023, 6, 15)],
            anomalies=[Anomaly(
                date=date(2023, 7, 1),
                value=5000.0,
                expected_value=1500.0,
                z_score=5.0,
                description='High anomaly detected'
            )],
            quality_metrics={},
            recommendations=['Check data for June 15']
        )
        system.data_validator.check_completeness = Mock(return_value=mock_report)
        
        # Validate data
        report = system._validate_data(sample_data_dict)
        
        assert not report.is_valid
        assert len(report.missing_dates) > 0
        assert len(report.anomalies) > 0


class TestComponentInteractions:
    """Test interactions between components."""
    
    @patch('wikipedia_health.analysis_system.WikimediaAPIClient')
    def test_data_flow_through_components(self, mock_client_class, sample_data_dict):
        """Test data flow from acquisition through analysis."""
        # Mock API client
        mock_client = Mock()
        mock_client.fetch_pageviews.return_value = sample_data_dict['pageviews'].to_dataframe()
        mock_client.fetch_editor_counts.return_value = sample_data_dict['editors'].to_dataframe()
        mock_client.fetch_edit_volumes.return_value = sample_data_dict['edits'].to_dataframe()
        mock_client_class.return_value = mock_client
        
        system = AnalysisSystem()
        system.api_client = mock_client
        
        # Mock validator
        mock_report = ValidationReport(
            is_valid=True,
            completeness_score=1.0,
            missing_dates=[],
            anomalies=[],
            quality_metrics={},
            recommendations=[]
        )
        system.data_validator.check_completeness = Mock(return_value=mock_report)
        
        # Acquire and validate data
        data = system._acquire_data(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            platforms=['desktop']
        )
        
        validation_report = system._validate_data(data)
        
        # Verify data flows correctly
        assert data is not None
        assert validation_report is not None
        assert validation_report.is_valid


class TestErrorPropagation:
    """Test error propagation across components."""
    
    @patch('wikipedia_health.analysis_system.WikimediaAPIClient')
    def test_error_in_data_acquisition_propagates(self, mock_client_class, temp_output_dir):
        """Test that errors in data acquisition propagate correctly."""
        # Mock API client to fail
        mock_client = Mock()
        mock_client.fetch_pageviews.side_effect = Exception("Network error")
        mock_client_class.return_value = mock_client
        
        system = AnalysisSystem()
        system.api_client = mock_client
        
        # Run full analysis should handle error gracefully
        with pytest.raises(Exception):
            system.run_full_analysis(
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                output_dir=temp_output_dir
            )
    
    def test_error_in_analysis_continues_pipeline(self, sample_data_dict, temp_output_dir):
        """Test that errors in one analysis don't stop entire pipeline."""
        system = AnalysisSystem()
        
        # Mock data acquisition to return sample data
        system._acquire_data = Mock(return_value=sample_data_dict)
        
        # Mock validation to pass
        mock_report = ValidationReport(
            is_valid=True,
            completeness_score=1.0,
            missing_dates=[],
            anomalies=[],
            quality_metrics={},
            recommendations=[]
        )
        system._validate_data = Mock(return_value=mock_report)
        
        # Mock one analysis to fail
        system.analyze_long_term_trends = Mock(side_effect=Exception("Analysis failed"))
        
        # Mock other analyses to succeed
        system.analyze_platform_dependency = Mock(return_value={'platform_risk': {}})
        system.analyze_seasonality = Mock(return_value={'seasonality_analysis': {}})
        system.analyze_campaigns = Mock(return_value={'campaign_evaluations': {}})
        system.analyze_external_shocks = Mock(return_value={'event_analyses': {}})
        system.generate_forecasts = Mock(return_value={'forecasts': {}})
        
        # Mock report generation
        system._generate_reports = Mock()
        
        # Mock analysis logger
        system.analysis_logger.start_analysis = Mock()
        system.analysis_logger.complete_analysis = Mock()
        
        # Run analysis
        results = system.run_full_analysis(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            output_dir=temp_output_dir,
            analysis_types=['trends', 'platforms']
        )
        
        # Should have error recorded but other analyses completed
        assert len(results['errors']) > 0
        assert 'platforms' in results


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""
    
    @patch('wikipedia_health.analysis_system.WikimediaAPIClient')
    def test_minimal_pipeline_execution(self, mock_client_class, sample_data_dict, temp_output_dir):
        """Test minimal end-to-end pipeline execution."""
        # Mock API client
        mock_client = Mock()
        mock_client.fetch_pageviews.return_value = sample_data_dict['pageviews'].to_dataframe()
        mock_client.fetch_editor_counts.return_value = sample_data_dict['editors'].to_dataframe()
        mock_client.fetch_edit_volumes.return_value = sample_data_dict['edits'].to_dataframe()
        mock_client_class.return_value = mock_client
        
        system = AnalysisSystem()
        system.api_client = mock_client
        
        # Mock validator
        mock_report = ValidationReport(
            is_valid=True,
            completeness_score=1.0,
            missing_dates=[],
            anomalies=[],
            quality_metrics={},
            recommendations=[]
        )
        system.data_validator.check_completeness = Mock(return_value=mock_report)
        
        # Mock all analysis functions to return minimal results
        system.analyze_long_term_trends = Mock(return_value={
            'structural_shifts': {},
            'findings': []
        })
        system.analyze_platform_dependency = Mock(return_value={
            'platform_risk': {},
            'findings': []
        })
        system.analyze_seasonality = Mock(return_value={
            'seasonality_analysis': {},
            'findings': []
        })
        system.analyze_campaigns = Mock(return_value={
            'campaign_evaluations': {},
            'findings': []
        })
        system.analyze_external_shocks = Mock(return_value={
            'event_analyses': {},
            'findings': []
        })
        system.generate_forecasts = Mock(return_value={
            'forecasts': {},
            'findings': []
        })
        
        # Mock report generation
        system._generate_reports = Mock()
        
        # Mock analysis logger
        system.analysis_logger.start_analysis = Mock()
        system.analysis_logger.complete_analysis = Mock()
        
        # Run full analysis
        results = system.run_full_analysis(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            output_dir=temp_output_dir
        )
        
        # Verify results structure
        assert 'metadata' in results
        assert 'data' in results
        assert 'findings' in results
        assert 'progress' in results
        
        # Verify metadata
        assert results['metadata']['start_date'] == date(2023, 1, 1)
        assert results['metadata']['end_date'] == date(2023, 12, 31)
        
        # Verify all analyses were called
        assert system.analyze_long_term_trends.called
        assert system.analyze_platform_dependency.called
        assert system.analyze_seasonality.called
        assert system.analyze_campaigns.called
        assert system.analyze_external_shocks.called
        assert system.generate_forecasts.called
    
    @patch('wikipedia_health.analysis_system.WikimediaAPIClient')
    def test_selective_analysis_execution(self, mock_client_class, sample_data_dict, temp_output_dir):
        """Test running only selected analyses."""
        # Mock API client
        mock_client = Mock()
        mock_client.fetch_pageviews.return_value = sample_data_dict['pageviews'].to_dataframe()
        mock_client.fetch_editor_counts.return_value = sample_data_dict['editors'].to_dataframe()
        mock_client.fetch_edit_volumes.return_value = sample_data_dict['edits'].to_dataframe()
        mock_client_class.return_value = mock_client
        
        system = AnalysisSystem()
        system.api_client = mock_client
        
        # Mock validator
        mock_report = ValidationReport(
            is_valid=True,
            completeness_score=1.0,
            missing_dates=[],
            anomalies=[],
            quality_metrics={},
            recommendations=[]
        )
        system.data_validator.check_completeness = Mock(return_value=mock_report)
        
        # Mock only selected analyses
        system.analyze_long_term_trends = Mock(return_value={'structural_shifts': {}})
        system.analyze_seasonality = Mock(return_value={'seasonality_analysis': {}})
        
        # Mock report generation
        system._generate_reports = Mock()
        
        # Mock analysis logger
        system.analysis_logger.start_analysis = Mock()
        system.analysis_logger.complete_analysis = Mock()
        
        # Run only trends and seasonality
        results = system.run_full_analysis(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            output_dir=temp_output_dir,
            analysis_types=['trends', 'seasonality']
        )
        
        # Verify only selected analyses were called
        assert system.analyze_long_term_trends.called
        assert system.analyze_seasonality.called
        
        # Verify results contain only selected analyses
        assert 'trends' in results
        assert 'seasonality' in results
        assert 'platforms' not in results
        assert 'campaigns' not in results


class TestReportGeneration:
    """Test report generation integration."""
    
    def test_generate_reports_with_findings(self, temp_output_dir):
        """Test report generation with findings."""
        from wikipedia_health.models import Finding, TestResult
        
        system = AnalysisSystem()
        
        # Create sample findings
        findings = [
            Finding(
                finding_id='F1',
                description='Test finding',
                evidence=[TestResult(
                    test_name='t-test',
                    statistic=2.5,
                    p_value=0.01,
                    effect_size=0.5,
                    confidence_interval=(0.2, 0.8),
                    is_significant=True,
                    alpha=0.05,
                    interpretation='Significant difference'
                )],
                causal_effects=[],
                confidence_level='high',
                requirements_validated=['1.1']
            )
        ]
        
        results = {
            'findings': findings,
            'metadata': {},
            'data': {}
        }
        
        # Mock report functions
        with patch('wikipedia_health.analysis_system.generate_finding_report') as mock_finding_report, \
             patch('wikipedia_health.analysis_system.generate_evidence_report') as mock_evidence_report, \
             patch('wikipedia_health.analysis_system.export_report_to_file') as mock_export:
            
            mock_finding_report.return_value = "Finding Report"
            mock_evidence_report.return_value = "Evidence Report"
            
            # Generate reports
            system._generate_reports(results, temp_output_dir)
            
            # Verify reports were generated
            assert mock_finding_report.called
            assert mock_evidence_report.called
            assert mock_export.call_count == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
