"""Tests for CLI functionality."""

import pytest
import json
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from wikipedia_health.cli import (
    parse_date,
    parse_campaign_dates,
    parse_event_dates,
    parse_external_events,
    create_parser,
    main
)
from wikipedia_health.config import Config


class TestDateParsing:
    """Tests for date parsing functions."""
    
    def test_parse_date_valid(self):
        """Test parsing valid date string."""
        result = parse_date("2023-06-15")
        assert result == date(2023, 6, 15)
    
    def test_parse_date_invalid_format(self):
        """Test parsing invalid date format."""
        with pytest.raises(ValueError, match="Invalid date format"):
            parse_date("15-06-2023")
    
    def test_parse_date_invalid_date(self):
        """Test parsing invalid date."""
        with pytest.raises(ValueError):
            parse_date("2023-13-45")


class TestCampaignDateParsing:
    """Tests for campaign date parsing."""
    
    def test_parse_campaign_dates_single(self):
        """Test parsing single campaign date."""
        result = parse_campaign_dates("2023-06-15:Summer Campaign")
        assert len(result) == 1
        assert result[0] == (date(2023, 6, 15), "Summer Campaign")
    
    def test_parse_campaign_dates_multiple(self):
        """Test parsing multiple campaign dates."""
        result = parse_campaign_dates(
            "2023-06-15:Summer Campaign,2023-12-01:Winter Campaign"
        )
        assert len(result) == 2
        assert result[0] == (date(2023, 6, 15), "Summer Campaign")
        assert result[1] == (date(2023, 12, 1), "Winter Campaign")
    
    def test_parse_campaign_dates_empty(self):
        """Test parsing empty campaign string."""
        result = parse_campaign_dates("")
        assert len(result) == 0
    
    def test_parse_campaign_dates_invalid_format(self):
        """Test parsing invalid campaign format."""
        result = parse_campaign_dates("2023-06-15")  # Missing name
        assert len(result) == 0  # Should skip invalid entries


class TestEventDateParsing:
    """Tests for event date parsing."""
    
    def test_parse_event_dates_single(self):
        """Test parsing single event date."""
        result = parse_event_dates("2023-06-15:Election:political")
        assert len(result) == 1
        assert result[0] == (date(2023, 6, 15), "Election", "political")
    
    def test_parse_event_dates_multiple(self):
        """Test parsing multiple event dates."""
        result = parse_event_dates(
            "2023-06-15:Election:political,2023-07-20:Earthquake:natural_disaster"
        )
        assert len(result) == 2
        assert result[0] == (date(2023, 6, 15), "Election", "political")
        assert result[1] == (date(2023, 7, 20), "Earthquake", "natural_disaster")
    
    def test_parse_event_dates_invalid_category(self):
        """Test parsing event with invalid category."""
        result = parse_event_dates("2023-06-15:Event:invalid_category")
        assert len(result) == 0  # Should skip invalid categories
    
    def test_parse_event_dates_valid_categories(self):
        """Test all valid event categories."""
        categories = ['political', 'natural_disaster', 'celebrity', 'scientific']
        for category in categories:
            result = parse_event_dates(f"2023-06-15:Event:{category}")
            assert len(result) == 1
            assert result[0][2] == category


class TestExternalEventParsing:
    """Tests for external event parsing."""
    
    def test_parse_external_events_single(self):
        """Test parsing single external event."""
        result = parse_external_events("2022-11-30:ChatGPT Launch")
        assert len(result) == 1
        assert result[0] == (date(2022, 11, 30), "ChatGPT Launch")
    
    def test_parse_external_events_multiple(self):
        """Test parsing multiple external events."""
        result = parse_external_events(
            "2022-11-30:ChatGPT Launch,2024-05-14:Google AI Overviews"
        )
        assert len(result) == 2


class TestArgumentParser:
    """Tests for argument parser."""
    
    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser is not None
    
    def test_parser_full_command(self):
        """Test parsing full analysis command."""
        parser = create_parser()
        args = parser.parse_args([
            'full',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31'
        ])
        assert args.command == 'full'
        assert args.start_date == '2020-01-01'
        assert args.end_date == '2023-12-31'
    
    def test_parser_trends_command(self):
        """Test parsing trends command."""
        parser = create_parser()
        args = parser.parse_args([
            'trends',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31',
            '--external-events', '2022-11-30:ChatGPT'
        ])
        assert args.command == 'trends'
        assert args.external_events == '2022-11-30:ChatGPT'
    
    def test_parser_platforms_command(self):
        """Test parsing platforms command."""
        parser = create_parser()
        args = parser.parse_args([
            'platforms',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31',
            '--platforms', 'desktop', 'mobile-web'
        ])
        assert args.command == 'platforms'
        assert args.platforms == ['desktop', 'mobile-web']
    
    def test_parser_campaigns_command(self):
        """Test parsing campaigns command."""
        parser = create_parser()
        args = parser.parse_args([
            'campaigns',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31',
            '--campaign-dates', '2021-06-15:Summer'
        ])
        assert args.command == 'campaigns'
        assert args.campaign_dates == '2021-06-15:Summer'
    
    def test_parser_events_command(self):
        """Test parsing events command."""
        parser = create_parser()
        args = parser.parse_args([
            'events',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31',
            '--event-dates', '2021-06-15:Event:political'
        ])
        assert args.command == 'events'
        assert args.event_dates == '2021-06-15:Event:political'
    
    def test_parser_forecasts_command(self):
        """Test parsing forecasts command."""
        parser = create_parser()
        args = parser.parse_args([
            'forecasts',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31',
            '--horizon', '90'
        ])
        assert args.command == 'forecasts'
        assert args.horizon == 90
    
    def test_parser_global_options(self):
        """Test parsing global options."""
        parser = create_parser()
        args = parser.parse_args([
            '--config', 'custom.yaml',
            '--verbose',
            '--output-dir', 'output',
            '--output-format', 'json',
            'full',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31'
        ])
        assert args.config == Path('custom.yaml')
        assert args.verbose is True
        assert args.output_dir == Path('output')
        assert args.output_format == 'json'


class TestCLIExecution:
    """Tests for CLI execution."""
    
    @patch('wikipedia_health.cli.AnalysisSystem')
    @patch('wikipedia_health.cli.load_config')
    def test_main_no_command(self, mock_load_config, mock_system):
        """Test main with no command."""
        result = main([])
        assert result == 1  # Should return error code
    
    @patch('wikipedia_health.cli.AnalysisSystem')
    @patch('wikipedia_health.cli.load_config')
    def test_main_full_analysis(self, mock_load_config, mock_system):
        """Test main with full analysis command."""
        # Setup mocks
        mock_config = Mock(spec=Config)
        mock_load_config.return_value = mock_config
        
        mock_system_instance = Mock()
        mock_system_instance.run_full_analysis.return_value = {'status': 'success'}
        mock_system.return_value = mock_system_instance
        
        # Run CLI
        result = main([
            'full',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31'
        ])
        
        # Verify
        assert result == 0
        mock_system_instance.run_full_analysis.assert_called_once()
    
    @patch('wikipedia_health.cli.AnalysisSystem')
    @patch('wikipedia_health.cli.load_config')
    def test_main_trends_analysis(self, mock_load_config, mock_system):
        """Test main with trends analysis command."""
        # Setup mocks
        mock_config = Mock(spec=Config)
        mock_validation_config = Mock()
        mock_validation_config.platforms = ['desktop', 'mobile-web', 'mobile-app']
        mock_config.validation = mock_validation_config
        mock_load_config.return_value = mock_config
        
        mock_system_instance = Mock()
        mock_system_instance._acquire_data.return_value = {}
        mock_system_instance.analyze_long_term_trends.return_value = {'status': 'success'}
        mock_system.return_value = mock_system_instance
        
        # Run CLI
        result = main([
            'trends',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31'
        ])
        
        # Verify
        assert result == 0
        mock_system_instance.analyze_long_term_trends.assert_called_once()
    
    @patch('wikipedia_health.cli.AnalysisSystem')
    @patch('wikipedia_health.cli.load_config')
    def test_main_with_config_file(self, mock_load_config, mock_system):
        """Test main with custom config file."""
        # Setup mocks
        mock_config = Mock(spec=Config)
        mock_load_config.return_value = mock_config
        
        mock_system_instance = Mock()
        mock_system_instance.run_full_analysis.return_value = {'status': 'success'}
        mock_system.return_value = mock_system_instance
        
        # Run CLI
        result = main([
            '--config', 'custom.yaml',
            'full',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31'
        ])
        
        # Verify
        assert result == 0
        mock_load_config.assert_called_once()
    
    @patch('wikipedia_health.cli.AnalysisSystem')
    @patch('wikipedia_health.cli.load_config')
    def test_main_error_handling(self, mock_load_config, mock_system):
        """Test main error handling."""
        # Setup mocks to raise exception
        mock_config = Mock(spec=Config)
        mock_load_config.return_value = mock_config
        
        mock_system_instance = Mock()
        mock_system_instance.run_full_analysis.side_effect = Exception("Test error")
        mock_system.return_value = mock_system_instance
        
        # Run CLI
        result = main([
            'full',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31'
        ])
        
        # Verify error code returned
        assert result == 1


class TestConfigurationLoading:
    """Tests for configuration loading in CLI."""
    
    @patch('wikipedia_health.cli.AnalysisSystem')
    @patch('wikipedia_health.cli.load_config')
    def test_config_override_significance_level(self, mock_load_config, mock_system):
        """Test overriding significance level from CLI."""
        # Setup mocks
        mock_config = Mock(spec=Config)
        mock_config.statistical = Mock()
        mock_load_config.return_value = mock_config
        
        mock_system_instance = Mock()
        mock_system_instance.run_full_analysis.return_value = {'status': 'success'}
        mock_system.return_value = mock_system_instance
        
        # Run CLI with significance level override
        result = main([
            'full',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31',
            '--significance-level', '0.01'
        ])
        
        # Verify significance level was overridden
        assert result == 0
        assert mock_config.statistical.significance_level == 0.01


class TestOutputGeneration:
    """Tests for output generation."""
    
    @patch('wikipedia_health.cli.AnalysisSystem')
    @patch('wikipedia_health.cli.load_config')
    def test_output_directory_creation(self, mock_load_config, mock_system):
        """Test output directory is created."""
        # Setup mocks
        mock_config = Mock(spec=Config)
        mock_validation_config = Mock()
        mock_validation_config.platforms = ['desktop']
        mock_config.validation = mock_validation_config
        mock_load_config.return_value = mock_config
        
        mock_system_instance = Mock()
        mock_system_instance._acquire_data.return_value = {}
        mock_system_instance.analyze_long_term_trends.return_value = {'status': 'success'}
        mock_system.return_value = mock_system_instance
        
        # Run CLI with custom output directory
        output_dir = Path('test_output')
        result = main([
            '--output-dir', str(output_dir),
            'trends',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31'
        ])
        
        # Verify
        assert result == 0
        # Output directory should be created (mocked in actual implementation)


class TestCampaignAnalysisCLI:
    """Tests for campaign analysis CLI."""
    
    @patch('wikipedia_health.cli.AnalysisSystem')
    @patch('wikipedia_health.cli.load_config')
    def test_campaigns_with_valid_dates(self, mock_load_config, mock_system):
        """Test campaigns analysis with valid dates."""
        # Setup mocks
        mock_config = Mock(spec=Config)
        mock_validation_config = Mock()
        mock_validation_config.platforms = ['desktop']
        mock_config.validation = mock_validation_config
        mock_load_config.return_value = mock_config
        
        mock_system_instance = Mock()
        mock_system_instance._acquire_data.return_value = {}
        mock_system_instance.analyze_campaigns.return_value = {'status': 'success'}
        mock_system.return_value = mock_system_instance
        
        # Run CLI
        result = main([
            'campaigns',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31',
            '--campaign-dates', '2021-06-15:Summer,2021-12-01:Winter'
        ])
        
        # Verify
        assert result == 0
        mock_system_instance.analyze_campaigns.assert_called_once()
    
    @patch('wikipedia_health.cli.AnalysisSystem')
    @patch('wikipedia_health.cli.load_config')
    def test_campaigns_with_no_valid_dates(self, mock_load_config, mock_system):
        """Test campaigns analysis with no valid dates."""
        # Setup mocks
        mock_config = Mock(spec=Config)
        mock_load_config.return_value = mock_config
        
        mock_system_instance = Mock()
        mock_system.return_value = mock_system_instance
        
        # Run CLI with invalid campaign dates
        result = main([
            'campaigns',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31',
            '--campaign-dates', 'invalid'
        ])
        
        # Verify error code
        assert result == 1


class TestEventsAnalysisCLI:
    """Tests for events analysis CLI."""
    
    @patch('wikipedia_health.cli.AnalysisSystem')
    @patch('wikipedia_health.cli.load_config')
    def test_events_with_valid_dates(self, mock_load_config, mock_system):
        """Test events analysis with valid dates."""
        # Setup mocks
        mock_config = Mock(spec=Config)
        mock_validation_config = Mock()
        mock_validation_config.platforms = ['desktop']
        mock_config.validation = mock_validation_config
        mock_load_config.return_value = mock_config
        
        mock_system_instance = Mock()
        mock_system_instance._acquire_data.return_value = {}
        mock_system_instance.analyze_external_shocks.return_value = {'status': 'success'}
        mock_system.return_value = mock_system_instance
        
        # Run CLI
        result = main([
            'events',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31',
            '--event-dates', '2021-06-15:Election:political'
        ])
        
        # Verify
        assert result == 0
        mock_system_instance.analyze_external_shocks.assert_called_once()


class TestForecastsCLI:
    """Tests for forecasts CLI."""
    
    @patch('wikipedia_health.cli.AnalysisSystem')
    @patch('wikipedia_health.cli.load_config')
    def test_forecasts_with_custom_horizon(self, mock_load_config, mock_system):
        """Test forecasts with custom horizon."""
        # Setup mocks
        mock_config = Mock(spec=Config)
        mock_validation_config = Mock()
        mock_validation_config.platforms = ['desktop']
        mock_config.validation = mock_validation_config
        mock_load_config.return_value = mock_config
        
        mock_system_instance = Mock()
        mock_system_instance._acquire_data.return_value = {}
        mock_system_instance.generate_forecasts.return_value = {'status': 'success'}
        mock_system.return_value = mock_system_instance
        
        # Run CLI
        result = main([
            'forecasts',
            '--start-date', '2020-01-01',
            '--end-date', '2023-12-31',
            '--horizon', '180'
        ])
        
        # Verify
        assert result == 0
        # Verify horizon was passed correctly
        call_args = mock_system_instance.generate_forecasts.call_args
        assert call_args[1]['horizon'] == 180
