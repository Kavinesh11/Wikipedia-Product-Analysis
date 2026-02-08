"""Unit tests for visualization and reporting functions.

Tests specific examples and edge cases for visualization components.
"""

import pytest
import pandas as pd
from pandas import Series
import numpy as np
from datetime import date, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from wikipedia_health.models import (
    TimeSeriesData,
    TestResult,
    CausalEffect,
    ForecastResult,
    Finding,
)
from wikipedia_health.visualization import (
    plot_trend_with_confidence_bands,
    plot_campaign_effect,
    plot_forecast,
    plot_comparison,
    add_statistical_annotations,
    generate_summary_table,
    generate_finding_report,
    generate_evidence_report,
)


class TestPlotGeneration:
    """Test plot generation with various data inputs."""
    
    def test_plot_trend_basic(self):
        """Test basic trend plot generation."""
        series = Series(np.random.randn(50).cumsum() + 100)
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        
        fig = plot_trend_with_confidence_bands(series, dates=dates)
        
        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)
    
    def test_plot_trend_with_confidence_bands(self):
        """Test trend plot with confidence intervals."""
        series = Series(np.random.randn(50).cumsum() + 100)
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        ci_lower = series - 10
        ci_upper = series + 10
        
        fig = plot_trend_with_confidence_bands(
            series,
            dates=dates,
            confidence_interval=(ci_lower, ci_upper)
        )
        
        assert fig is not None
        # Should have filled area for confidence bands
        assert len(fig.axes[0].collections) >= 1 or len(fig.axes[0].patches) >= 1
        plt.close(fig)
    
    def test_plot_trend_without_dates(self):
        """Test trend plot without date index."""
        series = Series(np.random.randn(50).cumsum() + 100)
        
        fig = plot_trend_with_confidence_bands(series)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_campaign_effect_basic(self):
        """Test campaign effect plot generation."""
        n_points = 60
        observed = Series(np.random.randn(n_points).cumsum() + 100)
        counterfactual = observed - 10
        dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
        
        causal_effect = CausalEffect(
            effect_size=10.0,
            confidence_interval=(8.0, 12.0),
            p_value=0.001,
            method='ITSA',
            counterfactual=counterfactual,
            observed=observed,
            treatment_period=(dates[20], dates[40])
        )
        
        fig = plot_campaign_effect(causal_effect, dates=dates)
        
        assert fig is not None
        assert len(fig.axes) > 0
        # Should have at least 2 lines (observed + counterfactual)
        assert len(fig.axes[0].lines) >= 2
        plt.close(fig)
    
    def test_plot_campaign_effect_with_annotations(self):
        """Test campaign effect plot with statistical annotations."""
        n_points = 60
        observed = Series(np.random.randn(n_points).cumsum() + 100)
        counterfactual = observed - 10
        dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
        
        causal_effect = CausalEffect(
            effect_size=10.0,
            confidence_interval=(8.0, 12.0),
            p_value=0.001,
            method='ITSA',
            counterfactual=counterfactual,
            observed=observed,
            treatment_period=(dates[20], dates[40])
        )
        
        fig = plot_campaign_effect(causal_effect, dates=dates, show_annotations=True)
        
        assert fig is not None
        # Should have text annotations
        assert len(fig.axes[0].texts) > 0
        plt.close(fig)
    
    def test_plot_forecast_basic(self):
        """Test forecast plot generation."""
        forecast_result = ForecastResult(
            point_forecast=Series(np.random.randn(30).cumsum() + 100),
            lower_bound=Series(np.random.randn(30).cumsum() + 90),
            upper_bound=Series(np.random.randn(30).cumsum() + 110),
            confidence_level=0.95,
            model_type='ARIMA',
            horizon=30
        )
        
        fig = plot_forecast(forecast_result)
        
        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)
    
    def test_plot_forecast_with_historical(self):
        """Test forecast plot with historical data."""
        historical = Series(np.random.randn(50).cumsum() + 100)
        historical_dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        
        forecast_result = ForecastResult(
            point_forecast=Series(np.random.randn(30).cumsum() + 100),
            lower_bound=Series(np.random.randn(30).cumsum() + 90),
            upper_bound=Series(np.random.randn(30).cumsum() + 110),
            confidence_level=0.95,
            model_type='ARIMA',
            horizon=30
        )
        
        fig = plot_forecast(
            forecast_result,
            historical=historical,
            historical_dates=historical_dates
        )
        
        assert fig is not None
        # Should have multiple lines (historical + forecast + bounds)
        assert len(fig.axes[0].lines) >= 2
        plt.close(fig)
    
    def test_plot_comparison_basic(self):
        """Test comparison plot generation."""
        groups = {
            'Group A': Series(np.random.randn(50) + 100),
            'Group B': Series(np.random.randn(50) + 105),
            'Group C': Series(np.random.randn(50) + 102)
        }
        
        fig = plot_comparison(groups)
        
        assert fig is not None
        assert len(fig.axes) > 0
        # Should have bars for each group
        assert len(fig.axes[0].patches) >= 3
        plt.close(fig)
    
    def test_plot_comparison_with_test_result(self):
        """Test comparison plot with statistical test results."""
        groups = {
            'Group A': Series(np.random.randn(50) + 100),
            'Group B': Series(np.random.randn(50) + 105)
        }
        
        test_result = TestResult(
            test_name='t-test',
            statistic=2.5,
            p_value=0.01,
            effect_size=0.5,
            confidence_interval=(0.2, 0.8),
            is_significant=True,
            alpha=0.05,
            interpretation="Significant difference between groups"
        )
        
        fig = plot_comparison(groups, test_result=test_result, show_significance=True)
        
        assert fig is not None
        # Should have text annotations with p-values
        assert len(fig.axes[0].texts) > 0
        plt.close(fig)
    
    def test_plot_comparison_with_error_bars(self):
        """Test comparison plot with error bars."""
        groups = {
            'Group A': Series(np.random.randn(50) + 100),
            'Group B': Series(np.random.randn(50) + 105)
        }
        
        fig = plot_comparison(groups, show_error_bars=True)
        
        assert fig is not None
        # Error bars should be present
        assert len(fig.axes[0].containers) > 0
        plt.close(fig)


class TestAnnotationPlacement:
    """Test annotation placement and formatting."""
    
    def test_add_statistical_annotations(self):
        """Test adding statistical annotations to existing plot."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        test_results = [
            TestResult(
                test_name='t-test',
                statistic=2.5,
                p_value=0.01,
                effect_size=0.5,
                confidence_interval=(0.2, 0.8),
                is_significant=True,
                alpha=0.05,
                interpretation="Test interpretation"
            )
        ]
        
        add_statistical_annotations(ax, test_results, position='top_right')
        
        # Should have added text annotations
        assert len(ax.texts) > 0
        plt.close(fig)
    
    def test_add_multiple_annotations(self):
        """Test adding multiple statistical annotations."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        test_results = [
            TestResult(
                test_name='t-test',
                statistic=2.5,
                p_value=0.01,
                effect_size=0.5,
                confidence_interval=(0.2, 0.8),
                is_significant=True,
                alpha=0.05,
                interpretation="Test 1"
            ),
            TestResult(
                test_name='ANOVA',
                statistic=3.2,
                p_value=0.001,
                effect_size=0.8,
                confidence_interval=(0.5, 1.1),
                is_significant=True,
                alpha=0.05,
                interpretation="Test 2"
            )
        ]
        
        add_statistical_annotations(ax, test_results, position='top_left')
        
        assert len(ax.texts) > 0
        plt.close(fig)


class TestReportGeneration:
    """Test report generation functions."""
    
    def test_generate_summary_table_basic(self):
        """Test basic summary table generation."""
        test_results = [
            TestResult(
                test_name='t-test',
                statistic=2.5,
                p_value=0.01,
                effect_size=0.5,
                confidence_interval=(0.2, 0.8),
                is_significant=True,
                alpha=0.05,
                interpretation="Significant difference"
            )
        ]
        
        table = generate_summary_table(test_results)
        
        assert table is not None
        assert len(table) == 1
        assert 'Test' in table.columns
        assert 'p-value' in table.columns
        assert 'Effect Size' in table.columns
    
    def test_generate_summary_table_with_causal_effects(self):
        """Test summary table with causal effects."""
        test_results = [
            TestResult(
                test_name='t-test',
                statistic=2.5,
                p_value=0.01,
                effect_size=0.5,
                confidence_interval=(0.2, 0.8),
                is_significant=True,
                alpha=0.05,
                interpretation="Test"
            )
        ]
        
        causal_effects = [
            CausalEffect(
                effect_size=10.0,
                confidence_interval=(8.0, 12.0),
                p_value=0.001,
                method='ITSA',
                counterfactual=Series([100, 101, 102]),
                observed=Series([110, 111, 112]),
                treatment_period=(date(2020, 1, 1), date(2020, 1, 3))
            )
        ]
        
        table = generate_summary_table(test_results, causal_effects)
        
        assert table is not None
        assert len(table) == 2  # 1 test + 1 causal effect
    
    def test_generate_finding_report_basic(self):
        """Test basic finding report generation."""
        finding = Finding(
            finding_id='TEST_001',
            description='Test finding description',
            evidence=[
                TestResult(
                    test_name='t-test',
                    statistic=2.5,
                    p_value=0.01,
                    effect_size=0.5,
                    confidence_interval=(0.2, 0.8),
                    is_significant=True,
                    alpha=0.05,
                    interpretation="Test"
                )
            ],
            confidence_level='high',
            requirements_validated=['13.6']
        )
        
        report = generate_finding_report(finding)
        
        assert report is not None
        assert 'FINDING REPORT' in report
        assert 'TEST_001' in report
        assert 'DESCRIPTION' in report
        assert 'STATISTICAL EVIDENCE' in report
    
    def test_generate_finding_report_with_causal_effects(self):
        """Test finding report with causal effects."""
        finding = Finding(
            finding_id='TEST_002',
            description='Test finding with causal analysis',
            evidence=[],
            causal_effects=[
                CausalEffect(
                    effect_size=10.0,
                    confidence_interval=(8.0, 12.0),
                    p_value=0.001,
                    method='ITSA',
                    counterfactual=Series([100, 101, 102]),
                    observed=Series([110, 111, 112]),
                    treatment_period=(date(2020, 1, 1), date(2020, 1, 3))
                )
            ],
            confidence_level='medium'
        )
        
        report = generate_finding_report(finding)
        
        assert report is not None
        assert 'CAUSAL ANALYSIS' in report
        assert 'ITSA' in report
    
    def test_generate_evidence_report_basic(self):
        """Test basic evidence report generation."""
        findings = [
            Finding(
                finding_id='FINDING_1',
                description='First finding',
                evidence=[
                    TestResult(
                        test_name='t-test',
                        statistic=2.5,
                        p_value=0.01,
                        effect_size=0.5,
                        confidence_interval=(0.2, 0.8),
                        is_significant=True,
                        alpha=0.05,
                        interpretation="Test"
                    )
                ],
                confidence_level='high'
            ),
            Finding(
                finding_id='FINDING_2',
                description='Second finding',
                evidence=[],
                confidence_level='medium'
            )
        ]
        
        report = generate_evidence_report(findings)
        
        assert report is not None
        assert 'EXECUTIVE SUMMARY' in report
        assert 'Total Findings: 2' in report
        assert 'FINDINGS SUMMARY' in report
    
    def test_generate_evidence_report_with_cross_validation(self):
        """Test evidence report with cross-validation summary."""
        findings = [
            Finding(
                finding_id='FINDING_1',
                description='Test finding',
                evidence=[],
                confidence_level='high',
                requirements_validated=['13.1', '13.2']
            )
        ]
        
        report = generate_evidence_report(findings, include_cross_validation=True)
        
        assert report is not None
        assert 'CROSS-VALIDATION SUMMARY' in report
        assert 'Requirements Validated' in report


class TestReportTableStructure:
    """Test report table structure and formatting."""
    
    def test_summary_table_columns(self):
        """Test that summary table has all required columns."""
        test_results = [
            TestResult(
                test_name='t-test',
                statistic=2.5,
                p_value=0.01,
                effect_size=0.5,
                confidence_interval=(0.2, 0.8),
                is_significant=True,
                alpha=0.05,
                interpretation="Test"
            )
        ]
        
        table = generate_summary_table(test_results)
        
        required_columns = [
            'Test', 'Statistic', 'p-value', 'Significant',
            'Effect Size', '95% CI Lower', '95% CI Upper', 'Interpretation'
        ]
        
        for col in required_columns:
            assert col in table.columns
    
    def test_summary_table_significance_indicators(self):
        """Test that summary table includes significance indicators."""
        test_results = [
            TestResult(
                test_name='t-test',
                statistic=2.5,
                p_value=0.0001,  # Changed to be < 0.001
                effect_size=0.5,
                confidence_interval=(0.2, 0.8),
                is_significant=True,
                alpha=0.05,
                interpretation="Test"
            )
        ]
        
        table = generate_summary_table(test_results)
        
        # Should have *** for p < 0.001
        assert '***' in table['p-value'].iloc[0]
    
    def test_summary_table_effect_magnitude(self):
        """Test that summary table includes effect magnitude interpretation."""
        test_results = [
            TestResult(
                test_name='t-test',
                statistic=2.5,
                p_value=0.01,
                effect_size=0.9,  # Large effect
                confidence_interval=(0.7, 1.1),
                is_significant=True,
                alpha=0.05,
                interpretation="Test"
            )
        ]
        
        table = generate_summary_table(test_results)
        
        assert 'Effect Magnitude' in table.columns
        assert table['Effect Magnitude'].iloc[0] in ['negligible', 'small', 'medium', 'large']


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_plot_with_empty_series(self):
        """Test plotting with empty series."""
        series = Series([])
        
        # Should handle gracefully - matplotlib can plot empty series
        try:
            fig = plot_trend_with_confidence_bands(series)
            plt.close(fig)
            # If it doesn't raise, that's fine - it handles empty data gracefully
        except (ValueError, IndexError):
            # If it does raise, that's also acceptable
            pass
    
    def test_plot_with_single_point(self):
        """Test plotting with single data point."""
        series = Series([100])
        
        fig = plot_trend_with_confidence_bands(series)
        assert fig is not None
        plt.close(fig)
    
    def test_report_with_no_evidence(self):
        """Test report generation with no evidence."""
        finding = Finding(
            finding_id='TEST_EMPTY',
            description='Finding with no evidence',
            evidence=[],
            confidence_level='low'
        )
        
        report = generate_finding_report(finding)
        
        assert report is not None
        assert 'TEST_EMPTY' in report
    
    def test_summary_table_with_empty_list(self):
        """Test summary table with empty test results."""
        table = generate_summary_table([])
        
        assert table is not None
        assert len(table) == 0
