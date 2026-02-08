"""Unit tests for core data models."""

import pytest
import pandas as pd
from datetime import date, datetime
from wikipedia_health.models import (
    TimeSeriesData,
    TestResult,
    CausalEffect,
    ForecastResult,
    DecompositionResult,
    Anomaly,
    ValidationReport,
    Changepoint,
    Finding,
)


class TestTimeSeriesData:
    """Tests for TimeSeriesData dataclass."""
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        dates = pd.Series(pd.date_range('2020-01-01', periods=5))
        values = pd.Series([100, 110, 105, 115, 120])
        
        ts_data = TimeSeriesData(
            date=dates,
            values=values,
            platform='desktop',
            metric_type='pageviews',
            metadata={'source': 'test'}
        )
        
        df = ts_data.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'date' in df.columns
        assert 'values' in df.columns
        assert df['platform'].iloc[0] == 'desktop'
        assert df['metric_type'].iloc[0] == 'pageviews'
    
    def test_resample(self):
        """Test resampling to different frequency."""
        dates = pd.Series(pd.date_range('2020-01-01', periods=7))
        values = pd.Series([10, 20, 30, 40, 50, 60, 70])
        
        ts_data = TimeSeriesData(
            date=dates,
            values=values,
            platform='mobile-web',
            metric_type='pageviews'
        )
        
        resampled = ts_data.resample('W')
        
        assert len(resampled.values) < len(values)
        assert 'resampled_frequency' in resampled.metadata
        assert resampled.metadata['resampled_frequency'] == 'W'
    
    def test_filter_date_range(self):
        """Test filtering by date range."""
        dates = pd.Series(pd.date_range('2020-01-01', periods=10))
        values = pd.Series(range(10))
        
        ts_data = TimeSeriesData(
            date=dates,
            values=values,
            platform='desktop',
            metric_type='editors'
        )
        
        filtered = ts_data.filter_date_range(date(2020, 1, 3), date(2020, 1, 7))
        
        assert len(filtered.values) == 5
        assert 'filtered_range' in filtered.metadata


class TestTestResult:
    """Tests for TestResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TestResult(
            test_name='t-test',
            statistic=2.5,
            p_value=0.012,
            effect_size=0.8,
            confidence_interval=(0.2, 1.4),
            is_significant=True,
            alpha=0.05,
            interpretation='Significant difference detected'
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['test_name'] == 't-test'
        assert result_dict['p_value'] == 0.012
        assert result_dict['is_significant'] is True
        assert result_dict['confidence_interval'] == (0.2, 1.4)


class TestCausalEffect:
    """Tests for CausalEffect dataclass."""
    
    def test_percentage_effect(self):
        """Test percentage effect calculation."""
        counterfactual = pd.Series([100, 100, 100, 100])
        observed = pd.Series([120, 125, 130, 135])
        
        effect = CausalEffect(
            effect_size=27.5,
            confidence_interval=(20.0, 35.0),
            p_value=0.001,
            method='ITSA',
            counterfactual=counterfactual,
            observed=observed,
            treatment_period=(date(2020, 1, 1), date(2020, 1, 4))
        )
        
        pct_effect = effect.percentage_effect()
        
        assert abs(pct_effect - 27.5) < 0.01  # Allow for floating point precision
    
    def test_percentage_effect_zero_counterfactual(self):
        """Test percentage effect with zero counterfactual."""
        counterfactual = pd.Series([0, 0, 0, 0])
        observed = pd.Series([10, 10, 10, 10])
        
        effect = CausalEffect(
            effect_size=10.0,
            confidence_interval=(5.0, 15.0),
            p_value=0.05,
            method='DiD',
            counterfactual=counterfactual,
            observed=observed,
            treatment_period=(date(2020, 1, 1), date(2020, 1, 4))
        )
        
        pct_effect = effect.percentage_effect()
        
        assert pct_effect == 0.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        counterfactual = pd.Series([100, 100])
        observed = pd.Series([120, 125])
        
        effect = CausalEffect(
            effect_size=22.5,
            confidence_interval=(15.0, 30.0),
            p_value=0.002,
            method='SyntheticControl',
            counterfactual=counterfactual,
            observed=observed,
            treatment_period=(date(2020, 1, 1), date(2020, 1, 2))
        )
        
        effect_dict = effect.to_dict()
        
        assert effect_dict['effect_size'] == 22.5
        assert effect_dict['method'] == 'SyntheticControl'
        assert 'percentage_effect' in effect_dict


class TestForecastResult:
    """Tests for ForecastResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        forecast = ForecastResult(
            point_forecast=pd.Series([100, 105, 110]),
            lower_bound=pd.Series([90, 95, 100]),
            upper_bound=pd.Series([110, 115, 120]),
            confidence_level=0.95,
            model_type='ARIMA',
            horizon=3
        )
        
        forecast_dict = forecast.to_dict()
        
        assert forecast_dict['model_type'] == 'ARIMA'
        assert forecast_dict['horizon'] == 3
        assert forecast_dict['confidence_level'] == 0.95
        assert len(forecast_dict['point_forecast']) == 3


class TestDecompositionResult:
    """Tests for DecompositionResult dataclass."""
    
    def test_reconstruct(self):
        """Test reconstruction of original series."""
        trend = pd.Series([100, 101, 102, 103])
        seasonal = pd.Series([5, -5, 5, -5])
        residual = pd.Series([1, -1, 0, 1])
        
        decomp = DecompositionResult(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            method='STL',
            parameters={'period': 2}
        )
        
        reconstructed = decomp.reconstruct()
        expected = pd.Series([106, 95, 107, 99])
        
        pd.testing.assert_series_equal(reconstructed, expected)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        decomp = DecompositionResult(
            trend=pd.Series([100, 101]),
            seasonal=pd.Series([5, -5]),
            residual=pd.Series([0, 0]),
            method='X-13-ARIMA-SEATS',
            parameters={'seasonal_period': 7}
        )
        
        decomp_dict = decomp.to_dict()
        
        assert decomp_dict['method'] == 'X-13-ARIMA-SEATS'
        assert 'seasonal_period' in decomp_dict['parameters']


class TestValidationReport:
    """Tests for ValidationReport dataclass."""
    
    def test_summary(self):
        """Test summary generation."""
        anomalies = [
            Anomaly(
                date=date(2020, 1, 5),
                value=1000,
                expected_value=100,
                z_score=5.0,
                description='Spike detected'
            )
        ]
        
        report = ValidationReport(
            is_valid=False,
            completeness_score=0.95,
            missing_dates=[date(2020, 1, 10)],
            anomalies=anomalies,
            quality_metrics={'variance': 0.5},
            recommendations=['Review anomaly on 2020-01-05']
        )
        
        summary = report.summary()
        
        assert 'FAIL' in summary
        assert '95.00%' in summary
        assert 'Missing Dates: 1' in summary
        assert 'Anomalies Detected: 1' in summary
        assert 'variance' in summary


class TestChangepoint:
    """Tests for Changepoint dataclass."""
    
    def test_is_significant(self):
        """Test significance check."""
        cp = Changepoint(
            date=date(2020, 6, 1),
            index=150,
            confidence=0.98,
            magnitude=50.0,
            direction='increase',
            pre_mean=100.0,
            post_mean=150.0
        )
        
        # confidence=0.98 means significant at alpha=0.05 (0.98 > 0.95)
        assert cp.is_significant(alpha=0.05) is True
        # But not at alpha=0.01 (0.98 < 0.99)
        assert cp.is_significant(alpha=0.01) is False
    
    def test_is_not_significant(self):
        """Test non-significant changepoint."""
        cp = Changepoint(
            date=date(2020, 6, 1),
            index=150,
            confidence=0.90,
            magnitude=10.0,
            direction='increase',
            pre_mean=100.0,
            post_mean=110.0
        )
        
        assert cp.is_significant(alpha=0.05) is False


class TestFinding:
    """Tests for Finding dataclass."""
    
    def test_summary(self):
        """Test finding summary generation."""
        evidence = [
            TestResult(
                test_name='t-test',
                statistic=3.0,
                p_value=0.003,
                effect_size=0.8,
                confidence_interval=(0.3, 1.3),
                is_significant=True,
                alpha=0.05,
                interpretation='Significant'
            )
        ]
        
        finding = Finding(
            finding_id='F001',
            description='Significant traffic increase detected',
            evidence=evidence,
            confidence_level='high',
            requirements_validated=['1.1', '2.1']
        )
        
        summary = finding.summary()
        
        assert 'F001' in summary
        assert 'HIGH' in summary
        assert 't-test' in summary
        assert '1.1, 2.1' in summary
    
    def test_evidence_strength(self):
        """Test evidence strength calculation."""
        evidence = [
            TestResult(
                test_name='test1',
                statistic=2.0,
                p_value=0.04,
                effect_size=0.5,
                confidence_interval=(0.1, 0.9),
                is_significant=True,
                alpha=0.05,
                interpretation='Significant'
            ),
            TestResult(
                test_name='test2',
                statistic=1.5,
                p_value=0.10,
                effect_size=0.3,
                confidence_interval=(-0.1, 0.7),
                is_significant=False,
                alpha=0.05,
                interpretation='Not significant'
            )
        ]
        
        finding = Finding(
            finding_id='F002',
            description='Mixed evidence',
            evidence=evidence,
            confidence_level='medium'
        )
        
        strength = finding.evidence_strength()
        
        assert 0.0 <= strength <= 1.0
        assert strength > 0.0  # Should have some strength
    
    def test_evidence_strength_no_evidence(self):
        """Test evidence strength with no evidence."""
        finding = Finding(
            finding_id='F003',
            description='No evidence',
            evidence=[],
            confidence_level='low'
        )
        
        strength = finding.evidence_strength()
        
        assert strength == 0.0
