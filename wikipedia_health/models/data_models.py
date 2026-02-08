"""Core data models for Wikipedia Product Health Analysis.

This module defines the fundamental data structures used throughout the system,
including time series data, statistical results, causal effects, and validation reports.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from pandas import Series, DataFrame


@dataclass
class TimeSeriesData:
    """Time series data container with metadata.
    
    Attributes:
        date: DatetimeIndex or Series of dates
        values: Series of numeric values
        platform: Platform identifier ('desktop', 'mobile-web', 'mobile-app', 'all')
        metric_type: Type of metric ('pageviews', 'editors', 'edits')
        metadata: Additional metadata (source, acquisition_date, filters applied)
    """
    date: Series
    values: Series
    platform: str
    metric_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dataframe(cls, df: DataFrame, metric_type: str, platform: str = 'all', metadata: Optional[Dict[str, Any]] = None) -> 'TimeSeriesData':
        """Create TimeSeriesData from a DataFrame.
        
        Args:
            df: DataFrame with 'date' and 'values' columns (or similar)
            metric_type: Type of metric ('pageviews', 'editors', 'edits')
            platform: Platform identifier (default: 'all')
            metadata: Optional metadata dictionary
            
        Returns:
            TimeSeriesData instance
        """
        # Handle different column naming conventions
        date_col = None
        value_col = None
        
        # Try to find date column
        for col in ['date', 'timestamp', 'datetime']:
            if col in df.columns:
                date_col = col
                break
        
        # Try to find value column
        for col in ['values', 'value', 'count', 'views', 'editors', 'edits']:
            if col in df.columns:
                value_col = col
                break
        
        if date_col is None or value_col is None:
            raise ValueError(f"DataFrame must have date and value columns. Found columns: {df.columns.tolist()}")
        
        # Extract platform if present in dataframe
        if 'platform' in df.columns and len(df['platform'].unique()) == 1:
            platform = df['platform'].iloc[0]
        
        return cls(
            date=pd.Series(df[date_col].values),
            values=pd.Series(df[value_col].values),
            platform=platform,
            metric_type=metric_type,
            metadata=metadata or {}
        )
    
    def to_dataframe(self) -> DataFrame:
        """Convert to pandas DataFrame.
        
        Returns:
            DataFrame with date and values columns
        """
        df = pd.DataFrame({
            'date': self.date,
            'values': self.values,
            'platform': self.platform,
            'metric_type': self.metric_type
        })
        return df
    
    def resample(self, frequency: str) -> 'TimeSeriesData':
        """Resample time series to different frequency.
        
        Args:
            frequency: Pandas frequency string ('D', 'W', 'M', etc.)
            
        Returns:
            New TimeSeriesData with resampled data
        """
        df = self.to_dataframe()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        resampled = df['values'].resample(frequency).sum()
        
        return TimeSeriesData(
            date=pd.Series(resampled.index),
            values=pd.Series(resampled.values),
            platform=self.platform,
            metric_type=self.metric_type,
            metadata={**self.metadata, 'resampled_frequency': frequency}
        )
    
    def filter_date_range(self, start: date, end: date) -> 'TimeSeriesData':
        """Filter data to specified date range.
        
        Args:
            start: Start date (inclusive)
            end: End date (inclusive)
            
        Returns:
            New TimeSeriesData filtered to date range
        """
        df = self.to_dataframe()
        df['date'] = pd.to_datetime(df['date'])
        mask = (df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))
        filtered = df[mask]
        
        return TimeSeriesData(
            date=filtered['date'].reset_index(drop=True),
            values=filtered['values'].reset_index(drop=True),
            platform=self.platform,
            metric_type=self.metric_type,
            metadata={**self.metadata, 'filtered_range': (start, end)}
        )



@dataclass
class TestResult:
    """Statistical test result with evidence.
    
    Attributes:
        test_name: Name of the statistical test
        statistic: Test statistic value
        p_value: P-value from the test
        effect_size: Magnitude of the effect
        confidence_interval: Tuple of (lower, upper) bounds
        is_significant: Whether result is significant at alpha level
        alpha: Significance level used
        interpretation: Human-readable summary
    """
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    alpha: float
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary with all test result fields
        """
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'effect_size': self.effect_size,
            'confidence_interval': self.confidence_interval,
            'is_significant': self.is_significant,
            'alpha': self.alpha,
            'interpretation': self.interpretation
        }
    
    def plot(self) -> Any:
        """Generate visualization of test result.
        
        Returns:
            Matplotlib figure (placeholder for now)
        """
        # Placeholder - will be implemented in visualization module
        raise NotImplementedError("Visualization will be implemented in visualization module")


@dataclass
class CausalEffect:
    """Causal effect estimate with counterfactual.
    
    Attributes:
        effect_size: Estimated causal impact
        confidence_interval: Tuple of (lower, upper) bounds
        p_value: Statistical significance
        method: Causal inference method used
        counterfactual: Predicted values without intervention
        observed: Actual observed values
        treatment_period: Tuple of (start_date, end_date)
    """
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str
    counterfactual: Series
    observed: Series
    treatment_period: Tuple[date, date]
    
    def percentage_effect(self) -> float:
        """Calculate percentage effect relative to counterfactual.
        
        Returns:
            Percentage change from counterfactual
        """
        counterfactual_mean = self.counterfactual.mean()
        if counterfactual_mean == 0:
            return 0.0
        return (self.effect_size / counterfactual_mean) * 100
    
    def plot_comparison(self) -> Any:
        """Generate observed vs counterfactual plot.
        
        Returns:
            Matplotlib figure (placeholder for now)
        """
        # Placeholder - will be implemented in visualization module
        raise NotImplementedError("Visualization will be implemented in visualization module")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary with all causal effect fields
        """
        return {
            'effect_size': self.effect_size,
            'confidence_interval': self.confidence_interval,
            'p_value': self.p_value,
            'method': self.method,
            'percentage_effect': self.percentage_effect(),
            'treatment_period': self.treatment_period
        }


@dataclass
class ForecastResult:
    """Forecast result with uncertainty quantification.
    
    Attributes:
        point_forecast: Series of point predictions
        lower_bound: Lower prediction interval
        upper_bound: Upper prediction interval
        confidence_level: Confidence level for intervals (e.g., 0.95)
        model_type: Type of forecasting model used
        horizon: Number of periods forecasted
    """
    point_forecast: Series
    lower_bound: Series
    upper_bound: Series
    confidence_level: float
    model_type: str
    horizon: int
    
    def plot(self, historical: Optional[Series] = None) -> Any:
        """Generate forecast plot with uncertainty bands.
        
        Args:
            historical: Optional historical data to plot alongside forecast
            
        Returns:
            Matplotlib figure (placeholder for now)
        """
        # Placeholder - will be implemented in visualization module
        raise NotImplementedError("Visualization will be implemented in visualization module")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary with all forecast result fields
        """
        return {
            'point_forecast': self.point_forecast.tolist(),
            'lower_bound': self.lower_bound.tolist(),
            'upper_bound': self.upper_bound.tolist(),
            'confidence_level': self.confidence_level,
            'model_type': self.model_type,
            'horizon': self.horizon
        }


@dataclass
class DecompositionResult:
    """Time series decomposition result.
    
    Attributes:
        trend: Trend component
        seasonal: Seasonal component
        residual: Residual component
        method: Decomposition method used
        parameters: Method-specific parameters
    """
    trend: Series
    seasonal: Series
    residual: Series
    method: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def reconstruct(self) -> Series:
        """Reconstruct original series from components.
        
        Returns:
            Series reconstructed from trend + seasonal + residual
        """
        return self.trend + self.seasonal + self.residual
    
    def plot(self) -> Any:
        """Generate decomposition plot showing all components.
        
        Returns:
            Matplotlib figure (placeholder for now)
        """
        # Placeholder - will be implemented in visualization module
        raise NotImplementedError("Visualization will be implemented in visualization module")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary with all decomposition result fields
        """
        return {
            'trend': self.trend.tolist(),
            'seasonal': self.seasonal.tolist(),
            'residual': self.residual.tolist(),
            'method': self.method,
            'parameters': self.parameters
        }



@dataclass
class Anomaly:
    """Detected anomaly in data.
    
    Attributes:
        date: Date of anomaly
        value: Anomalous value
        expected_value: Expected value based on model
        z_score: Z-score indicating severity
        description: Human-readable description
    """
    date: date
    value: float
    expected_value: float
    z_score: float
    description: str


@dataclass
class ValidationReport:
    """Data validation report.
    
    Attributes:
        is_valid: Overall validation status
        completeness_score: Score from 0-1 indicating data completeness
        missing_dates: List of dates with missing data
        anomalies: List of detected anomalies
        quality_metrics: Various data quality indicators
        recommendations: Suggested actions for data issues
    """
    is_valid: bool
    completeness_score: float
    missing_dates: List[date]
    anomalies: List[Anomaly]
    quality_metrics: Dict[str, float]
    recommendations: List[str]
    
    def summary(self) -> str:
        """Generate human-readable summary of validation results.
        
        Returns:
            String summary of validation report
        """
        summary_lines = [
            f"Validation Status: {'PASS' if self.is_valid else 'FAIL'}",
            f"Completeness Score: {self.completeness_score:.2%}",
            f"Missing Dates: {len(self.missing_dates)}",
            f"Anomalies Detected: {len(self.anomalies)}",
        ]
        
        if self.quality_metrics:
            summary_lines.append("\nQuality Metrics:")
            for metric, value in self.quality_metrics.items():
                summary_lines.append(f"  {metric}: {value:.4f}")
        
        if self.recommendations:
            summary_lines.append("\nRecommendations:")
            for rec in self.recommendations:
                summary_lines.append(f"  - {rec}")
        
        return "\n".join(summary_lines)


@dataclass
class Changepoint:
    """Detected changepoint in time series.
    
    Attributes:
        date: Date of changepoint
        index: Index position in series
        confidence: Confidence score (0-1)
        magnitude: Size of change
        direction: Direction of change ('increase', 'decrease')
        pre_mean: Mean value before changepoint
        post_mean: Mean value after changepoint
    """
    date: date
    index: int
    confidence: float
    magnitude: float
    direction: str
    pre_mean: float
    post_mean: float
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if changepoint is statistically significant.
        
        Args:
            alpha: Significance level
            
        Returns:
            True if confidence > (1 - alpha)
        """
        return self.confidence > (1 - alpha)


@dataclass
class Finding:
    """Analysis finding with supporting evidence.
    
    Attributes:
        finding_id: Unique identifier for finding
        description: Human-readable description
        evidence: List of statistical test results
        causal_effects: List of causal effect estimates (if applicable)
        confidence_level: Overall confidence ('high', 'medium', 'low')
        requirements_validated: List of requirement IDs this finding validates
    """
    finding_id: str
    description: str
    evidence: List[TestResult]
    causal_effects: List[CausalEffect] = field(default_factory=list)
    confidence_level: str = 'medium'
    requirements_validated: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate human-readable summary of finding.
        
        Returns:
            String summary of finding with evidence
        """
        summary_lines = [
            f"Finding ID: {self.finding_id}",
            f"Confidence: {self.confidence_level.upper()}",
            f"\nDescription:",
            f"  {self.description}",
        ]
        
        if self.evidence:
            summary_lines.append(f"\nStatistical Evidence ({len(self.evidence)} tests):")
            for test in self.evidence:
                sig_marker = "***" if test.is_significant else ""
                summary_lines.append(
                    f"  - {test.test_name}: p={test.p_value:.4f}, "
                    f"effect={test.effect_size:.4f} {sig_marker}"
                )
        
        if self.causal_effects:
            summary_lines.append(f"\nCausal Effects ({len(self.causal_effects)}):")
            for effect in self.causal_effects:
                summary_lines.append(
                    f"  - {effect.method}: effect={effect.effect_size:.4f} "
                    f"({effect.percentage_effect():.2f}%), p={effect.p_value:.4f}"
                )
        
        if self.requirements_validated:
            summary_lines.append(f"\nValidates Requirements: {', '.join(self.requirements_validated)}")
        
        return "\n".join(summary_lines)
    
    def evidence_strength(self) -> float:
        """Calculate aggregate evidence strength score.
        
        Returns:
            Score from 0-1 indicating overall evidence strength
        """
        if not self.evidence:
            return 0.0
        
        # Calculate based on number of significant tests and effect sizes
        significant_tests = sum(1 for test in self.evidence if test.is_significant)
        significance_ratio = significant_tests / len(self.evidence)
        
        # Average effect size (normalized)
        avg_effect = sum(abs(test.effect_size) for test in self.evidence) / len(self.evidence)
        
        # Combine metrics (weighted average)
        strength = 0.6 * significance_ratio + 0.4 * min(avg_effect, 1.0)
        
        return strength
