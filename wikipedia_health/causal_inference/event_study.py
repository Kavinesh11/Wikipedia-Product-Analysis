"""Event Study methodology for causal inference.

This module implements event study analysis to measure the impact of external
events by comparing observed values to baseline predictions.
"""

from datetime import date, timedelta
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from pandas import Series
from scipy import stats
from wikipedia_health.models import CausalEffect, TestResult
from wikipedia_health.time_series.forecaster import Forecaster, ARIMAModel, ProphetModel


class BaselineModel:
    """Baseline forecast model for event study.
    
    Attributes:
        model: Fitted forecast model (ARIMA or Prophet)
        model_type: Type of model ('arima' or 'prophet')
        baseline_window: Number of days used for baseline
        event_date: Date of the event
    """
    
    def __init__(
        self,
        model,
        model_type: str,
        baseline_window: int,
        event_date: date
    ):
        self.model = model
        self.model_type = model_type
        self.baseline_window = baseline_window
        self.event_date = event_date


class EventImpact:
    """Event impact measurement result.
    
    Attributes:
        observed: Observed values during event period
        predicted: Predicted values (baseline forecast)
        difference: Observed - predicted (abnormal returns)
        confidence_interval: CI for cumulative abnormal return
        car: Cumulative abnormal return
    """
    
    def __init__(
        self,
        observed: Series,
        predicted: Series,
        difference: Series,
        confidence_interval: Tuple[float, float],
        car: float
    ):
        self.observed = observed
        self.predicted = predicted
        self.difference = difference
        self.confidence_interval = confidence_interval
        self.car = car


class EventStudyAnalyzer:
    """Event Study analysis for measuring external event impacts.
    
    Uses baseline forecasting to construct counterfactual predictions,
    then measures abnormal returns during the event period.
    """
    
    def __init__(self, min_baseline_window: int = 60):
        """Initialize event study analyzer.
        
        Args:
            min_baseline_window: Minimum baseline period in days
        """
        self.min_baseline_window = min_baseline_window
        self.forecaster = Forecaster()
    
    def fit_baseline(
        self,
        series: Series,
        event_date: date,
        baseline_window: int = 90,
        method: str = 'arima'
    ) -> BaselineModel:
        """Fit baseline model using pre-event data.
        
        Args:
            series: Time series data with DatetimeIndex
            event_date: Date of the event
            baseline_window: Number of days before event to use for baseline
            method: Forecasting method ('arima' or 'prophet')
            
        Returns:
            BaselineModel fitted on pre-event data
            
        Raises:
            ValueError: If insufficient data or invalid method
        """
        if baseline_window < self.min_baseline_window:
            raise ValueError(
                f"Baseline window must be at least {self.min_baseline_window} days, "
                f"got {baseline_window}"
            )
        
        # Ensure series has DatetimeIndex
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series must have DatetimeIndex")
        
        # Get pre-event data
        event_ts = pd.Timestamp(event_date)
        pre_event = series[series.index < event_ts]
        
        if len(pre_event) < baseline_window:
            raise ValueError(
                f"Insufficient pre-event data: need {baseline_window} days, "
                f"got {len(pre_event)}"
            )
        
        # Use only the specified baseline window
        baseline_data = pre_event.iloc[-baseline_window:]
        
        # Fit baseline model
        if method.lower() == 'arima':
            model = self.forecaster.fit_arima(baseline_data)
            model_type = 'arima'
        elif method.lower() == 'prophet':
            model = self.forecaster.fit_prophet(baseline_data)
            model_type = 'prophet'
        else:
            raise ValueError(f"Unknown method: {method}. Use 'arima' or 'prophet'")
        
        return BaselineModel(
            model=model,
            model_type=model_type,
            baseline_window=baseline_window,
            event_date=event_date
        )
    
    def estimate_event_impact(
        self,
        series: Series,
        baseline: BaselineModel,
        event_date: date,
        post_window: int = 30,
        confidence_level: float = 0.95
    ) -> EventImpact:
        """Estimate event impact by comparing observed to predicted values.
        
        Calculates cumulative abnormal return (CAR) as the sum of differences
        between observed and predicted values.
        
        Args:
            series: Time series data with DatetimeIndex
            baseline: Fitted baseline model
            event_date: Date of the event
            post_window: Number of days after event to analyze
            confidence_level: Confidence level for intervals
            
        Returns:
            EventImpact with observed, predicted, and abnormal returns
        """
        event_ts = pd.Timestamp(event_date)
        
        # Get post-event data
        post_event = series[series.index >= event_ts]
        
        if len(post_event) == 0:
            raise ValueError("No post-event data available")
        
        # Limit to post_window
        if len(post_event) > post_window:
            post_event = post_event.iloc[:post_window]
        
        actual_window = len(post_event)
        
        # Generate baseline forecast
        forecast_result = self.forecaster.forecast(
            baseline.model,
            horizon=actual_window,
            confidence_level=confidence_level
        )
        
        # Align forecast with post-event dates
        predicted = Series(
            forecast_result.point_forecast.values,
            index=post_event.index,
            name='predicted'
        )
        
        # Calculate abnormal returns (observed - predicted)
        abnormal_returns = post_event.values - predicted.values
        
        # Cumulative abnormal return (CAR)
        car = np.sum(abnormal_returns)
        
        # Calculate confidence interval for CAR using bootstrap
        n_bootstrap = 1000
        bootstrap_cars = []
        
        for _ in range(n_bootstrap):
            # Resample abnormal returns with replacement
            resampled = np.random.choice(
                abnormal_returns,
                size=len(abnormal_returns),
                replace=True
            )
            bootstrap_cars.append(np.sum(resampled))
        
        # Calculate CI from bootstrap distribution
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        ci_lower = np.percentile(bootstrap_cars, lower_percentile)
        ci_upper = np.percentile(bootstrap_cars, upper_percentile)
        
        return EventImpact(
            observed=post_event,
            predicted=predicted,
            difference=Series(abnormal_returns, index=post_event.index, name='abnormal_return'),
            confidence_interval=(ci_lower, ci_upper),
            car=car
        )
    
    def test_significance(
        self,
        impact: EventImpact,
        alpha: float = 0.05
    ) -> Tuple[bool, float]:
        """Test whether event impact is statistically significant.
        
        Uses z-scores to test if observed values exceed prediction intervals.
        
        Args:
            impact: EventImpact from estimate_event_impact
            alpha: Significance level
            
        Returns:
            Tuple of (is_significant, p_value)
        """
        # Test if CAR is significantly different from zero
        abnormal_returns = impact.difference.values
        
        # Calculate z-score for CAR
        mean_ar = np.mean(abnormal_returns)
        std_ar = np.std(abnormal_returns, ddof=1)
        n = len(abnormal_returns)
        
        if std_ar > 0:
            # Standard error of CAR
            se_car = std_ar * np.sqrt(n)
            z_score = impact.car / se_car
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            z_score = 0.0
            p_value = 1.0
        
        is_significant = p_value < alpha
        
        return is_significant, p_value
    
    def measure_persistence(
        self,
        series: Series,
        event_date: date,
        max_window: int = 180,
        threshold: float = 0.1
    ) -> int:
        """Measure persistence of event impact (half-life).
        
        Calculates how many days it takes for the impact to decay back
        to baseline levels.
        
        Args:
            series: Time series data with DatetimeIndex
            event_date: Date of the event
            max_window: Maximum window to search for return to baseline
            threshold: Threshold for considering "returned to baseline" (as fraction)
            
        Returns:
            Number of days until return to baseline (or max_window if not returned)
        """
        event_ts = pd.Timestamp(event_date)
        
        # Get pre-event baseline mean
        pre_event = series[series.index < event_ts]
        if len(pre_event) < 30:
            raise ValueError("Insufficient pre-event data for baseline calculation")
        
        baseline_mean = pre_event.iloc[-90:].mean() if len(pre_event) >= 90 else pre_event.mean()
        
        # Get post-event data
        post_event = series[series.index >= event_ts]
        
        if len(post_event) == 0:
            raise ValueError("No post-event data available")
        
        # Limit to max_window
        if len(post_event) > max_window:
            post_event = post_event.iloc[:max_window]
        
        # Calculate rolling mean to smooth out noise
        window_size = min(7, len(post_event) // 4)
        if window_size < 1:
            window_size = 1
        
        rolling_mean = post_event.rolling(window=window_size, min_periods=1).mean()
        
        # Find when rolling mean returns to within threshold of baseline
        threshold_value = abs(baseline_mean * threshold)
        
        for i, value in enumerate(rolling_mean):
            if abs(value - baseline_mean) <= threshold_value:
                return i + 1  # Return day number (1-indexed)
        
        # If never returns to baseline within max_window
        return max_window
    
    def analyze_event(
        self,
        series: Series,
        event_date: date,
        baseline_window: int = 90,
        post_window: int = 30,
        method: str = 'arima',
        confidence_level: float = 0.95,
        alpha: float = 0.05
    ) -> CausalEffect:
        """Complete event study analysis.
        
        Convenience method that performs full event study: fits baseline,
        estimates impact, tests significance, and returns CausalEffect.
        
        Args:
            series: Time series data with DatetimeIndex
            event_date: Date of the event
            baseline_window: Days before event for baseline
            post_window: Days after event to analyze
            method: Forecasting method ('arima' or 'prophet')
            confidence_level: Confidence level for intervals
            alpha: Significance level for testing
            
        Returns:
            CausalEffect with event impact estimate
        """
        # Fit baseline model
        baseline = self.fit_baseline(
            series,
            event_date,
            baseline_window=baseline_window,
            method=method
        )
        
        # Estimate event impact
        impact = self.estimate_event_impact(
            series,
            baseline,
            event_date,
            post_window=post_window,
            confidence_level=confidence_level
        )
        
        # Test significance
        is_significant, p_value = self.test_significance(impact, alpha=alpha)
        
        # Calculate end date
        end_date = impact.observed.index[-1].date()
        
        return CausalEffect(
            effect_size=impact.car,
            confidence_interval=impact.confidence_interval,
            p_value=p_value,
            method='EventStudy',
            counterfactual=impact.predicted,
            observed=impact.observed,
            treatment_period=(event_date, end_date)
        )
