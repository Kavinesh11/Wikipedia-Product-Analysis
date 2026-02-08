"""Interrupted Time Series Analysis (ITSA) for causal inference.

This module implements interrupted time series analysis with segmented regression
to estimate causal effects of interventions.
"""

from datetime import date
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from pandas import Series
from scipy import stats
from sklearn.linear_model import LinearRegression
from wikipedia_health.models import CausalEffect, TestResult


class ITSAModel:
    """Fitted interrupted time series model.
    
    Attributes:
        pre_model: Regression model for pre-intervention period
        post_model: Regression model for post-intervention period
        intervention_date: Date of intervention
        pre_period_length: Length of pre-intervention period in days
        series: Original time series data
    """
    
    def __init__(
        self,
        pre_model: LinearRegression,
        post_model: LinearRegression,
        intervention_date: date,
        pre_period_length: int,
        series: Series
    ):
        self.pre_model = pre_model
        self.post_model = post_model
        self.intervention_date = intervention_date
        self.pre_period_length = pre_period_length
        self.series = series


class InterruptedTimeSeriesAnalyzer:
    """Interrupted Time Series Analysis for causal inference.
    
    Implements segmented regression to estimate the causal effect of an
    intervention by comparing observed post-intervention values to a
    counterfactual baseline predicted from pre-intervention trends.
    """
    
    def __init__(self, min_pre_period: int = 60):
        """Initialize ITSA analyzer.
        
        Args:
            min_pre_period: Minimum required pre-intervention period in days
        """
        self.min_pre_period = min_pre_period
    
    def fit(
        self,
        series: Series,
        intervention_date: date,
        pre_period_length: int = 90
    ) -> ITSAModel:
        """Fit interrupted time series model with segmented regression.
        
        Args:
            series: Time series data with DatetimeIndex
            intervention_date: Date when intervention occurred
            pre_period_length: Length of pre-intervention period to use (days)
            
        Returns:
            Fitted ITSAModel
            
        Raises:
            ValueError: If pre_period_length < min_pre_period or insufficient data
        """
        if pre_period_length < self.min_pre_period:
            raise ValueError(
                f"Pre-intervention period must be at least {self.min_pre_period} days, "
                f"got {pre_period_length}"
            )
        
        # Ensure series has DatetimeIndex
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series must have DatetimeIndex")
        
        # Convert intervention_date to Timestamp
        intervention_ts = pd.Timestamp(intervention_date)
        
        # Split data into pre and post intervention periods
        pre_data = series[series.index < intervention_ts]
        post_data = series[series.index >= intervention_ts]
        
        if len(pre_data) < pre_period_length:
            raise ValueError(
                f"Insufficient pre-intervention data: need {pre_period_length} days, "
                f"got {len(pre_data)}"
            )
        
        if len(post_data) == 0:
            raise ValueError("No post-intervention data available")
        
        # Use only the specified pre_period_length
        pre_data = pre_data.iloc[-pre_period_length:]
        
        # Fit pre-intervention model (linear trend)
        pre_X = np.arange(len(pre_data)).reshape(-1, 1)
        pre_y = pre_data.values
        pre_model = LinearRegression()
        pre_model.fit(pre_X, pre_y)
        
        # Fit post-intervention model (linear trend)
        post_X = np.arange(len(post_data)).reshape(-1, 1)
        post_y = post_data.values
        post_model = LinearRegression()
        post_model.fit(post_X, post_y)
        
        return ITSAModel(
            pre_model=pre_model,
            post_model=post_model,
            intervention_date=intervention_date,
            pre_period_length=pre_period_length,
            series=series
        )
    
    def construct_counterfactual(
        self,
        model: ITSAModel,
        post_period: Tuple[date, date]
    ) -> Series:
        """Construct counterfactual baseline for post-intervention period.
        
        Predicts what would have happened without the intervention by
        extrapolating the pre-intervention trend.
        
        Args:
            model: Fitted ITSAModel
            post_period: Tuple of (start_date, end_date) for prediction
            
        Returns:
            Series of counterfactual predictions with DatetimeIndex
        """
        start_date, end_date = post_period
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        # Get post-intervention data
        post_data = model.series[
            (model.series.index >= start_ts) & (model.series.index <= end_ts)
        ]
        
        if len(post_data) == 0:
            raise ValueError("No data in specified post period")
        
        # Extrapolate pre-intervention trend
        # X values continue from where pre-period ended
        pre_period_end = model.pre_period_length
        X_counterfactual = np.arange(
            pre_period_end,
            pre_period_end + len(post_data)
        ).reshape(-1, 1)
        
        counterfactual_values = model.pre_model.predict(X_counterfactual)
        
        return Series(
            counterfactual_values,
            index=post_data.index,
            name='counterfactual'
        )
    
    def estimate_effect(
        self,
        model: ITSAModel,
        post_period_length: int = 90,
        confidence_level: float = 0.95
    ) -> CausalEffect:
        """Estimate average treatment effect with confidence interval.
        
        Args:
            model: Fitted ITSAModel
            post_period_length: Length of post-intervention period to analyze
            confidence_level: Confidence level for interval (default 0.95)
            
        Returns:
            CausalEffect with effect size, CI, and p-value
        """
        intervention_ts = pd.Timestamp(model.intervention_date)
        
        # Get post-intervention data
        post_data = model.series[model.series.index >= intervention_ts]
        
        if len(post_data) > post_period_length:
            post_data = post_data.iloc[:post_period_length]
        
        # Construct counterfactual
        end_date = post_data.index[-1].date()
        counterfactual = self.construct_counterfactual(
            model,
            (model.intervention_date, end_date)
        )
        
        # Calculate treatment effect (observed - counterfactual)
        treatment_effects = post_data.values - counterfactual.values
        
        # Average treatment effect (ATE)
        ate = np.mean(treatment_effects)
        
        # Calculate confidence interval using bootstrap
        n_bootstrap = 1000
        bootstrap_ates = []
        
        for _ in range(n_bootstrap):
            # Resample treatment effects with replacement
            resampled = np.random.choice(
                treatment_effects,
                size=len(treatment_effects),
                replace=True
            )
            bootstrap_ates.append(np.mean(resampled))
        
        # Calculate CI from bootstrap distribution
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        ci_lower = np.percentile(bootstrap_ates, lower_percentile)
        ci_upper = np.percentile(bootstrap_ates, upper_percentile)
        
        # Calculate p-value using t-test against null hypothesis (effect = 0)
        t_stat, p_value = stats.ttest_1samp(treatment_effects, 0)
        
        return CausalEffect(
            effect_size=ate,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method='ITSA',
            counterfactual=counterfactual,
            observed=post_data,
            treatment_period=(model.intervention_date, end_date)
        )
    
    def test_parallel_trends(
        self,
        model: ITSAModel,
        alpha: float = 0.05
    ) -> TestResult:
        """Test parallel trends assumption for ITSA validity.
        
        Tests whether the pre-intervention trend is stable by checking
        for structural breaks in the pre-period.
        
        Args:
            model: Fitted ITSAModel
            alpha: Significance level for test
            
        Returns:
            TestResult with test statistics and interpretation
        """
        intervention_ts = pd.Timestamp(model.intervention_date)
        
        # Get pre-intervention data
        pre_data = model.series[model.series.index < intervention_ts]
        pre_data = pre_data.iloc[-model.pre_period_length:]
        
        # Split pre-period in half
        mid_point = len(pre_data) // 2
        first_half = pre_data.iloc[:mid_point]
        second_half = pre_data.iloc[mid_point:]
        
        # Fit linear models to each half
        X1 = np.arange(len(first_half)).reshape(-1, 1)
        y1 = first_half.values
        model1 = LinearRegression()
        model1.fit(X1, y1)
        
        X2 = np.arange(len(second_half)).reshape(-1, 1)
        y2 = second_half.values
        model2 = LinearRegression()
        model2.fit(X2, y2)
        
        # Compare slopes (parallel trends means similar slopes)
        slope1 = model1.coef_[0]
        slope2 = model2.coef_[0]
        
        # Calculate residuals for each model
        residuals1 = y1 - model1.predict(X1)
        residuals2 = y2 - model2.predict(X2)
        
        # Pooled standard error
        pooled_var = (
            np.sum(residuals1**2) + np.sum(residuals2**2)
        ) / (len(y1) + len(y2) - 4)
        
        # Standard error of slope difference
        se_diff = np.sqrt(
            pooled_var * (1/np.sum((X1 - X1.mean())**2) + 1/np.sum((X2 - X2.mean())**2))
        )
        
        # T-statistic for slope difference
        if se_diff > 0:
            t_stat = (slope1 - slope2) / se_diff
            df = len(y1) + len(y2) - 4
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            t_stat = 0.0
            p_value = 1.0
        
        is_significant = p_value < alpha
        
        if is_significant:
            interpretation = (
                f"Parallel trends assumption VIOLATED (p={p_value:.4f}). "
                f"Pre-intervention trend is not stable. Consider using alternative methods."
            )
        else:
            interpretation = (
                f"Parallel trends assumption SATISFIED (p={p_value:.4f}). "
                f"Pre-intervention trend is stable."
            )
        
        return TestResult(
            test_name='Parallel Trends Test (ITSA)',
            statistic=t_stat,
            p_value=p_value,
            effect_size=abs(slope1 - slope2),
            confidence_interval=(slope1 - slope2 - 1.96*se_diff, slope1 - slope2 + 1.96*se_diff),
            is_significant=is_significant,
            alpha=alpha,
            interpretation=interpretation
        )
