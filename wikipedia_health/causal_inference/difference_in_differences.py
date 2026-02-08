"""Difference-in-Differences (DiD) analysis for causal inference.

This module implements difference-in-differences methodology to estimate
causal effects by comparing treatment and control groups before and after
an intervention.
"""

from datetime import date
from typing import Tuple
import numpy as np
import pandas as pd
from pandas import Series
from scipy import stats
from sklearn.linear_model import LinearRegression
from wikipedia_health.models import CausalEffect, TestResult


class DiDModel:
    """Fitted Difference-in-Differences model.
    
    Attributes:
        treatment_series: Treatment group time series
        control_series: Control group time series
        intervention_date: Date of intervention
        pre_treatment_mean: Mean of treatment group before intervention
        post_treatment_mean: Mean of treatment group after intervention
        pre_control_mean: Mean of control group before intervention
        post_control_mean: Mean of control group after intervention
    """
    
    def __init__(
        self,
        treatment_series: Series,
        control_series: Series,
        intervention_date: date,
        pre_treatment_mean: float,
        post_treatment_mean: float,
        pre_control_mean: float,
        post_control_mean: float
    ):
        self.treatment_series = treatment_series
        self.control_series = control_series
        self.intervention_date = intervention_date
        self.pre_treatment_mean = pre_treatment_mean
        self.post_treatment_mean = post_treatment_mean
        self.pre_control_mean = pre_control_mean
        self.post_control_mean = post_control_mean


class DifferenceInDifferencesAnalyzer:
    """Difference-in-Differences analysis for causal inference.
    
    Estimates causal effects by comparing changes in treatment group
    to changes in control group, controlling for time-invariant confounders.
    """
    
    def __init__(self, min_pre_period: int = 60):
        """Initialize DiD analyzer.
        
        Args:
            min_pre_period: Minimum required pre-intervention period in days
        """
        self.min_pre_period = min_pre_period
    
    def fit(
        self,
        treatment_series: Series,
        control_series: Series,
        intervention_date: date
    ) -> DiDModel:
        """Fit difference-in-differences model.
        
        Args:
            treatment_series: Time series for treatment group (with DatetimeIndex)
            control_series: Time series for control group (with DatetimeIndex)
            intervention_date: Date when intervention occurred
            
        Returns:
            Fitted DiDModel
            
        Raises:
            ValueError: If insufficient data or misaligned series
        """
        # Ensure both series have DatetimeIndex
        if not isinstance(treatment_series.index, pd.DatetimeIndex):
            raise ValueError("Treatment series must have DatetimeIndex")
        if not isinstance(control_series.index, pd.DatetimeIndex):
            raise ValueError("Control series must have DatetimeIndex")
        
        # Convert intervention_date to Timestamp
        intervention_ts = pd.Timestamp(intervention_date)
        
        # Split into pre and post periods
        pre_treatment = treatment_series[treatment_series.index < intervention_ts]
        post_treatment = treatment_series[treatment_series.index >= intervention_ts]
        pre_control = control_series[control_series.index < intervention_ts]
        post_control = control_series[control_series.index >= intervention_ts]
        
        # Validate sufficient data
        if len(pre_treatment) < self.min_pre_period:
            raise ValueError(
                f"Insufficient pre-intervention treatment data: need {self.min_pre_period} days, "
                f"got {len(pre_treatment)}"
            )
        if len(pre_control) < self.min_pre_period:
            raise ValueError(
                f"Insufficient pre-intervention control data: need {self.min_pre_period} days, "
                f"got {len(pre_control)}"
            )
        if len(post_treatment) == 0 or len(post_control) == 0:
            raise ValueError("No post-intervention data available")
        
        # Calculate means for each group and period
        pre_treatment_mean = pre_treatment.mean()
        post_treatment_mean = post_treatment.mean()
        pre_control_mean = pre_control.mean()
        post_control_mean = post_control.mean()
        
        return DiDModel(
            treatment_series=treatment_series,
            control_series=control_series,
            intervention_date=intervention_date,
            pre_treatment_mean=pre_treatment_mean,
            post_treatment_mean=post_treatment_mean,
            pre_control_mean=pre_control_mean,
            post_control_mean=post_control_mean
        )
    
    def estimate_effect(
        self,
        model: DiDModel,
        confidence_level: float = 0.95
    ) -> CausalEffect:
        """Estimate DiD treatment effect with confidence interval.
        
        The DiD estimator is:
        (post_treatment - pre_treatment) - (post_control - pre_control)
        
        Args:
            model: Fitted DiDModel
            confidence_level: Confidence level for interval (default 0.95)
            
        Returns:
            CausalEffect with DiD estimate, CI, and p-value
        """
        intervention_ts = pd.Timestamp(model.intervention_date)
        
        # Get data splits
        pre_treatment = model.treatment_series[model.treatment_series.index < intervention_ts]
        post_treatment = model.treatment_series[model.treatment_series.index >= intervention_ts]
        pre_control = model.control_series[model.control_series.index < intervention_ts]
        post_control = model.control_series[model.control_series.index >= intervention_ts]
        
        # Calculate DiD estimator
        treatment_diff = model.post_treatment_mean - model.pre_treatment_mean
        control_diff = model.post_control_mean - model.pre_control_mean
        did_estimate = treatment_diff - control_diff
        
        # Calculate standard error using regression approach
        # Create panel data structure
        n_pre_treat = len(pre_treatment)
        n_post_treat = len(post_treatment)
        n_pre_control = len(pre_control)
        n_post_control = len(post_control)
        
        # Construct regression variables
        y = np.concatenate([
            pre_treatment.values,
            post_treatment.values,
            pre_control.values,
            post_control.values
        ])
        
        # Treatment indicator (1 for treatment group, 0 for control)
        treat = np.concatenate([
            np.ones(n_pre_treat + n_post_treat),
            np.zeros(n_pre_control + n_post_control)
        ])
        
        # Post indicator (1 for post-period, 0 for pre-period)
        post = np.concatenate([
            np.zeros(n_pre_treat),
            np.ones(n_post_treat),
            np.zeros(n_pre_control),
            np.ones(n_post_control)
        ])
        
        # Interaction term (treatment * post) - this is the DiD estimator
        treat_post = treat * post
        
        # Fit regression: y = β0 + β1*treat + β2*post + β3*treat*post + ε
        X = np.column_stack([treat, post, treat_post])
        reg = LinearRegression()
        reg.fit(X, y)
        
        # DiD coefficient is the third coefficient (treat_post)
        did_coef = reg.coef_[2]
        
        # Calculate standard error
        y_pred = reg.predict(X)
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (len(y) - X.shape[1] - 1)
        
        # Variance-covariance matrix
        XtX_inv = np.linalg.inv(X.T @ X)
        var_covar = mse * XtX_inv
        se_did = np.sqrt(var_covar[2, 2])
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        df = len(y) - X.shape[1] - 1
        t_crit = stats.t.ppf(1 - alpha/2, df)
        ci_lower = did_coef - t_crit * se_did
        ci_upper = did_coef + t_crit * se_did
        
        # Calculate p-value
        t_stat = did_coef / se_did if se_did > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Construct counterfactual for treatment group
        # Counterfactual = pre_treatment_mean + control_diff
        counterfactual_mean = model.pre_treatment_mean + control_diff
        counterfactual = Series(
            [counterfactual_mean] * len(post_treatment),
            index=post_treatment.index,
            name='counterfactual'
        )
        
        return CausalEffect(
            effect_size=did_estimate,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            method='DiD',
            counterfactual=counterfactual,
            observed=post_treatment,
            treatment_period=(model.intervention_date, post_treatment.index[-1].date())
        )
    
    def test_parallel_trends(
        self,
        model: DiDModel,
        pre_period: Tuple[date, date],
        alpha: float = 0.05
    ) -> TestResult:
        """Test parallel trends assumption in pre-intervention period.
        
        Tests whether treatment and control groups had parallel trends
        before the intervention - a critical assumption for DiD validity.
        
        Args:
            model: Fitted DiDModel
            pre_period: Tuple of (start_date, end_date) for pre-period to test
            alpha: Significance level for test
            
        Returns:
            TestResult with test statistics and interpretation
        """
        start_ts = pd.Timestamp(pre_period[0])
        end_ts = pd.Timestamp(pre_period[1])
        intervention_ts = pd.Timestamp(model.intervention_date)
        
        # Get pre-intervention data for specified period
        pre_treatment = model.treatment_series[
            (model.treatment_series.index >= start_ts) &
            (model.treatment_series.index < intervention_ts) &
            (model.treatment_series.index <= end_ts)
        ]
        pre_control = model.control_series[
            (model.control_series.index >= start_ts) &
            (model.control_series.index < intervention_ts) &
            (model.control_series.index <= end_ts)
        ]
        
        if len(pre_treatment) < 10 or len(pre_control) < 10:
            raise ValueError("Insufficient data for parallel trends test (need at least 10 points)")
        
        # Fit linear trends to both series
        X_treat = np.arange(len(pre_treatment)).reshape(-1, 1)
        y_treat = pre_treatment.values
        model_treat = LinearRegression()
        model_treat.fit(X_treat, y_treat)
        slope_treat = model_treat.coef_[0]
        
        X_control = np.arange(len(pre_control)).reshape(-1, 1)
        y_control = pre_control.values
        model_control = LinearRegression()
        model_control.fit(X_control, y_control)
        slope_control = model_control.coef_[0]
        
        # Calculate residuals
        residuals_treat = y_treat - model_treat.predict(X_treat)
        residuals_control = y_control - model_control.predict(X_control)
        
        # Pooled variance
        pooled_var = (
            np.sum(residuals_treat**2) + np.sum(residuals_control**2)
        ) / (len(y_treat) + len(y_control) - 4)
        
        # Standard error of slope difference
        se_diff = np.sqrt(
            pooled_var * (
                1/np.sum((X_treat - X_treat.mean())**2) +
                1/np.sum((X_control - X_control.mean())**2)
            )
        )
        
        # T-statistic for slope difference
        if se_diff > 0:
            t_stat = (slope_treat - slope_control) / se_diff
            df = len(y_treat) + len(y_control) - 4
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            t_stat = 0.0
            p_value = 1.0
        
        is_significant = p_value < alpha
        
        if is_significant:
            interpretation = (
                f"Parallel trends assumption VIOLATED (p={p_value:.4f}). "
                f"Treatment and control groups had different pre-intervention trends. "
                f"DiD estimates may be biased."
            )
        else:
            interpretation = (
                f"Parallel trends assumption SATISFIED (p={p_value:.4f}). "
                f"Treatment and control groups had similar pre-intervention trends."
            )
        
        return TestResult(
            test_name='Parallel Trends Test (DiD)',
            statistic=t_stat,
            p_value=p_value,
            effect_size=abs(slope_treat - slope_control),
            confidence_interval=(
                slope_treat - slope_control - 1.96*se_diff,
                slope_treat - slope_control + 1.96*se_diff
            ),
            is_significant=is_significant,
            alpha=alpha,
            interpretation=interpretation
        )
    
    def placebo_test(
        self,
        model: DiDModel,
        placebo_date: date,
        alpha: float = 0.05
    ) -> TestResult:
        """Perform placebo test for robustness check.
        
        Tests for a "fake" intervention at an earlier date. If DiD is valid,
        we should find no significant effect at the placebo date.
        
        Args:
            model: Fitted DiDModel
            placebo_date: Date for placebo intervention (must be before real intervention)
            alpha: Significance level for test
            
        Returns:
            TestResult with placebo test results
        """
        placebo_ts = pd.Timestamp(placebo_date)
        intervention_ts = pd.Timestamp(model.intervention_date)
        
        if placebo_ts >= intervention_ts:
            raise ValueError("Placebo date must be before actual intervention date")
        
        # Get data before actual intervention
        pre_intervention_treatment = model.treatment_series[
            model.treatment_series.index < intervention_ts
        ]
        pre_intervention_control = model.control_series[
            model.control_series.index < intervention_ts
        ]
        
        # Split at placebo date
        placebo_pre_treatment = pre_intervention_treatment[
            pre_intervention_treatment.index < placebo_ts
        ]
        placebo_post_treatment = pre_intervention_treatment[
            pre_intervention_treatment.index >= placebo_ts
        ]
        placebo_pre_control = pre_intervention_control[
            pre_intervention_control.index < placebo_ts
        ]
        placebo_post_control = pre_intervention_control[
            pre_intervention_control.index >= placebo_ts
        ]
        
        if (len(placebo_pre_treatment) < 30 or len(placebo_post_treatment) < 30 or
            len(placebo_pre_control) < 30 or len(placebo_post_control) < 30):
            raise ValueError("Insufficient data for placebo test (need at least 30 points in each period)")
        
        # Calculate placebo DiD estimate
        treatment_diff = placebo_post_treatment.mean() - placebo_pre_treatment.mean()
        control_diff = placebo_post_control.mean() - placebo_pre_control.mean()
        placebo_did = treatment_diff - control_diff
        
        # Calculate standard error using t-test approach
        # Difference in differences for each observation
        treat_diffs = placebo_post_treatment.values - placebo_pre_treatment.mean()
        control_diffs = placebo_post_control.values - placebo_pre_control.mean()
        
        # Standard error of the difference
        se_treat = np.std(treat_diffs, ddof=1) / np.sqrt(len(treat_diffs))
        se_control = np.std(control_diffs, ddof=1) / np.sqrt(len(control_diffs))
        se_did = np.sqrt(se_treat**2 + se_control**2)
        
        # T-statistic
        if se_did > 0:
            t_stat = placebo_did / se_did
            df = len(treat_diffs) + len(control_diffs) - 2
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            t_stat = 0.0
            p_value = 1.0
        
        is_significant = p_value < alpha
        
        if is_significant:
            interpretation = (
                f"Placebo test FAILED (p={p_value:.4f}). "
                f"Significant effect detected at placebo date, suggesting DiD assumptions "
                f"may be violated or confounding factors present."
            )
        else:
            interpretation = (
                f"Placebo test PASSED (p={p_value:.4f}). "
                f"No significant effect at placebo date, supporting DiD validity."
            )
        
        return TestResult(
            test_name='Placebo Test (DiD)',
            statistic=t_stat,
            p_value=p_value,
            effect_size=placebo_did,
            confidence_interval=(
                placebo_did - 1.96*se_did,
                placebo_did + 1.96*se_did
            ),
            is_significant=is_significant,
            alpha=alpha,
            interpretation=interpretation
        )
