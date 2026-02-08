"""Confidence interval calculation for statistical validation.

This module implements various methods for calculating confidence intervals
for estimates, including bootstrap and parametric approaches.
"""

from typing import Callable, Tuple, Optional
import numpy as np
from pandas import Series, DataFrame
from scipy import stats
from dataclasses import dataclass


@dataclass
class ConfidenceIntervalCalculator:
    """Calculates confidence intervals for statistical estimates.
    
    Implements bootstrap, parametric, and prediction interval methods
    for quantifying uncertainty in estimates.
    """
    
    def bootstrap_ci(
        self,
        data: Series,
        statistic: Callable[[np.ndarray], float],
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for a statistic.
        
        Uses the percentile bootstrap method to estimate confidence intervals
        for any statistic without assuming a specific distribution.
        
        Args:
            data: Sample data
            statistic: Function that computes the statistic from data
                      (e.g., np.mean, np.median, np.std)
            confidence_level: Confidence level (default 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples to generate
        
        Returns:
            Tuple of (lower_bound, upper_bound) for the confidence interval
        """
        # Remove NaN values
        clean_data = data.dropna().values
        
        if len(clean_data) == 0:
            return (np.nan, np.nan)
        
        # Generate bootstrap samples
        rng = np.random.RandomState(42)  # For reproducibility
        bootstrap_statistics = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = rng.choice(clean_data, size=len(clean_data), replace=True)
            bootstrap_statistics.append(statistic(bootstrap_sample))
        
        bootstrap_statistics = np.array(bootstrap_statistics)
        
        # Calculate percentiles for confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_statistics, lower_percentile)
        upper_bound = np.percentile(bootstrap_statistics, upper_percentile)
        
        return (float(lower_bound), float(upper_bound))
    
    def parametric_ci(
        self,
        data: Series,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate parametric confidence interval for the mean.
        
        Assumes data follows a normal distribution and uses the t-distribution
        to calculate confidence intervals for the population mean.
        
        Args:
            data: Sample data
            confidence_level: Confidence level (default 0.95 for 95% CI)
        
        Returns:
            Tuple of (lower_bound, upper_bound) for the confidence interval
        """
        # Remove NaN values
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            return (np.nan, np.nan)
        
        # Calculate sample statistics
        mean = clean_data.mean()
        std_error = clean_data.sem()  # Standard error of the mean
        n = len(clean_data)
        
        # Calculate t-critical value
        alpha = 1 - confidence_level
        df = n - 1
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        
        # Calculate confidence interval
        margin_of_error = t_critical * std_error
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        
        return (float(lower_bound), float(upper_bound))
    
    def prediction_interval(
        self,
        model: any,
        horizon: int,
        confidence_level: float = 0.95
    ) -> DataFrame:
        """Calculate prediction intervals for forecasts.
        
        Computes prediction intervals for future time points based on
        a fitted forecasting model. The intervals account for both
        parameter uncertainty and future randomness.
        
        Args:
            model: Fitted forecasting model (must have predict method)
            horizon: Number of time periods to forecast
            confidence_level: Confidence level (default 0.95 for 95% CI)
        
        Returns:
            DataFrame with columns: 'lower_bound', 'upper_bound' for each time point
        """
        # Check if model has the required methods
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a 'predict' method")
        
        # Get point forecast
        try:
            forecast = model.predict(n_periods=horizon)
        except TypeError:
            # Try alternative parameter name
            forecast = model.predict(steps=horizon)
        
        # Calculate prediction intervals
        # This is a simplified implementation - actual implementation depends on model type
        if hasattr(model, 'predict_interval'):
            # Model has built-in prediction interval method
            lower, upper = model.predict_interval(
                n_periods=horizon,
                alpha=1 - confidence_level
            )
        elif hasattr(model, 'conf_int'):
            # Alternative method name
            intervals = model.conf_int(alpha=1 - confidence_level)
            lower = intervals[:, 0]
            upper = intervals[:, 1]
        else:
            # Fallback: use residual standard error for approximate intervals
            if hasattr(model, 'resid'):
                residuals = model.resid
                residual_std = np.std(residuals)
            elif hasattr(model, 'residuals'):
                residuals = model.residuals
                residual_std = np.std(residuals)
            else:
                # Last resort: assume 10% error
                residual_std = np.mean(np.abs(forecast)) * 0.1
            
            # Calculate z-score for confidence level
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(1 - alpha / 2)
            
            # Prediction intervals widen with horizon (simplified)
            margin = z_score * residual_std * np.sqrt(1 + np.arange(1, horizon + 1) / horizon)
            lower = forecast - margin
            upper = forecast + margin
        
        # Create DataFrame
        result = DataFrame({
            'lower_bound': lower,
            'upper_bound': upper
        })
        
        return result
    
    def proportion_ci(
        self,
        successes: int,
        trials: int,
        confidence_level: float = 0.95,
        method: str = 'wilson'
    ) -> Tuple[float, float]:
        """Calculate confidence interval for a proportion.
        
        Computes confidence intervals for binomial proportions using
        various methods (Wilson, Clopper-Pearson, or normal approximation).
        
        Args:
            successes: Number of successes
            trials: Total number of trials
            confidence_level: Confidence level (default 0.95 for 95% CI)
            method: Method to use ('wilson', 'clopper-pearson', or 'normal')
        
        Returns:
            Tuple of (lower_bound, upper_bound) for the confidence interval
        """
        if trials == 0:
            return (np.nan, np.nan)
        
        p = successes / trials
        alpha = 1 - confidence_level
        
        if method == 'wilson':
            # Wilson score interval (recommended for most cases)
            z = stats.norm.ppf(1 - alpha / 2)
            denominator = 1 + z**2 / trials
            center = (p + z**2 / (2 * trials)) / denominator
            margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
            
            lower_bound = center - margin
            upper_bound = center + margin
            
        elif method == 'clopper-pearson':
            # Clopper-Pearson exact interval (conservative)
            if successes == 0:
                lower_bound = 0.0
            else:
                lower_bound = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
            
            if successes == trials:
                upper_bound = 1.0
            else:
                upper_bound = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
                
        else:  # normal approximation
            # Normal approximation (Wald interval) - use with caution for small samples
            z = stats.norm.ppf(1 - alpha / 2)
            margin = z * np.sqrt(p * (1 - p) / trials)
            
            lower_bound = max(0.0, p - margin)
            upper_bound = min(1.0, p + margin)
        
        # Clamp bounds to [0, 1] to handle numerical precision issues
        # Use tolerance for values very close to boundaries
        tolerance = 1e-10
        if lower_bound < tolerance:
            lower_bound = 0.0
        if upper_bound > 1.0 - tolerance:
            upper_bound = 1.0
        
        lower_bound = max(0.0, min(1.0, lower_bound))
        upper_bound = max(0.0, min(1.0, upper_bound))
        
        # Ensure lower_bound <= upper_bound
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound
        
        return (float(lower_bound), float(upper_bound))
    
    def difference_ci(
        self,
        sample1: Series,
        sample2: Series,
        confidence_level: float = 0.95,
        method: str = 'parametric'
    ) -> Tuple[float, float]:
        """Calculate confidence interval for difference between two means.
        
        Args:
            sample1: First sample data
            sample2: Second sample data
            confidence_level: Confidence level (default 0.95 for 95% CI)
            method: Method to use ('parametric' or 'bootstrap')
        
        Returns:
            Tuple of (lower_bound, upper_bound) for the confidence interval
        """
        # Remove NaN values
        s1 = sample1.dropna()
        s2 = sample2.dropna()
        
        if len(s1) == 0 or len(s2) == 0:
            return (np.nan, np.nan)
        
        if method == 'bootstrap':
            # Bootstrap method
            def diff_statistic(data1, data2):
                return np.mean(data1) - np.mean(data2)
            
            rng = np.random.RandomState(42)
            bootstrap_diffs = []
            
            for _ in range(10000):
                boot_s1 = rng.choice(s1, size=len(s1), replace=True)
                boot_s2 = rng.choice(s2, size=len(s2), replace=True)
                bootstrap_diffs.append(diff_statistic(boot_s1, boot_s2))
            
            alpha = 1 - confidence_level
            lower_bound = np.percentile(bootstrap_diffs, (alpha / 2) * 100)
            upper_bound = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)
            
        else:  # parametric
            # Parametric method using t-distribution
            mean_diff = s1.mean() - s2.mean()
            
            # Pooled standard error
            n1, n2 = len(s1), len(s2)
            pooled_std = np.sqrt(((n1 - 1) * s1.std() ** 2 + 
                                  (n2 - 1) * s2.std() ** 2) / 
                                 (n1 + n2 - 2))
            se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
            
            # t-critical value
            df = n1 + n2 - 2
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha / 2, df)
            
            # Confidence interval
            margin = t_critical * se_diff
            lower_bound = mean_diff - margin
            upper_bound = mean_diff + margin
        
        return (float(lower_bound), float(upper_bound))
