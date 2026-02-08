"""Effect size calculation for statistical validation.

This module implements various effect size measures for quantifying
the magnitude of differences and relationships in data.
"""

from typing import Tuple
import numpy as np
from pandas import Series
from dataclasses import dataclass


@dataclass
class EffectSizeCalculator:
    """Calculates effect sizes for statistical comparisons.
    
    Implements various effect size measures including Cohen's d, Hedges' g,
    percentage change, and absolute differences with confidence intervals.
    """
    
    def cohens_d(
        self,
        sample1: Series,
        sample2: Series
    ) -> float:
        """Calculate Cohen's d effect size.
        
        Cohen's d measures the standardized difference between two means.
        It represents the difference in standard deviation units.
        
        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large
        
        Args:
            sample1: First sample data
            sample2: Second sample data
        
        Returns:
            Cohen's d effect size
        """
        # Remove NaN values
        s1 = sample1.dropna()
        s2 = sample2.dropna()
        
        if len(s1) == 0 or len(s2) == 0:
            return np.nan
        
        # Calculate means
        mean1 = s1.mean()
        mean2 = s2.mean()
        
        # Calculate pooled standard deviation
        n1, n2 = len(s1), len(s2)
        var1 = s1.var(ddof=1)
        var2 = s2.var(ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Calculate Cohen's d
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (mean1 - mean2) / pooled_std
        
        return float(cohens_d)
    
    def hedges_g(
        self,
        sample1: Series,
        sample2: Series
    ) -> float:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d).
        
        Hedges' g is a bias-corrected version of Cohen's d that provides
        a more accurate estimate for small sample sizes.
        
        Args:
            sample1: First sample data
            sample2: Second sample data
        
        Returns:
            Hedges' g effect size
        """
        # Calculate Cohen's d first
        d = self.cohens_d(sample1, sample2)
        
        if np.isnan(d):
            return np.nan
        
        # Calculate correction factor
        n1 = len(sample1.dropna())
        n2 = len(sample2.dropna())
        n = n1 + n2
        
        # Correction factor (approximation)
        correction = 1 - (3 / (4 * n - 9))
        
        # Apply correction
        hedges_g = d * correction
        
        return float(hedges_g)
    
    def percentage_change(
        self,
        baseline: float,
        treatment: float
    ) -> float:
        """Calculate percentage change from baseline to treatment.
        
        Computes the relative change as a percentage.
        
        Args:
            baseline: Baseline value
            treatment: Treatment/comparison value
        
        Returns:
            Percentage change ((treatment - baseline) / baseline * 100)
        """
        # Handle very small baseline values (near zero)
        if abs(baseline) < 1e-10:
            # Treat as zero baseline
            if abs(treatment) < 1e-10:
                return 0.0
            else:
                return np.inf if treatment > 0 else -np.inf
        
        pct_change = ((treatment - baseline) / baseline) * 100
        
        return float(pct_change)
    
    def absolute_difference(
        self,
        sample1: Series,
        sample2: Series
    ) -> Tuple[float, Tuple[float, float]]:
        """Calculate absolute difference between means with confidence interval.
        
        Computes the raw difference between two sample means along with
        a 95% confidence interval for the difference.
        
        Args:
            sample1: First sample data
            sample2: Second sample data
        
        Returns:
            Tuple of (difference, (ci_lower, ci_upper))
        """
        # Remove NaN values
        s1 = sample1.dropna()
        s2 = sample2.dropna()
        
        if len(s1) == 0 or len(s2) == 0:
            return (np.nan, (np.nan, np.nan))
        
        # Calculate difference
        diff = s1.mean() - s2.mean()
        
        # Calculate confidence interval using t-distribution
        n1, n2 = len(s1), len(s2)
        
        # Pooled standard deviation
        var1 = s1.var(ddof=1)
        var2 = s2.var(ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Standard error of difference
        se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
        
        # t-critical value for 95% CI
        from scipy import stats
        df = n1 + n2 - 2
        t_critical = stats.t.ppf(0.975, df)
        
        # Confidence interval
        margin = t_critical * se_diff
        ci_lower = diff - margin
        ci_upper = diff + margin
        
        return (float(diff), (float(ci_lower), float(ci_upper)))
    
    def relative_risk(
        self,
        exposed_events: int,
        exposed_total: int,
        control_events: int,
        control_total: int
    ) -> Tuple[float, Tuple[float, float]]:
        """Calculate relative risk with confidence interval.
        
        Relative risk (risk ratio) measures the ratio of probabilities
        of an event occurring in two groups.
        
        Args:
            exposed_events: Number of events in exposed group
            exposed_total: Total number in exposed group
            control_events: Number of events in control group
            control_total: Total number in control group
        
        Returns:
            Tuple of (relative_risk, (ci_lower, ci_upper))
        """
        if exposed_total == 0 or control_total == 0:
            return (np.nan, (np.nan, np.nan))
        
        # Calculate proportions
        p_exposed = exposed_events / exposed_total
        p_control = control_events / control_total
        
        if p_control == 0:
            return (np.inf, (np.nan, np.nan))
        
        # Calculate relative risk
        rr = p_exposed / p_control
        
        # Calculate confidence interval using log transformation
        if exposed_events == 0 or control_events == 0:
            # Cannot calculate CI with zero events
            return (float(rr), (np.nan, np.nan))
        
        from scipy import stats
        
        # Standard error of log(RR)
        se_log_rr = np.sqrt(
            (1 / exposed_events) - (1 / exposed_total) +
            (1 / control_events) - (1 / control_total)
        )
        
        # 95% CI on log scale
        z_critical = stats.norm.ppf(0.975)
        log_rr = np.log(rr)
        log_ci_lower = log_rr - z_critical * se_log_rr
        log_ci_upper = log_rr + z_critical * se_log_rr
        
        # Transform back to original scale
        ci_lower = np.exp(log_ci_lower)
        ci_upper = np.exp(log_ci_upper)
        
        return (float(rr), (float(ci_lower), float(ci_upper)))
    
    def odds_ratio(
        self,
        exposed_events: int,
        exposed_total: int,
        control_events: int,
        control_total: int
    ) -> Tuple[float, Tuple[float, float]]:
        """Calculate odds ratio with confidence interval.
        
        Odds ratio measures the odds of an event in one group relative
        to the odds in another group.
        
        Args:
            exposed_events: Number of events in exposed group
            exposed_total: Total number in exposed group
            control_events: Number of events in control group
            control_total: Total number in control group
        
        Returns:
            Tuple of (odds_ratio, (ci_lower, ci_upper))
        """
        # Calculate non-events
        exposed_nonevents = exposed_total - exposed_events
        control_nonevents = control_total - control_events
        
        if exposed_nonevents == 0 or control_nonevents == 0:
            return (np.inf, (np.nan, np.nan))
        
        if exposed_events == 0 or control_events == 0:
            return (0.0, (np.nan, np.nan))
        
        # Calculate odds ratio
        odds_exposed = exposed_events / exposed_nonevents
        odds_control = control_events / control_nonevents
        
        if odds_control == 0:
            return (np.inf, (np.nan, np.nan))
        
        or_value = odds_exposed / odds_control
        
        # Calculate confidence interval using log transformation
        from scipy import stats
        
        # Standard error of log(OR)
        se_log_or = np.sqrt(
            (1 / exposed_events) + (1 / exposed_nonevents) +
            (1 / control_events) + (1 / control_nonevents)
        )
        
        # 95% CI on log scale
        z_critical = stats.norm.ppf(0.975)
        log_or = np.log(or_value)
        log_ci_lower = log_or - z_critical * se_log_or
        log_ci_upper = log_or + z_critical * se_log_or
        
        # Transform back to original scale
        ci_lower = np.exp(log_ci_lower)
        ci_upper = np.exp(log_ci_upper)
        
        return (float(or_value), (float(ci_lower), float(ci_upper)))
    
    def correlation_effect_size(
        self,
        sample1: Series,
        sample2: Series
    ) -> Tuple[float, Tuple[float, float]]:
        """Calculate Pearson correlation as effect size with confidence interval.
        
        Correlation coefficient measures the strength and direction of
        linear relationship between two variables.
        
        Args:
            sample1: First variable data
            sample2: Second variable data
        
        Returns:
            Tuple of (correlation, (ci_lower, ci_upper))
        """
        from scipy import stats
        
        # Remove NaN values (pairwise)
        combined = np.column_stack([sample1, sample2])
        combined = combined[~np.isnan(combined).any(axis=1)]
        
        if len(combined) < 3:
            return (np.nan, (np.nan, np.nan))
        
        s1 = combined[:, 0]
        s2 = combined[:, 1]
        
        # Calculate correlation
        r, _ = stats.pearsonr(s1, s2)
        
        # Calculate confidence interval using Fisher's z-transformation
        n = len(s1)
        
        # Fisher's z-transformation
        z = np.arctanh(r)
        se_z = 1 / np.sqrt(n - 3)
        
        # 95% CI on z scale
        z_critical = stats.norm.ppf(0.975)
        z_ci_lower = z - z_critical * se_z
        z_ci_upper = z + z_critical * se_z
        
        # Transform back to correlation scale
        ci_lower = np.tanh(z_ci_lower)
        ci_upper = np.tanh(z_ci_upper)
        
        return (float(r), (float(ci_lower), float(ci_upper)))
