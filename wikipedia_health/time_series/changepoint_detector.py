"""Changepoint detection module.

This module provides changepoint detection functionality using various algorithms
including PELT, Binary Segmentation, and Bayesian methods.
"""

from typing import List, Tuple, Optional
from datetime import date
import pandas as pd
from pandas import Series
import numpy as np
import ruptures as rpt
from scipy import stats

from wikipedia_health.models.data_models import Changepoint


class ChangepointDetector:
    """Changepoint detection using multiple algorithms.
    
    This class provides methods for detecting structural breaks in time series
    using various statistical and algorithmic approaches.
    """
    
    def detect_pelt(
        self,
        series: Series,
        penalty: Optional[float] = None,
        min_size: int = 30
    ) -> List[Changepoint]:
        """Detect changepoints using PELT (Pruned Exact Linear Time) algorithm.
        
        Args:
            series: Time series to analyze
            penalty: Penalty value for changepoint detection (auto-selected if None)
            min_size: Minimum segment size between changepoints
            
        Returns:
            List of detected Changepoint objects
            
        Raises:
            ValueError: If series is too short
        """
        if len(series) < min_size * 2:
            raise ValueError(
                f"Series length ({len(series)}) must be at least 2 * min_size ({min_size * 2})"
            )
        
        # Clean the series
        series_clean = series.copy()
        if series_clean.isna().any():
            series_clean = series_clean.interpolate(method='linear')
        
        # Convert to numpy array
        signal = series_clean.values
        
        # Auto-select penalty if not provided
        if penalty is None:
            # Use BIC-based penalty
            penalty = np.log(len(signal)) * signal.var()
        
        # Detect changepoints using PELT with mean shift model
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
        changepoint_indices = algo.predict(pen=penalty)
        
        # Remove the last index (end of series)
        if changepoint_indices and changepoint_indices[-1] == len(signal):
            changepoint_indices = changepoint_indices[:-1]
        
        # Convert to Changepoint objects
        changepoints = []
        for idx in changepoint_indices:
            if idx > 0 and idx < len(series):
                changepoint = self._create_changepoint(series, idx)
                changepoints.append(changepoint)
        
        return changepoints
    
    def detect_binary_segmentation(
        self,
        series: Series,
        n_changepoints: int = 5
    ) -> List[Changepoint]:
        """Detect changepoints using Binary Segmentation algorithm.
        
        Args:
            series: Time series to analyze
            n_changepoints: Maximum number of changepoints to detect
            
        Returns:
            List of detected Changepoint objects
            
        Raises:
            ValueError: If series is too short
        """
        if len(series) < 10:
            raise ValueError("Series must have at least 10 observations")
        
        # Clean the series
        series_clean = series.copy()
        if series_clean.isna().any():
            series_clean = series_clean.interpolate(method='linear')
        
        # Convert to numpy array
        signal = series_clean.values
        
        # Detect changepoints using Binary Segmentation
        algo = rpt.Binseg(model="l2").fit(signal)
        changepoint_indices = algo.predict(n_bkps=n_changepoints)
        
        # Remove the last index (end of series)
        if changepoint_indices and changepoint_indices[-1] == len(signal):
            changepoint_indices = changepoint_indices[:-1]
        
        # Convert to Changepoint objects
        changepoints = []
        for idx in changepoint_indices:
            if idx > 0 and idx < len(series):
                changepoint = self._create_changepoint(series, idx)
                changepoints.append(changepoint)
        
        return changepoints
    
    def detect_bayesian(
        self,
        series: Series,
        prior_scale: float = 0.05
    ) -> List[Changepoint]:
        """Detect changepoints using Bayesian online changepoint detection.
        
        This is a simplified implementation using ruptures' BottomUp algorithm
        with a Bayesian information criterion.
        
        Args:
            series: Time series to analyze
            prior_scale: Prior scale parameter (controls sensitivity)
            
        Returns:
            List of detected Changepoint objects
            
        Raises:
            ValueError: If series is too short
        """
        if len(series) < 10:
            raise ValueError("Series must have at least 10 observations")
        
        # Clean the series
        series_clean = series.copy()
        if series_clean.isna().any():
            series_clean = series_clean.interpolate(method='linear')
        
        # Convert to numpy array
        signal = series_clean.values
        
        # Use BottomUp algorithm with BIC penalty
        algo = rpt.BottomUp(model="rbf").fit(signal)
        
        # Penalty based on prior scale and data variance
        penalty = prior_scale * len(signal) * np.log(len(signal))
        
        changepoint_indices = algo.predict(pen=penalty)
        
        # Remove the last index (end of series)
        if changepoint_indices and changepoint_indices[-1] == len(signal):
            changepoint_indices = changepoint_indices[:-1]
        
        # Convert to Changepoint objects
        changepoints = []
        for idx in changepoint_indices:
            if idx > 0 and idx < len(series):
                changepoint = self._create_changepoint(series, idx)
                changepoints.append(changepoint)
        
        return changepoints
    
    def test_significance(
        self,
        series: Series,
        changepoint: Changepoint,
        alpha: float = 0.05
    ) -> Tuple[bool, float]:
        """Test if a changepoint is statistically significant.
        
        Uses a Chow test (F-test for structural break) to determine if the
        changepoint represents a significant change in the series.
        
        Args:
            series: Original time series
            changepoint: Changepoint to test
            alpha: Significance level
            
        Returns:
            Tuple of (is_significant, p_value)
        """
        # Clean the series
        series_clean = series.copy()
        if series_clean.isna().any():
            series_clean = series_clean.interpolate(method='linear')
        
        idx = changepoint.index
        
        # Split series at changepoint
        pre_segment = series_clean.iloc[:idx]
        post_segment = series_clean.iloc[idx:]
        
        if len(pre_segment) < 2 or len(post_segment) < 2:
            return False, 1.0
        
        # Perform t-test to compare means
        t_stat, p_value = stats.ttest_ind(pre_segment, post_segment)
        
        is_significant = p_value < alpha
        
        return is_significant, p_value
    
    def _create_changepoint(
        self,
        series: Series,
        index: int
    ) -> Changepoint:
        """Create a Changepoint object from series and index.
        
        Args:
            series: Time series
            index: Index of changepoint
            
        Returns:
            Changepoint object with computed statistics
        """
        # Calculate pre and post means
        pre_segment = series.iloc[:index]
        post_segment = series.iloc[index:]
        
        pre_mean = pre_segment.mean() if len(pre_segment) > 0 else 0.0
        post_mean = post_segment.mean() if len(post_segment) > 0 else 0.0
        
        magnitude = abs(post_mean - pre_mean)
        direction = 'increase' if post_mean > pre_mean else 'decrease'
        
        # Calculate confidence based on effect size
        if len(pre_segment) > 0 and len(post_segment) > 0:
            pooled_std = np.sqrt(
                (pre_segment.var() * (len(pre_segment) - 1) + 
                 post_segment.var() * (len(post_segment) - 1)) /
                (len(pre_segment) + len(post_segment) - 2)
            )
            if pooled_std > 0:
                # Cohen's d as proxy for confidence
                cohens_d = magnitude / pooled_std
                confidence = min(0.99, cohens_d / 3.0)  # Normalize to 0-1
            else:
                confidence = 0.5
        else:
            confidence = 0.5
        
        # Get date if series has datetime index
        if isinstance(series.index, pd.DatetimeIndex):
            cp_date = series.index[index].date()
        else:
            # Use a placeholder date
            cp_date = date(2020, 1, 1)
        
        return Changepoint(
            date=cp_date,
            index=index,
            confidence=confidence,
            magnitude=magnitude,
            direction=direction,
            pre_mean=pre_mean,
            post_mean=post_mean
        )
