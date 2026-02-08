"""Multi-dimensional analyzer for correlating pageviews with engagement metrics.

This module implements the MultiDimensionalAnalyzer class that correlates
pageview patterns with user engagement depth, editor activity, and content
quality metrics to distinguish between passive consumption and active engagement.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from scipy import stats
from dataclasses import dataclass

from wikipedia_health.models.data_models import TimeSeriesData, TestResult, Anomaly
from wikipedia_health.statistical_validation.hypothesis_tester import HypothesisTester


@dataclass
class EngagementMetrics:
    """Container for engagement analysis results.
    
    Attributes:
        engagement_ratio: Editors per 1000 pageviews
        correlation: Correlation coefficient between pageviews and editors
        correlation_p_value: P-value for correlation significance
        platform: Platform identifier
        time_period: Tuple of (start_date, end_date)
    """
    engagement_ratio: float
    correlation: float
    correlation_p_value: float
    platform: str
    time_period: Tuple[str, str]


@dataclass
class EngagementShift:
    """Detected shift in engagement patterns.
    
    Attributes:
        date: Date of shift detection
        pre_ratio: Engagement ratio before shift
        post_ratio: Engagement ratio after shift
        test_result: Statistical test result for shift significance
        direction: Direction of shift ('increase', 'decrease')
    """
    date: str
    pre_ratio: float
    post_ratio: float
    test_result: TestResult
    direction: str


@dataclass
class PlatformEngagement:
    """Platform-specific engagement comparison.
    
    Attributes:
        platform: Platform identifier
        engagement_ratio: Editors per 1000 pageviews
        pageview_mean: Mean pageviews
        editor_mean: Mean active editors
        engagement_quality: Quality classification ('high', 'medium', 'low')
    """
    platform: str
    engagement_ratio: float
    pageview_mean: float
    editor_mean: float
    engagement_quality: str


class MultiDimensionalAnalyzer:
    """Analyzes multi-dimensional engagement patterns.
    
    Correlates pageview patterns with editor activity and edit volume to
    distinguish between passive consumption spikes and active engagement increases.
    Implements Requirements 4.1-4.6.
    """
    
    def __init__(self):
        """Initialize the multi-dimensional analyzer."""
        self.hypothesis_tester = HypothesisTester()
    
    def correlate_pageviews_editors(
        self,
        pageviews: TimeSeriesData,
        editors: TimeSeriesData
    ) -> EngagementMetrics:
        """Correlate pageview changes with editor activity.
        
        Computes correlation between pageviews and active editor counts to
        detect engagement quality shifts. High correlation suggests that
        traffic increases are accompanied by increased active participation.
        
        Validates: Requirements 4.1, 4.2
        
        Args:
            pageviews: Time series of pageview data
            editors: Time series of active editor counts
        
        Returns:
            EngagementMetrics with correlation coefficient, p-value, and engagement ratio
        
        Raises:
            ValueError: If time series have different lengths or date ranges
        """
        # Align time series by date
        pv_df = pageviews.to_dataframe()
        ed_df = editors.to_dataframe()
        
        # Ensure dates are datetime
        pv_df['date'] = pd.to_datetime(pv_df['date'])
        ed_df['date'] = pd.to_datetime(ed_df['date'])
        
        # Merge on date
        merged = pd.merge(
            pv_df[['date', 'values']].rename(columns={'values': 'pageviews'}),
            ed_df[['date', 'values']].rename(columns={'values': 'editors'}),
            on='date',
            how='inner'
        )
        
        if len(merged) == 0:
            raise ValueError("No overlapping dates between pageviews and editors data")
        
        # Calculate correlation
        # Handle constant input (zero variance) case
        if merged['pageviews'].std() == 0 or merged['editors'].std() == 0:
            # If either series has zero variance, correlation is undefined
            # Set to 0.0 with p-value of 1.0 (no correlation)
            correlation = 0.0
            p_value = 1.0
        else:
            correlation, p_value = stats.pearsonr(merged['pageviews'], merged['editors'])
        
        # Calculate engagement ratio (editors per 1000 pageviews)
        engagement_ratio = self.compute_engagement_ratio(
            merged['pageviews'],
            merged['editors']
        )
        
        # Determine time period
        time_period = (
            str(merged['date'].min().date()),
            str(merged['date'].max().date())
        )
        
        return EngagementMetrics(
            engagement_ratio=engagement_ratio,
            correlation=float(correlation),
            correlation_p_value=float(p_value),
            platform=pageviews.platform,
            time_period=time_period
        )
    
    def compute_engagement_ratio(
        self,
        pageviews: Series,
        editors: Series
    ) -> float:
        """Compute engagement ratio (editors per 1000 pageviews).
        
        The engagement ratio serves as a proxy for engagement depth.
        Higher ratios indicate more active participation relative to
        passive consumption.
        
        Validates: Requirement 4.2
        
        Args:
            pageviews: Series of pageview counts
            editors: Series of active editor counts
        
        Returns:
            Mean engagement ratio (editors per 1000 pageviews)
        """
        # Avoid division by zero
        pageviews_safe = pageviews.replace(0, np.nan)
        
        # Calculate ratio for each time point
        ratios = (editors / pageviews_safe) * 1000
        
        # Return mean ratio (excluding NaN values)
        return float(ratios.mean())
    
    def detect_engagement_shifts(
        self,
        pageviews: TimeSeriesData,
        editors: TimeSeriesData,
        window_size: int = 30
    ) -> List[EngagementShift]:
        """Detect statistically significant shifts in engagement patterns.
        
        Uses rolling window analysis to detect periods where the engagement
        ratio (editors per 1000 pageviews) changes significantly. Performs
        statistical tests to confirm shifts are not due to random variation.
        
        Validates: Requirement 4.3
        
        Args:
            pageviews: Time series of pageview data
            editors: Time series of active editor counts
            window_size: Size of rolling window for shift detection (days)
        
        Returns:
            List of detected engagement shifts with statistical evidence
        """
        # Align time series
        pv_df = pageviews.to_dataframe()
        ed_df = editors.to_dataframe()
        
        pv_df['date'] = pd.to_datetime(pv_df['date'])
        ed_df['date'] = pd.to_datetime(ed_df['date'])
        
        merged = pd.merge(
            pv_df[['date', 'values']].rename(columns={'values': 'pageviews'}),
            ed_df[['date', 'values']].rename(columns={'values': 'editors'}),
            on='date',
            how='inner'
        ).sort_values('date')
        
        # Calculate engagement ratio for each time point
        merged['engagement_ratio'] = (merged['editors'] / merged['pageviews'].replace(0, np.nan)) * 1000
        
        # Detect shifts using rolling window comparison
        shifts = []
        
        # Scan through time series with sliding window
        for i in range(window_size, len(merged) - window_size):
            pre_window = merged.iloc[i-window_size:i]['engagement_ratio'].dropna()
            post_window = merged.iloc[i:i+window_size]['engagement_ratio'].dropna()
            
            # Skip if insufficient data
            if len(pre_window) < 10 or len(post_window) < 10:
                continue
            
            # Perform t-test to detect significant shift
            test_result = self.hypothesis_tester.t_test(
                pre_window,
                post_window,
                alternative='two-sided'
            )
            
            # If significant shift detected
            if test_result.is_significant:
                direction = 'increase' if post_window.mean() > pre_window.mean() else 'decrease'
                
                shift = EngagementShift(
                    date=str(merged.iloc[i]['date'].date()),
                    pre_ratio=float(pre_window.mean()),
                    post_ratio=float(post_window.mean()),
                    test_result=test_result,
                    direction=direction
                )
                shifts.append(shift)
                
                # Skip ahead to avoid detecting same shift multiple times
                i += window_size
        
        return shifts
    
    def cross_reference_anomalies(
        self,
        pageview_anomalies: List[Anomaly],
        editor_anomalies: List[Anomaly],
        time_window_days: int = 3
    ) -> Dict[str, List[Tuple[Anomaly, Optional[Anomaly]]]]:
        """Cross-reference anomalies across pageviews and editor activity.
        
        Distinguishes between passive consumption spikes (pageview anomalies
        without corresponding editor anomalies) and active engagement increases
        (anomalies in both metrics).
        
        Validates: Requirement 4.4
        
        Args:
            pageview_anomalies: List of detected pageview anomalies
            editor_anomalies: List of detected editor activity anomalies
            time_window_days: Time window for matching anomalies (days)
        
        Returns:
            Dictionary with three categories:
            - 'passive_consumption': Pageview anomalies without editor anomalies
            - 'active_engagement': Pageview anomalies with matching editor anomalies
            - 'editor_only': Editor anomalies without pageview anomalies
        """
        passive_consumption = []
        active_engagement = []
        editor_only = []
        
        # Convert dates to datetime for comparison
        pv_dates = {pd.to_datetime(a.date): a for a in pageview_anomalies}
        ed_dates = {pd.to_datetime(a.date): a for a in editor_anomalies}
        
        # Check each pageview anomaly for matching editor anomaly
        for pv_date, pv_anomaly in pv_dates.items():
            matched = False
            
            for ed_date, ed_anomaly in ed_dates.items():
                # Check if within time window
                days_diff = abs((pv_date - ed_date).days)
                
                if days_diff <= time_window_days:
                    active_engagement.append((pv_anomaly, ed_anomaly))
                    matched = True
                    break
            
            if not matched:
                passive_consumption.append((pv_anomaly, None))
        
        # Find editor anomalies without matching pageview anomalies
        for ed_date, ed_anomaly in ed_dates.items():
            matched = False
            
            for pv_date in pv_dates.keys():
                days_diff = abs((ed_date - pv_date).days)
                
                if days_diff <= time_window_days:
                    matched = True
                    break
            
            if not matched:
                editor_only.append((ed_anomaly, None))
        
        return {
            'passive_consumption': passive_consumption,
            'active_engagement': active_engagement,
            'editor_only': editor_only
        }
    
    def compare_platform_engagement(
        self,
        platform_data: Dict[str, Tuple[TimeSeriesData, TimeSeriesData]]
    ) -> Tuple[List[PlatformEngagement], TestResult]:
        """Compare engagement metrics across platforms.
        
        Analyzes engagement patterns across desktop, mobile web, and mobile app
        to identify platform-specific behavior patterns. Tests whether engagement
        ratios differ significantly across platforms.
        
        Validates: Requirements 4.5, 4.6
        
        Args:
            platform_data: Dictionary mapping platform names to tuples of
                          (pageviews, editors) TimeSeriesData
        
        Returns:
            Tuple of:
            - List of PlatformEngagement objects for each platform
            - TestResult from ANOVA testing platform differences
        """
        platform_engagements = []
        engagement_ratios_by_platform = []
        
        for platform, (pageviews, editors) in platform_data.items():
            # Align time series
            pv_df = pageviews.to_dataframe()
            ed_df = editors.to_dataframe()
            
            pv_df['date'] = pd.to_datetime(pv_df['date'])
            ed_df['date'] = pd.to_datetime(ed_df['date'])
            
            merged = pd.merge(
                pv_df[['date', 'values']].rename(columns={'values': 'pageviews'}),
                ed_df[['date', 'values']].rename(columns={'values': 'editors'}),
                on='date',
                how='inner'
            )
            
            # Calculate metrics
            pageview_mean = float(merged['pageviews'].mean())
            editor_mean = float(merged['editors'].mean())
            
            # Calculate engagement ratio
            engagement_ratio = self.compute_engagement_ratio(
                merged['pageviews'],
                merged['editors']
            )
            
            # Classify engagement quality
            engagement_quality = self._classify_engagement_quality(engagement_ratio)
            
            platform_engagements.append(PlatformEngagement(
                platform=platform,
                engagement_ratio=engagement_ratio,
                pageview_mean=pageview_mean,
                editor_mean=editor_mean,
                engagement_quality=engagement_quality
            ))
            
            # Store ratios for ANOVA
            ratios = (merged['editors'] / merged['pageviews'].replace(0, np.nan)) * 1000
            engagement_ratios_by_platform.append(ratios.dropna())
        
        # Perform ANOVA to test for significant differences across platforms
        if len(engagement_ratios_by_platform) >= 2:
            anova_result = self.hypothesis_tester.anova(engagement_ratios_by_platform)
        else:
            # Not enough platforms for ANOVA
            anova_result = TestResult(
                test_name='One-Way ANOVA',
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                alpha=0.05,
                interpretation='Insufficient platforms for comparison (need at least 2)'
            )
        
        return platform_engagements, anova_result
    
    def _classify_engagement_quality(self, engagement_ratio: float) -> str:
        """Classify engagement quality based on ratio thresholds.
        
        Args:
            engagement_ratio: Editors per 1000 pageviews
        
        Returns:
            Quality classification: 'high', 'medium', or 'low'
        """
        # Thresholds based on typical Wikipedia engagement patterns
        # These are heuristic values that could be calibrated with real data
        if engagement_ratio >= 1.0:
            return 'high'
        elif engagement_ratio >= 0.5:
            return 'medium'
        else:
            return 'low'
