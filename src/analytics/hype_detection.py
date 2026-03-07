"""Hype Detection Engine

Identifies trending topics and calculates hype scores based on multiple signals.
"""
from typing import List
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np

from src.storage.dto import HypeMetrics, SpikeEvent

logger = logging.getLogger(__name__)


class HypeDetectionEngine:
    """Detect trending topics and calculate hype scores
    
    Analyzes view velocity, edit growth, and content expansion to identify
    articles experiencing rapid attention growth and distinguish between
    sustained trends and temporary spikes.
    """
    
    def __init__(self, hype_threshold: float = 0.75):
        """Initialize hype detection engine
        
        Args:
            hype_threshold: Hype score threshold for trending flag (0-1)
        """
        if not 0 <= hype_threshold <= 1:
            raise ValueError("hype_threshold must be between 0 and 1")
        
        self.hype_threshold = hype_threshold
        logger.info(f"HypeDetectionEngine initialized with hype_threshold={hype_threshold}")
    
    def calculate_hype_score(
        self,
        view_velocity: float,
        edit_growth: float,
        content_expansion: float
    ) -> float:
        """Calculate composite hype score
        
        Combines view velocity, edit growth, and content expansion into a
        weighted hype score normalized to 0-1 range.
        
        Formula:
            hype_score = (0.5 * normalized_view_velocity + 
                         0.3 * normalized_edit_growth + 
                         0.2 * normalized_content_expansion)
        
        Args:
            view_velocity: Rate of pageview growth (percentage)
            edit_growth: Rate of edit activity growth (percentage)
            content_expansion: Rate of content size growth (percentage)
            
        Returns:
            Hype score (0-1)
            
        Validates: Requirements 9.1
        """
        # Normalize view velocity (cap at 1000% growth for normalization)
        normalized_view_velocity = min(abs(view_velocity) / 1000.0, 1.0)
        
        # Normalize edit growth (cap at 500% growth for normalization)
        normalized_edit_growth = min(abs(edit_growth) / 500.0, 1.0)
        
        # Normalize content expansion (cap at 300% growth for normalization)
        normalized_content_expansion = min(abs(content_expansion) / 300.0, 1.0)
        
        # Weighted combination (view velocity weighted highest)
        hype_score = (
            0.5 * normalized_view_velocity +
            0.3 * normalized_edit_growth +
            0.2 * normalized_content_expansion
        )
        
        logger.debug(
            f"Calculated hype score: {hype_score:.3f} "
            f"(view_velocity={view_velocity:.2f}%, "
            f"edit_growth={edit_growth:.2f}%, "
            f"content_expansion={content_expansion:.2f}%)"
        )
        
        return hype_score
    
    def calculate_attention_density(
        self,
        pageviews: pd.DataFrame,
        window_days: int = 7
    ) -> float:
        """Calculate sustained attention over time window
        
        Measures attention density as total pageviews divided by window duration.
        
        Formula:
            attention_density = total_pageviews / window_duration_days
        
        Args:
            pageviews: DataFrame with 'date' and 'views' columns
            window_days: Time window in days
            
        Returns:
            Attention density (views per day)
            
        Validates: Requirements 9.3
        """
        if pageviews.empty:
            return 0.0
        
        # Ensure we have required columns
        if 'views' not in pageviews.columns:
            raise ValueError("pageviews DataFrame must have 'views' column")
        
        # Calculate total views in the window
        total_views = pageviews['views'].sum()
        
        # Calculate attention density
        attention_density = total_views / window_days
        
        logger.debug(
            f"Calculated attention density: {attention_density:.2f} views/day "
            f"({total_views} total views over {window_days} days)"
        )
        
        return attention_density
    
    def detect_attention_spikes(
        self,
        pageviews: pd.DataFrame
    ) -> List[SpikeEvent]:
        """Identify sudden attention increases
        
        Detects spikes where pageviews exceed the rolling mean by more than
        2 standard deviations.
        
        Args:
            pageviews: DataFrame with 'date' and 'views' columns
            
        Returns:
            List of detected spike events
            
        Validates: Requirements 9.4
        """
        if pageviews.empty or len(pageviews) < 3:
            return []
        
        # Ensure we have required columns
        if 'date' not in pageviews.columns or 'views' not in pageviews.columns:
            raise ValueError("pageviews DataFrame must have 'date' and 'views' columns")
        
        # Sort by date
        pageviews = pageviews.sort_values('date').reset_index(drop=True)
        
        # Calculate rolling statistics (7-day window)
        window_size = min(7, len(pageviews) - 1)
        if window_size < 2:
            return []
        
        rolling_mean = pageviews['views'].rolling(window=window_size, min_periods=1).mean()
        rolling_std = pageviews['views'].rolling(window=window_size, min_periods=2).std()
        
        # Replace NaN std with 0
        rolling_std = rolling_std.fillna(0)
        
        # Detect spikes (views > mean + 2*std)
        spike_events = []
        in_spike = False
        spike_start_idx = None
        
        for idx in range(len(pageviews)):
            views = pageviews.loc[idx, 'views']
            mean = rolling_mean.iloc[idx]
            std = rolling_std.iloc[idx]
            
            # Check if this is a spike
            threshold = mean + 2 * std
            is_spike = views > threshold and std > 0
            
            if is_spike and not in_spike:
                # Start of new spike
                in_spike = True
                spike_start_idx = idx
            elif not is_spike and in_spike:
                # End of spike
                in_spike = False
                spike_duration = idx - spike_start_idx
                
                # Calculate magnitude
                spike_views = pageviews.loc[spike_start_idx:idx-1, 'views']
                spike_mean = rolling_mean.iloc[spike_start_idx:idx].mean()
                spike_std = rolling_std.iloc[spike_start_idx:idx].mean()
                
                if spike_std > 0:
                    magnitude = (spike_views.mean() - spike_mean) / spike_std
                else:
                    magnitude = 0.0
                
                # Classify spike type
                spike_type = self.distinguish_spike_types(
                    SpikeEvent(
                        timestamp=pageviews.loc[spike_start_idx, 'date'],
                        magnitude=magnitude,
                        duration_days=spike_duration,
                        spike_type="temporary"  # Will be updated
                    )
                ).spike_type
                
                spike_events.append(
                    SpikeEvent(
                        timestamp=pageviews.loc[spike_start_idx, 'date'],
                        magnitude=magnitude,
                        duration_days=spike_duration,
                        spike_type=spike_type
                    )
                )
        
        # Handle case where spike continues to end of data
        if in_spike and spike_start_idx is not None:
            spike_duration = len(pageviews) - spike_start_idx
            spike_views = pageviews.loc[spike_start_idx:, 'views']
            spike_mean = rolling_mean.iloc[spike_start_idx:].mean()
            spike_std = rolling_std.iloc[spike_start_idx:].mean()
            
            if spike_std > 0:
                magnitude = (spike_views.mean() - spike_mean) / spike_std
            else:
                magnitude = 0.0
            
            spike_type = self.distinguish_spike_types(
                SpikeEvent(
                    timestamp=pageviews.loc[spike_start_idx, 'date'],
                    magnitude=magnitude,
                    duration_days=spike_duration,
                    spike_type="temporary"
                )
            ).spike_type
            
            spike_events.append(
                SpikeEvent(
                    timestamp=pageviews.loc[spike_start_idx, 'date'],
                    magnitude=magnitude,
                    duration_days=spike_duration,
                    spike_type=spike_type
                )
            )
        
        logger.debug(f"Detected {len(spike_events)} attention spikes")
        
        return spike_events
    
    def distinguish_spike_types(
        self,
        spike: SpikeEvent
    ) -> SpikeEvent:
        """Classify spike as sustained growth or temporary spike
        
        Classifies as "sustained growth" if pageviews remain elevated for >7 days,
        otherwise "temporary spike".
        
        Args:
            spike: Spike event to classify
            
        Returns:
            Spike event with updated spike_type
            
        Validates: Requirements 9.5
        """
        # Classify based on duration
        if spike.duration_days > 7:
            spike_type = "sustained"
        else:
            spike_type = "temporary"
        
        logger.debug(
            f"Classified spike as {spike_type}: "
            f"duration={spike.duration_days} days, "
            f"magnitude={spike.magnitude:.2f} std devs"
        )
        
        return SpikeEvent(
            timestamp=spike.timestamp,
            magnitude=spike.magnitude,
            duration_days=spike.duration_days,
            spike_type=spike_type
        )
    
    def calculate_hype_metrics(
        self,
        article: str,
        pageviews: pd.DataFrame,
        view_velocity: float,
        edit_growth: float,
        content_expansion: float,
        window_days: int = 7
    ) -> HypeMetrics:
        """Calculate complete hype metrics for an article
        
        Helper method to compute all hype-related metrics.
        
        Args:
            article: Article title
            pageviews: DataFrame with pageview time series
            view_velocity: Rate of pageview growth (percentage)
            edit_growth: Rate of edit activity growth (percentage)
            content_expansion: Rate of content size growth (percentage)
            window_days: Time window for attention density calculation
            
        Returns:
            HypeMetrics with complete hype analysis
        """
        # Calculate hype score
        hype_score = self.calculate_hype_score(
            view_velocity,
            edit_growth,
            content_expansion
        )
        
        # Calculate attention density
        attention_density = self.calculate_attention_density(
            pageviews,
            window_days
        )
        
        # Detect spikes
        spike_events = self.detect_attention_spikes(pageviews)
        
        # Determine if trending
        is_trending = hype_score >= self.hype_threshold
        
        logger.info(
            f"Calculated hype metrics for {article}: "
            f"hype_score={hype_score:.3f}, "
            f"is_trending={is_trending}, "
            f"spikes={len(spike_events)}"
        )
        
        return HypeMetrics(
            article=article,
            hype_score=hype_score,
            view_velocity=view_velocity,
            edit_growth=edit_growth,
            content_expansion=content_expansion,
            attention_density=attention_density,
            is_trending=is_trending,
            spike_events=spike_events
        )
