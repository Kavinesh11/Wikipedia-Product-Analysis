"""Time series decomposition module.

This module provides time series decomposition functionality using various methods
including STL (Seasonal and Trend decomposition using Loess) and X-13-ARIMA-SEATS.
"""

from typing import Optional
import pandas as pd
from pandas import Series
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.filters.hp_filter import hpfilter

from wikipedia_health.models.data_models import DecompositionResult


class TimeSeriesDecomposer:
    """Time series decomposition using multiple methods.
    
    This class provides methods for decomposing time series into trend, seasonal,
    and residual components using various statistical techniques.
    """
    
    def decompose_stl(
        self,
        series: Series,
        period: int,
        seasonal: int = 7
    ) -> DecompositionResult:
        """Decompose time series using STL (Seasonal and Trend decomposition using Loess).
        
        Args:
            series: Time series to decompose
            period: Seasonal period (e.g., 7 for weekly, 365 for yearly)
            seasonal: Length of the seasonal smoother (must be odd)
            
        Returns:
            DecompositionResult with trend, seasonal, and residual components
            
        Raises:
            ValueError: If series is too short or parameters are invalid
        """
        if len(series) < 2 * period:
            raise ValueError(
                f"Series length ({len(series)}) must be at least 2 * period ({2 * period})"
            )
        
        # Ensure seasonal is odd
        if seasonal % 2 == 0:
            seasonal += 1
        
        # Create a copy and handle missing values
        series_clean = series.copy()
        if series_clean.isna().any():
            # Interpolate missing values for STL
            series_clean = series_clean.interpolate(method='linear')
        
        # Perform STL decomposition
        stl = STL(series_clean, period=period, seasonal=seasonal)
        result = stl.fit()
        
        return DecompositionResult(
            trend=result.trend,
            seasonal=result.seasonal,
            residual=result.resid,
            method='STL',
            parameters={
                'period': period,
                'seasonal': seasonal
            }
        )
    
    def decompose_x13(self, series: Series) -> DecompositionResult:
        """Decompose time series using X-13-ARIMA-SEATS.
        
        Note: This is a simplified implementation. Full X-13-ARIMA-SEATS requires
        the X-13 binary from the US Census Bureau. This implementation uses
        statsmodels' seasonal_decompose as a fallback.
        
        Args:
            series: Time series to decompose
            
        Returns:
            DecompositionResult with trend, seasonal, and residual components
            
        Raises:
            ValueError: If series is too short
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        if len(series) < 2:
            raise ValueError("Series must have at least 2 observations")
        
        # Create a copy and handle missing values
        series_clean = series.copy()
        if series_clean.isna().any():
            series_clean = series_clean.interpolate(method='linear')
        
        # Infer period from data if possible (assume daily data with weekly seasonality)
        period = 7
        if len(series_clean) >= 14:
            # Use seasonal_decompose as X-13 alternative
            result = seasonal_decompose(
                series_clean,
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )
            
            return DecompositionResult(
                trend=result.trend,
                seasonal=result.seasonal,
                residual=result.resid,
                method='X-13-ARIMA-SEATS (seasonal_decompose fallback)',
                parameters={'period': period, 'model': 'additive'}
            )
        else:
            # For very short series, return simple trend
            trend = series_clean.rolling(window=min(3, len(series_clean)), center=True).mean()
            trend = trend.fillna(series_clean.mean())
            seasonal = pd.Series(0, index=series_clean.index)
            residual = series_clean - trend
            
            return DecompositionResult(
                trend=trend,
                seasonal=seasonal,
                residual=residual,
                method='X-13-ARIMA-SEATS (simple trend fallback)',
                parameters={'period': None, 'model': 'simple'}
            )
    
    def extract_trend(
        self,
        series: Series,
        method: str = 'hp_filter'
    ) -> Series:
        """Extract trend component from time series.
        
        Args:
            series: Time series to extract trend from
            method: Method to use ('hp_filter' for Hodrick-Prescott filter,
                   'moving_average' for simple moving average)
            
        Returns:
            Series containing the trend component
            
        Raises:
            ValueError: If method is not recognized or series is too short
        """
        if len(series) < 2:
            raise ValueError("Series must have at least 2 observations")
        
        series_clean = series.copy()
        if series_clean.isna().any():
            series_clean = series_clean.interpolate(method='linear')
        
        if method == 'hp_filter':
            # Hodrick-Prescott filter
            # Lambda = 1600 is standard for quarterly data, 
            # 129600 for monthly, 6.25 for annual
            # For daily data, use a higher value
            lamb = 129600 if len(series_clean) > 100 else 1600
            trend, cycle = hpfilter(series_clean, lamb=lamb)
            return trend
        
        elif method == 'moving_average':
            # Simple moving average
            window = min(30, len(series_clean) // 3)
            if window < 1:
                window = 1
            trend = series_clean.rolling(window=window, center=True).mean()
            # Fill NaN values at edges
            trend = trend.fillna(method='bfill').fillna(method='ffill')
            return trend
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'hp_filter' or 'moving_average'")
    
    def extract_seasonality(
        self,
        series: Series,
        period: int
    ) -> Series:
        """Extract seasonal component from time series.
        
        Args:
            series: Time series to extract seasonality from
            period: Seasonal period (e.g., 7 for weekly, 365 for yearly)
            
        Returns:
            Series containing the seasonal component
            
        Raises:
            ValueError: If series is too short for the specified period
        """
        if len(series) < 2 * period:
            raise ValueError(
                f"Series length ({len(series)}) must be at least 2 * period ({2 * period})"
            )
        
        # Use STL to extract seasonality
        decomp = self.decompose_stl(series, period=period)
        return decomp.seasonal
