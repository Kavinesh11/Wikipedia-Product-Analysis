"""Time Series Forecaster

Implements demand forecasting using Prophet library.
Generates predictions with confidence intervals, detects seasonality,
and identifies hype events.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import numpy as np
from prophet import Prophet
from src.storage.dto import ForecastResult, SeasonalityPattern, SpikeEvent

logger = logging.getLogger(__name__)


class TimeSeriesForecaster:
    """Time series forecasting for Wikipedia pageview data
    
    Uses Facebook Prophet for demand prediction with automatic
    seasonality detection and trend analysis.
    """
    
    def __init__(self, model_type: str = "prophet"):
        """Initialize forecaster
        
        Args:
            model_type: Type of forecasting model (currently only "prophet" supported)
        """
        if model_type != "prophet":
            raise ValueError(f"Unsupported model_type: {model_type}. Only 'prophet' is supported.")
        
        self.model_type = model_type
        self._models = {}  # Cache trained models by article
        logger.info(f"Initialized TimeSeriesForecaster with model_type={model_type}")
    
    def train(self, historical_data: pd.DataFrame, article: str) -> Prophet:
        """Train forecasting model on historical pageviews
        
        Args:
            historical_data: DataFrame with 'date' and 'views' columns
            article: Article name for model caching
            
        Returns:
            Trained Prophet model
            
        Raises:
            ValueError: If insufficient data (< 90 days) or invalid format
        """
        # Validate input data
        if not isinstance(historical_data, pd.DataFrame):
            raise ValueError("historical_data must be a pandas DataFrame")
        
        if 'date' not in historical_data.columns or 'views' not in historical_data.columns:
            raise ValueError("historical_data must have 'date' and 'views' columns")
        
        # Check minimum data requirement (90 days)
        if len(historical_data) < 90:
            raise ValueError(
                f"Insufficient training data: {len(historical_data)} days. "
                f"Minimum 90 days required for reliable forecasting."
            )
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df = historical_data.copy()
        df = df.rename(columns={'date': 'ds', 'views': 'y'})
        
        # Ensure ds is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        # Ensure y is numeric
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Remove any NaN values
        df = df.dropna()
        
        if len(df) < 90:
            raise ValueError(
                f"After cleaning, only {len(df)} valid records remain. "
                f"Minimum 90 days required."
            )
        
        # Initialize and train Prophet model
        logger.info(f"Training Prophet model for article '{article}' with {len(df)} data points")
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,  # Flexibility of trend changes
            seasonality_prior_scale=10.0,  # Flexibility of seasonality
        )
        
        model.fit(df)
        
        # Cache the trained model
        self._models[article] = model
        
        logger.info(f"Successfully trained model for article '{article}'")
        return model
    
    def predict(
        self, 
        model: Prophet, 
        periods: int = 30,
        article: str = "unknown"
    ) -> ForecastResult:
        """Generate predictions with confidence intervals
        
        Args:
            model: Trained Prophet model
            periods: Number of days to forecast
            article: Article name for result
            
        Returns:
            ForecastResult with predictions and metadata
        """
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Generate predictions
        forecast = model.predict(future)
        
        # Extract predictions (only future periods)
        predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
        predictions = predictions.rename(columns={'ds': 'date'})
        
        # Detect seasonality from the model
        seasonality = self.detect_seasonality(model)
        
        # Calculate growth rate from historical data
        historical_views = model.history['y'].values
        growth_rate = self.calculate_growth_rate_from_series(historical_views)
        
        # Calculate confidence as inverse of prediction interval width
        avg_interval_width = (predictions['yhat_upper'] - predictions['yhat_lower']).mean()
        avg_prediction = predictions['yhat'].mean()
        confidence = max(0.0, min(1.0, 1.0 - (avg_interval_width / (2 * avg_prediction))))
        
        logger.info(
            f"Generated {periods}-day forecast for article '{article}' "
            f"with confidence={confidence:.2f}"
        )
        
        return ForecastResult(
            article=article,
            predictions=predictions,
            seasonality=seasonality,
            growth_rate=growth_rate,
            confidence=confidence
        )
    
    def detect_seasonality(self, model: Prophet) -> SeasonalityPattern:
        """Identify seasonal patterns in traffic
        
        Args:
            model: Trained Prophet model
            
        Returns:
            SeasonalityPattern with detected patterns
        """
        # Get seasonality components from the model
        seasonalities = model.seasonalities
        
        # Determine strongest seasonality
        if 'yearly' in seasonalities:
            # Check yearly seasonality strength
            yearly_strength = self._calculate_seasonality_strength(model, 'yearly')
            if yearly_strength > 0.1:
                return SeasonalityPattern(
                    period='yearly',
                    strength=yearly_strength,
                    peak_day=None  # Could extract peak month if needed
                )
        
        if 'weekly' in seasonalities:
            # Check weekly seasonality strength
            weekly_strength = self._calculate_seasonality_strength(model, 'weekly')
            if weekly_strength > 0.1:
                # Find peak day of week
                peak_day = self._find_peak_day_of_week(model)
                return SeasonalityPattern(
                    period='weekly',
                    strength=weekly_strength,
                    peak_day=peak_day
                )
        
        # No significant seasonality detected
        return SeasonalityPattern(
            period='none',
            strength=0.0,
            peak_day=None
        )
    
    def _calculate_seasonality_strength(self, model: Prophet, period: str) -> float:
        """Calculate strength of seasonal component
        
        Args:
            model: Trained Prophet model
            period: 'yearly' or 'weekly'
            
        Returns:
            Strength value between 0 and 1
        """
        try:
            # Get the seasonal component from model
            if not hasattr(model, 'history'):
                return 0.0
            
            # Make predictions on historical data
            forecast = model.predict(model.history)
            
            # Get seasonal component
            if period == 'yearly' and 'yearly' in forecast.columns:
                seasonal = forecast['yearly'].values
            elif period == 'weekly' and 'weekly' in forecast.columns:
                seasonal = forecast['weekly'].values
            else:
                return 0.0
            
            # Calculate strength as ratio of seasonal variance to total variance
            y = model.history['y'].values
            seasonal_var = np.var(seasonal)
            total_var = np.var(y)
            
            if total_var == 0:
                return 0.0
            
            strength = min(1.0, seasonal_var / total_var)
            return float(strength)
            
        except Exception as e:
            logger.warning(f"Error calculating seasonality strength: {e}")
            return 0.0
    
    def _find_peak_day_of_week(self, model: Prophet) -> int:
        """Find day of week with highest average traffic
        
        Args:
            model: Trained Prophet model
            
        Returns:
            Day of week (0=Monday, 6=Sunday)
        """
        try:
            if not hasattr(model, 'history'):
                return 0
            
            df = model.history.copy()
            df['day_of_week'] = df['ds'].dt.dayofweek
            
            # Calculate average views by day of week
            avg_by_day = df.groupby('day_of_week')['y'].mean()
            
            # Return day with highest average
            return int(avg_by_day.idxmax())
            
        except Exception as e:
            logger.warning(f"Error finding peak day of week: {e}")
            return 0
    
    def calculate_growth_rate(
        self, 
        data: pd.DataFrame, 
        period_days: int = 30
    ) -> float:
        """Calculate percentage growth over period
        
        Args:
            data: DataFrame with 'date' and 'views' columns
            period_days: Number of days to calculate growth over
            
        Returns:
            Growth rate as percentage
        """
        if len(data) < 2:
            return 0.0
        
        # Ensure data is sorted by date
        df = data.copy()
        if 'date' in df.columns:
            df = df.sort_values('date')
        elif 'ds' in df.columns:
            df = df.sort_values('ds')
        else:
            raise ValueError("Data must have 'date' or 'ds' column")
        
        # Get views column
        if 'views' in df.columns:
            views = df['views'].values
        elif 'y' in df.columns:
            views = df['y'].values
        else:
            raise ValueError("Data must have 'views' or 'y' column")
        
        # Calculate growth rate over the period
        if len(views) <= period_days:
            # Use all available data
            views_start = views[0]
            views_end = views[-1]
        else:
            # Use last period_days
            views_start = views[-period_days]
            views_end = views[-1]
        
        if views_start == 0:
            # Avoid division by zero
            return 0.0 if views_end == 0 else 100.0
        
        growth_rate = ((views_end - views_start) / views_start) * 100
        return float(growth_rate)
    
    def calculate_growth_rate_from_series(self, views: np.ndarray, period_days: int = 30) -> float:
        """Calculate growth rate from numpy array
        
        Args:
            views: Array of view counts
            period_days: Number of days to calculate growth over
            
        Returns:
            Growth rate as percentage
        """
        if len(views) < 2:
            return 0.0
        
        # Calculate growth rate over the period
        if len(views) <= period_days:
            views_start = views[0]
            views_end = views[-1]
        else:
            views_start = views[-period_days]
            views_end = views[-1]
        
        if views_start == 0:
            return 0.0 if views_end == 0 else 100.0
        
        growth_rate = ((views_end - views_start) / views_start) * 100
        return float(growth_rate)
    
    def detect_hype_events(
        self, 
        data: pd.DataFrame,
        article: str = "unknown"
    ) -> List[SpikeEvent]:
        """Flag hype events where growth exceeds 2 standard deviations
        
        Args:
            data: DataFrame with 'date' and 'views' columns
            article: Article name for logging
            
        Returns:
            List of detected spike events
        """
        if len(data) < 7:
            # Need at least a week of data for meaningful statistics
            return []
        
        df = data.copy()
        
        # Get views column
        if 'views' in df.columns:
            views = df['views'].values
        elif 'y' in df.columns:
            views = df['y'].values
        else:
            raise ValueError("Data must have 'views' or 'y' column")
        
        # Calculate rolling statistics
        mean = np.mean(views)
        std = np.std(views)
        
        if std == 0:
            # No variation, no spikes
            return []
        
        # Detect spikes (> 2 std dev above mean)
        threshold = mean + 2 * std
        spike_indices = np.where(views > threshold)[0]
        
        if len(spike_indices) == 0:
            return []
        
        # Group consecutive spikes into events
        spike_events = []
        current_spike_start = spike_indices[0]
        current_spike_end = spike_indices[0]
        
        for i in range(1, len(spike_indices)):
            if spike_indices[i] == current_spike_end + 1:
                # Consecutive spike
                current_spike_end = spike_indices[i]
            else:
                # Gap found, save current spike event
                spike_events.append(self._create_spike_event(
                    df, current_spike_start, current_spike_end, views, mean, std
                ))
                current_spike_start = spike_indices[i]
                current_spike_end = spike_indices[i]
        
        # Add the last spike event
        spike_events.append(self._create_spike_event(
            df, current_spike_start, current_spike_end, views, mean, std
        ))
        
        logger.info(f"Detected {len(spike_events)} hype events for article '{article}'")
        return spike_events
    
    def _create_spike_event(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        views: np.ndarray,
        mean: float,
        std: float
    ) -> SpikeEvent:
        """Create a SpikeEvent from spike indices
        
        Args:
            df: DataFrame with date information
            start_idx: Start index of spike
            end_idx: End index of spike
            views: Array of view counts
            mean: Mean views
            std: Standard deviation of views
            
        Returns:
            SpikeEvent object
        """
        duration_days = end_idx - start_idx + 1
        
        # Get timestamp
        if 'date' in df.columns:
            timestamp = df.iloc[start_idx]['date']
        elif 'ds' in df.columns:
            timestamp = df.iloc[start_idx]['ds']
        else:
            timestamp = datetime.now()
        
        # Ensure timestamp is datetime
        if not isinstance(timestamp, datetime):
            timestamp = pd.to_datetime(timestamp).to_pydatetime()
        
        # Calculate magnitude (average std devs above mean during spike)
        spike_views = views[start_idx:end_idx+1]
        avg_spike_views = np.mean(spike_views)
        magnitude = (avg_spike_views - mean) / std if std > 0 else 0.0
        
        # Classify spike type (sustained if > 7 days)
        spike_type = "sustained" if duration_days > 7 else "temporary"
        
        return SpikeEvent(
            timestamp=timestamp,
            magnitude=float(magnitude),
            duration_days=duration_days,
            spike_type=spike_type
        )
    
    def get_cached_model(self, article: str) -> Optional[Prophet]:
        """Get cached model for an article
        
        Args:
            article: Article name
            
        Returns:
            Cached Prophet model or None
        """
        return self._models.get(article)
    
    def clear_cache(self, article: Optional[str] = None):
        """Clear cached models
        
        Args:
            article: Specific article to clear, or None to clear all
        """
        if article:
            if article in self._models:
                del self._models[article]
                logger.info(f"Cleared cached model for article '{article}'")
        else:
            self._models.clear()
            logger.info("Cleared all cached models")
