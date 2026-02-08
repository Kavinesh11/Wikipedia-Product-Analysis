"""Forecasting module.

This module provides time series forecasting functionality using various methods
including ARIMA, Prophet, and Exponential Smoothing.
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import pandas as pd
from pandas import Series
import numpy as np
from pmdarima import auto_arima
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit

from wikipedia_health.models.data_models import ForecastResult


@dataclass
class ARIMAModel:
    """ARIMA model wrapper."""
    model: Any
    order: Tuple[int, int, int]
    aic: float
    bic: float


@dataclass
class ProphetModel:
    """Prophet model wrapper."""
    model: Prophet
    params: Dict[str, Any]


@dataclass
class ExponentialSmoothingModel:
    """Exponential Smoothing model wrapper."""
    model: Any
    seasonal_periods: int


@dataclass
class CrossValidationResult:
    """Cross-validation result."""
    mean_error: float
    std_error: float
    errors: list
    model_type: str


# Type alias for forecast models
ForecastModel = Any


class Forecaster:
    """Time series forecasting using multiple methods.
    
    This class provides methods for fitting various forecasting models and
    generating predictions with uncertainty quantification.
    """
    
    def fit_arima(
        self,
        series: Series,
        order: Optional[Tuple[int, int, int]] = None
    ) -> ARIMAModel:
        """Fit ARIMA model using auto_arima for automatic order selection.
        
        Args:
            series: Time series to fit
            order: ARIMA order (p, d, q). If None, auto-selected using AIC
            
        Returns:
            ARIMAModel object with fitted model
            
        Raises:
            ValueError: If series is too short or contains invalid values
        """
        if len(series) < 10:
            raise ValueError("Series must have at least 10 observations for ARIMA")
        
        # Clean the series
        series_clean = series.copy()
        if series_clean.isna().any():
            series_clean = series_clean.interpolate(method='linear')
        
        if order is None:
            # Auto-select order using pmdarima
            model = auto_arima(
                series_clean,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False
            )
            order = model.order
        else:
            # Fit with specified order
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(series_clean, order=order).fit()
        
        return ARIMAModel(
            model=model,
            order=order,
            aic=model.aic(),
            bic=model.bic()
        )
    
    def fit_prophet(
        self,
        series: Series,
        seasonality_mode: str = 'multiplicative',
        changepoint_prior_scale: float = 0.05
    ) -> ProphetModel:
        """Fit Prophet model for time series forecasting.
        
        Args:
            series: Time series to fit (must have datetime index)
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend (higher = more flexible)
            
        Returns:
            ProphetModel object with fitted model
            
        Raises:
            ValueError: If series is too short or doesn't have datetime index
        """
        if len(series) < 10:
            raise ValueError("Series must have at least 10 observations for Prophet")
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        if isinstance(series.index, pd.DatetimeIndex):
            df = pd.DataFrame({
                'ds': series.index,
                'y': series.values
            })
        else:
            # Create a datetime index if not present
            df = pd.DataFrame({
                'ds': pd.date_range(start='2020-01-01', periods=len(series), freq='D'),
                'y': series.values
            })
        
        # Handle missing values
        df = df.dropna()
        
        # Initialize and fit Prophet model
        model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True if len(df) > 365 else False
        )
        
        # Suppress Prophet's verbose output
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        
        model.fit(df)
        
        return ProphetModel(
            model=model,
            params={
                'seasonality_mode': seasonality_mode,
                'changepoint_prior_scale': changepoint_prior_scale
            }
        )
    
    def fit_exponential_smoothing(
        self,
        series: Series,
        seasonal_periods: int = 7
    ) -> ExponentialSmoothingModel:
        """Fit Exponential Smoothing model (Holt-Winters).
        
        Args:
            series: Time series to fit
            seasonal_periods: Number of periods in seasonal cycle
            
        Returns:
            ExponentialSmoothingModel object with fitted model
            
        Raises:
            ValueError: If series is too short
        """
        if len(series) < 2 * seasonal_periods:
            raise ValueError(
                f"Series length ({len(series)}) must be at least "
                f"2 * seasonal_periods ({2 * seasonal_periods})"
            )
        
        # Clean the series
        series_clean = series.copy()
        if series_clean.isna().any():
            series_clean = series_clean.interpolate(method='linear')
        
        # Fit Exponential Smoothing model
        try:
            model = ExponentialSmoothing(
                series_clean,
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add'
            ).fit()
        except:
            # Fallback to simpler model if seasonal model fails
            model = ExponentialSmoothing(
                series_clean,
                trend='add',
                seasonal=None
            ).fit()
            seasonal_periods = None
        
        return ExponentialSmoothingModel(
            model=model,
            seasonal_periods=seasonal_periods
        )
    
    def forecast(
        self,
        model: ForecastModel,
        horizon: int,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """Generate forecast with prediction intervals.
        
        Args:
            model: Fitted forecast model (ARIMAModel, ProphetModel, or ExponentialSmoothingModel)
            horizon: Number of periods to forecast
            confidence_level: Confidence level for prediction intervals (e.g., 0.95)
            
        Returns:
            ForecastResult with point forecast and prediction intervals
            
        Raises:
            ValueError: If model type is not recognized
        """
        if isinstance(model, ARIMAModel):
            return self._forecast_arima(model, horizon, confidence_level)
        elif isinstance(model, ProphetModel):
            return self._forecast_prophet(model, horizon, confidence_level)
        elif isinstance(model, ExponentialSmoothingModel):
            return self._forecast_exponential_smoothing(model, horizon, confidence_level)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
    
    def _forecast_arima(
        self,
        model: ARIMAModel,
        horizon: int,
        confidence_level: float
    ) -> ForecastResult:
        """Generate ARIMA forecast."""
        # Get forecast with confidence intervals
        forecast = model.model.predict(n_periods=horizon, return_conf_int=True, alpha=1-confidence_level)
        
        if isinstance(forecast, tuple):
            point_forecast, conf_int = forecast
        else:
            point_forecast = forecast
            # If no confidence intervals, create dummy ones
            conf_int = np.column_stack([
                point_forecast * 0.9,
                point_forecast * 1.1
            ])
        
        return ForecastResult(
            point_forecast=pd.Series(point_forecast),
            lower_bound=pd.Series(conf_int[:, 0]),
            upper_bound=pd.Series(conf_int[:, 1]),
            confidence_level=confidence_level,
            model_type='ARIMA',
            horizon=horizon
        )
    
    def _forecast_prophet(
        self,
        model: ProphetModel,
        horizon: int,
        confidence_level: float
    ) -> ForecastResult:
        """Generate Prophet forecast."""
        # Create future dataframe
        future = model.model.make_future_dataframe(periods=horizon, freq='D')
        
        # Generate forecast
        forecast = model.model.predict(future)
        
        # Extract forecast for future periods only
        forecast_future = forecast.tail(horizon)
        
        # Prophet uses 'yhat', 'yhat_lower', 'yhat_upper'
        return ForecastResult(
            point_forecast=pd.Series(forecast_future['yhat'].values),
            lower_bound=pd.Series(forecast_future['yhat_lower'].values),
            upper_bound=pd.Series(forecast_future['yhat_upper'].values),
            confidence_level=confidence_level,
            model_type='Prophet',
            horizon=horizon
        )
    
    def _forecast_exponential_smoothing(
        self,
        model: ExponentialSmoothingModel,
        horizon: int,
        confidence_level: float
    ) -> ForecastResult:
        """Generate Exponential Smoothing forecast."""
        # Get forecast
        forecast = model.model.forecast(steps=horizon)
        
        # Exponential Smoothing doesn't provide built-in prediction intervals
        # Estimate them using residual standard error
        residuals = model.model.resid
        std_error = np.std(residuals)
        
        # Calculate prediction intervals (approximate)
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Prediction intervals widen with horizon
        prediction_std = std_error * np.sqrt(np.arange(1, horizon + 1))
        
        lower_bound = forecast - z_score * prediction_std
        upper_bound = forecast + z_score * prediction_std
        
        return ForecastResult(
            point_forecast=pd.Series(forecast),
            lower_bound=pd.Series(lower_bound),
            upper_bound=pd.Series(upper_bound),
            confidence_level=confidence_level,
            model_type='ExponentialSmoothing',
            horizon=horizon
        )
    
    def cross_validate(
        self,
        series: Series,
        model_type: str,
        n_splits: int = 5
    ) -> CrossValidationResult:
        """Perform time series cross-validation.
        
        Args:
            series: Time series to validate
            model_type: Type of model ('arima', 'prophet', 'exponential_smoothing')
            n_splits: Number of cross-validation splits
            
        Returns:
            CrossValidationResult with error metrics
            
        Raises:
            ValueError: If series is too short or model_type is invalid
        """
        if len(series) < n_splits * 10:
            raise ValueError(
                f"Series length ({len(series)}) must be at least "
                f"n_splits * 10 ({n_splits * 10})"
            )
        
        # Clean the series
        series_clean = series.copy()
        if series_clean.isna().any():
            series_clean = series_clean.interpolate(method='linear')
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        errors = []
        
        for train_idx, test_idx in tscv.split(series_clean):
            train = series_clean.iloc[train_idx]
            test = series_clean.iloc[test_idx]
            
            try:
                # Fit model
                if model_type.lower() == 'arima':
                    model = self.fit_arima(train)
                elif model_type.lower() == 'prophet':
                    model = self.fit_prophet(train)
                elif model_type.lower() == 'exponential_smoothing':
                    model = self.fit_exponential_smoothing(train)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Forecast
                forecast_result = self.forecast(model, horizon=len(test))
                
                # Calculate error (RMSE)
                error = np.sqrt(np.mean((test.values - forecast_result.point_forecast.values) ** 2))
                errors.append(error)
                
            except Exception as e:
                # If a fold fails, record a large error
                errors.append(np.inf)
        
        return CrossValidationResult(
            mean_error=np.mean(errors),
            std_error=np.std(errors),
            errors=errors,
            model_type=model_type
        )
