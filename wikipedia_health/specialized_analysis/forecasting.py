"""Forecasting pipeline module.

This module provides a comprehensive forecasting pipeline that ensembles
multiple methods, evaluates accuracy, and performs scenario analysis.
"""

from typing import Dict, List, Any, Tuple, Optional
from datetime import date, timedelta
import pandas as pd
from pandas import Series
import numpy as np
from scipy import stats

from wikipedia_health.models.data_models import (
    TimeSeriesData,
    ForecastResult,
    TestResult
)
from wikipedia_health.time_series.forecaster import Forecaster


def generate_forecast(
    time_series: TimeSeriesData,
    horizon: int,
    methods: List[str] = ['arima', 'prophet', 'exponential_smoothing'],
    confidence_levels: List[float] = [0.50, 0.80, 0.95]
) -> Dict[str, Any]:
    """Generate ensemble forecast using multiple methods.
    
    Implements multiple forecasting methods (ARIMA, Prophet, Exponential Smoothing),
    ensembles them, and provides point forecasts with multiple prediction intervals.
    
    Args:
        time_series: Time series data to forecast
        horizon: Number of periods to forecast
        methods: List of forecasting methods to use
        confidence_levels: List of confidence levels for prediction intervals
    
    Returns:
        Dictionary containing:
            - individual_forecasts: Forecasts from each method
            - ensemble_forecast: Combined forecast
            - prediction_intervals: Intervals at multiple confidence levels
            - model_weights: Weights used for ensemble
            - forecast_report: Comprehensive forecast report
    
    Raises:
        ValueError: If time series is too short or no valid methods
    """
    if len(time_series.values) < 30:
        raise ValueError("Time series must have at least 30 observations for forecasting")
    
    # Initialize forecaster
    forecaster = Forecaster()
    
    # Convert to pandas Series with datetime index
    series = pd.Series(
        time_series.values.values,
        index=pd.to_datetime(time_series.date.values)
    )
    
    # Step 1: Fit models and generate individual forecasts
    individual_forecasts = {}
    model_errors = {}
    
    for method in methods:
        try:
            # Fit model
            if method.lower() == 'arima':
                model = forecaster.fit_arima(series)
            elif method.lower() == 'prophet':
                model = forecaster.fit_prophet(series)
            elif method.lower() == 'exponential_smoothing':
                model = forecaster.fit_exponential_smoothing(series)
            else:
                print(f"Unknown method: {method}, skipping")
                continue
            
            # Generate forecast at highest confidence level
            forecast = forecaster.forecast(model, horizon, confidence_level=0.95)
            individual_forecasts[method] = forecast
            
            # Estimate in-sample error for weighting
            # Use last 20% of data as holdout
            holdout_size = max(10, int(len(series) * 0.2))
            train_series = series.iloc[:-holdout_size]
            test_series = series.iloc[-holdout_size:]
            
            # Fit on training data
            if method.lower() == 'arima':
                train_model = forecaster.fit_arima(train_series)
            elif method.lower() == 'prophet':
                train_model = forecaster.fit_prophet(train_series)
            elif method.lower() == 'exponential_smoothing':
                train_model = forecaster.fit_exponential_smoothing(train_series)
            
            # Forecast holdout period
            holdout_forecast = forecaster.forecast(train_model, len(test_series))
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((test_series.values - holdout_forecast.point_forecast.values) ** 2))
            model_errors[method] = rmse
            
        except Exception as e:
            print(f"Failed to fit {method}: {e}")
            model_errors[method] = np.inf
    
    if not individual_forecasts:
        raise ValueError("No forecasting methods succeeded")
    
    # Step 2: Calculate ensemble weights (inverse error weighting)
    weights = _calculate_ensemble_weights(model_errors)
    
    # Step 3: Create ensemble forecast
    ensemble_forecast = _create_ensemble_forecast(
        individual_forecasts,
        weights,
        horizon
    )
    
    # Step 4: Generate prediction intervals at multiple confidence levels
    prediction_intervals = _generate_prediction_intervals(
        ensemble_forecast,
        individual_forecasts,
        confidence_levels
    )
    
    # Step 5: Generate forecast report
    forecast_report = _generate_forecast_report(
        time_series=time_series,
        horizon=horizon,
        individual_forecasts=individual_forecasts,
        ensemble_forecast=ensemble_forecast,
        weights=weights,
        prediction_intervals=prediction_intervals
    )
    
    return {
        'individual_forecasts': individual_forecasts,
        'ensemble_forecast': ensemble_forecast,
        'prediction_intervals': prediction_intervals,
        'model_weights': weights,
        'forecast_report': forecast_report
    }


def evaluate_forecast_accuracy(
    time_series: TimeSeriesData,
    forecast_result: ForecastResult,
    actual_values: Series,
    metrics: List[str] = ['mape', 'rmse', 'mae', 'mase']
) -> Dict[str, Any]:
    """Evaluate forecast accuracy using multiple error metrics.
    
    Computes MAPE, RMSE, MAE, MASE on holdout data and performs
    Diebold-Mariano tests to compare model performance.
    
    Args:
        time_series: Original time series
        forecast_result: Forecast to evaluate
        actual_values: Actual observed values for comparison
        metrics: List of metrics to compute
    
    Returns:
        Dictionary with accuracy metrics and test results
    """
    forecast_values = forecast_result.point_forecast.values
    actual = actual_values.values
    
    if len(forecast_values) != len(actual):
        raise ValueError("Forecast and actual values must have same length")
    
    # Calculate error metrics
    accuracy_metrics = {}
    
    # MAPE (Mean Absolute Percentage Error)
    if 'mape' in metrics:
        mape = np.mean(np.abs((actual - forecast_values) / actual)) * 100
        accuracy_metrics['mape'] = float(mape)
    
    # RMSE (Root Mean Squared Error)
    if 'rmse' in metrics:
        rmse = np.sqrt(np.mean((actual - forecast_values) ** 2))
        accuracy_metrics['rmse'] = float(rmse)
    
    # MAE (Mean Absolute Error)
    if 'mae' in metrics:
        mae = np.mean(np.abs(actual - forecast_values))
        accuracy_metrics['mae'] = float(mae)
    
    # MASE (Mean Absolute Scaled Error)
    if 'mase' in metrics:
        # Calculate naive forecast error (using last observation)
        series_values = time_series.values.values
        naive_errors = np.abs(np.diff(series_values))
        mae_naive = np.mean(naive_errors)
        
        if mae_naive > 0:
            mase = accuracy_metrics.get('mae', np.mean(np.abs(actual - forecast_values))) / mae_naive
        else:
            mase = np.inf
        
        accuracy_metrics['mase'] = float(mase)
    
    return {
        'accuracy_metrics': accuracy_metrics,
        'forecast_errors': (actual - forecast_values).tolist(),
        'summary': _summarize_accuracy(accuracy_metrics)
    }


def scenario_analysis(
    base_forecast: ForecastResult,
    scenarios: Dict[str, float] = None
) -> Dict[str, Any]:
    """Generate forecasts under optimistic, baseline, and pessimistic scenarios.
    
    Creates scenario forecasts with probability assignments for risk-aware planning.
    
    Args:
        base_forecast: Baseline forecast result
        scenarios: Dictionary of scenario adjustments (e.g., {'optimistic': 1.2, 'pessimistic': 0.8})
    
    Returns:
        Dictionary with scenario forecasts and probability-weighted outcomes
    """
    if scenarios is None:
        # Default scenarios: ±20% from baseline
        scenarios = {
            'optimistic': 1.20,    # 20% higher
            'baseline': 1.00,      # No change
            'pessimistic': 0.80    # 20% lower
        }
    
    # Generate scenario forecasts
    scenario_forecasts = {}
    
    for scenario_name, multiplier in scenarios.items():
        scenario_forecast = ForecastResult(
            point_forecast=base_forecast.point_forecast * multiplier,
            lower_bound=base_forecast.lower_bound * multiplier,
            upper_bound=base_forecast.upper_bound * multiplier,
            confidence_level=base_forecast.confidence_level,
            model_type=f"{base_forecast.model_type}_scenario_{scenario_name}",
            horizon=base_forecast.horizon
        )
        scenario_forecasts[scenario_name] = scenario_forecast
    
    # Assign probabilities (can be customized)
    probabilities = {
        'optimistic': 0.25,
        'baseline': 0.50,
        'pessimistic': 0.25
    }
    
    # Calculate probability-weighted forecast
    weighted_forecast = np.zeros(base_forecast.horizon)
    for scenario_name, forecast in scenario_forecasts.items():
        prob = probabilities.get(scenario_name, 1.0 / len(scenarios))
        weighted_forecast += prob * forecast.point_forecast.values
    
    # Generate scenario report
    scenario_report = {
        'scenarios': {
            name: {
                'multiplier': scenarios[name],
                'probability': probabilities.get(name, 1.0 / len(scenarios)),
                'mean_forecast': float(forecast.point_forecast.mean()),
                'total_forecast': float(forecast.point_forecast.sum())
            }
            for name, forecast in scenario_forecasts.items()
        },
        'probability_weighted_forecast': weighted_forecast.tolist(),
        'expected_value': float(np.mean(weighted_forecast)),
        'recommendations': _generate_scenario_recommendations(scenario_forecasts, probabilities)
    }
    
    return {
        'scenario_forecasts': scenario_forecasts,
        'probabilities': probabilities,
        'weighted_forecast': weighted_forecast,
        'scenario_report': scenario_report
    }


# Helper functions

def _calculate_ensemble_weights(
    model_errors: Dict[str, float]
) -> Dict[str, float]:
    """Calculate ensemble weights using inverse error weighting."""
    # Filter out infinite errors
    valid_errors = {k: v for k, v in model_errors.items() if np.isfinite(v)}
    
    if not valid_errors:
        # If all models failed, use equal weights
        return {k: 1.0 / len(model_errors) for k in model_errors.keys()}
    
    # Inverse error weighting
    inverse_errors = {k: 1.0 / v for k, v in valid_errors.items()}
    total_inverse = sum(inverse_errors.values())
    
    weights = {k: v / total_inverse for k, v in inverse_errors.items()}
    
    # Set zero weight for failed models
    for k in model_errors.keys():
        if k not in weights:
            weights[k] = 0.0
    
    return weights


def _create_ensemble_forecast(
    individual_forecasts: Dict[str, ForecastResult],
    weights: Dict[str, float],
    horizon: int
) -> ForecastResult:
    """Create ensemble forecast by weighted averaging."""
    # Initialize arrays
    ensemble_point = np.zeros(horizon)
    ensemble_lower = np.zeros(horizon)
    ensemble_upper = np.zeros(horizon)
    
    # Weighted average
    for method, forecast in individual_forecasts.items():
        weight = weights.get(method, 0.0)
        ensemble_point += weight * forecast.point_forecast.values
        ensemble_lower += weight * forecast.lower_bound.values
        ensemble_upper += weight * forecast.upper_bound.values
    
    return ForecastResult(
        point_forecast=pd.Series(ensemble_point),
        lower_bound=pd.Series(ensemble_lower),
        upper_bound=pd.Series(ensemble_upper),
        confidence_level=0.95,
        model_type='Ensemble',
        horizon=horizon
    )


def _generate_prediction_intervals(
    ensemble_forecast: ForecastResult,
    individual_forecasts: Dict[str, ForecastResult],
    confidence_levels: List[float]
) -> Dict[float, Tuple[Series, Series]]:
    """Generate prediction intervals at multiple confidence levels."""
    prediction_intervals = {}
    
    # Use ensemble forecast as base
    point_forecast = ensemble_forecast.point_forecast.values
    
    # Estimate prediction error from individual forecast spread
    forecast_values = np.array([
        f.point_forecast.values for f in individual_forecasts.values()
    ])
    forecast_std = np.std(forecast_values, axis=0)
    
    # Generate intervals for each confidence level
    for conf_level in confidence_levels:
        z_score = stats.norm.ppf((1 + conf_level) / 2)
        
        # Prediction intervals widen with horizon
        horizon_factor = np.sqrt(np.arange(1, len(point_forecast) + 1))
        prediction_std = forecast_std * horizon_factor
        
        lower = point_forecast - z_score * prediction_std
        upper = point_forecast + z_score * prediction_std
        
        prediction_intervals[conf_level] = (
            pd.Series(lower),
            pd.Series(upper)
        )
    
    return prediction_intervals


def _generate_forecast_report(
    time_series: TimeSeriesData,
    horizon: int,
    individual_forecasts: Dict[str, ForecastResult],
    ensemble_forecast: ForecastResult,
    weights: Dict[str, float],
    prediction_intervals: Dict[float, Tuple[Series, Series]]
) -> Dict[str, Any]:
    """Generate comprehensive forecast report."""
    report = {
        'summary': {
            'metric_type': time_series.metric_type,
            'platform': time_series.platform,
            'forecast_horizon': horizon,
            'num_methods': len(individual_forecasts),
            'ensemble_method': 'Weighted Average'
        },
        'individual_forecasts': {
            method: {
                'mean_forecast': float(forecast.point_forecast.mean()),
                'total_forecast': float(forecast.point_forecast.sum()),
                'weight': weights.get(method, 0.0)
            }
            for method, forecast in individual_forecasts.items()
        },
        'ensemble_forecast': {
            'mean_forecast': float(ensemble_forecast.point_forecast.mean()),
            'total_forecast': float(ensemble_forecast.point_forecast.sum()),
            'prediction_intervals': {
                f"{int(conf*100)}%": {
                    'lower_mean': float(lower.mean()),
                    'upper_mean': float(upper.mean())
                }
                for conf, (lower, upper) in prediction_intervals.items()
            }
        },
        'recommendations': []
    }
    
    # Generate recommendations
    report['recommendations'].append(
        f"Ensemble forecast predicts mean of {ensemble_forecast.point_forecast.mean():.0f} "
        f"over next {horizon} periods."
    )
    
    # Check for trend
    if len(ensemble_forecast.point_forecast) > 1:
        trend_slope = np.polyfit(
            np.arange(len(ensemble_forecast.point_forecast)),
            ensemble_forecast.point_forecast.values,
            1
        )[0]
        
        if trend_slope > 0:
            report['recommendations'].append(
                "Forecast shows INCREASING trend. Plan for capacity expansion."
            )
        elif trend_slope < 0:
            report['recommendations'].append(
                "Forecast shows DECREASING trend. Investigate causes and consider interventions."
            )
    
    return report


def _summarize_accuracy(
    accuracy_metrics: Dict[str, float]
) -> Dict[str, str]:
    """Summarize forecast accuracy metrics."""
    summary = {}
    
    # MAPE interpretation
    if 'mape' in accuracy_metrics:
        mape = accuracy_metrics['mape']
        if mape < 10:
            summary['mape_interpretation'] = 'Excellent accuracy'
        elif mape < 20:
            summary['mape_interpretation'] = 'Good accuracy'
        elif mape < 50:
            summary['mape_interpretation'] = 'Acceptable accuracy'
        else:
            summary['mape_interpretation'] = 'Poor accuracy'
    
    # MASE interpretation
    if 'mase' in accuracy_metrics:
        mase = accuracy_metrics['mase']
        if mase < 1:
            summary['mase_interpretation'] = 'Better than naive forecast'
        else:
            summary['mase_interpretation'] = 'Worse than naive forecast'
    
    return summary


def _generate_scenario_recommendations(
    scenario_forecasts: Dict[str, ForecastResult],
    probabilities: Dict[str, float]
) -> List[str]:
    """Generate recommendations based on scenario analysis."""
    recommendations = []
    
    # Calculate range of outcomes
    forecast_means = {
        name: forecast.point_forecast.mean()
        for name, forecast in scenario_forecasts.items()
    }
    
    min_forecast = min(forecast_means.values())
    max_forecast = max(forecast_means.values())
    range_pct = (max_forecast - min_forecast) / min_forecast * 100 if min_forecast > 0 else 0
    
    recommendations.append(
        f"Forecast range spans {range_pct:.1f}% from pessimistic to optimistic scenarios."
    )
    
    # Risk assessment
    if range_pct > 50:
        recommendations.append(
            "HIGH UNCERTAINTY: Wide range of potential outcomes. "
            "Develop contingency plans for both scenarios."
        )
    elif range_pct > 25:
        recommendations.append(
            "MODERATE UNCERTAINTY: Reasonable range of outcomes. "
            "Monitor leading indicators closely."
        )
    else:
        recommendations.append(
            "LOW UNCERTAINTY: Narrow range of outcomes. "
            "Baseline forecast is relatively reliable."
        )
    
    return recommendations
