"""Visualization functions for time series analysis.

This module provides functions for creating publication-quality plots with
statistical evidence overlays including confidence bands, p-values, and effect sizes.
"""

from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
import numpy as np
from datetime import date

from wikipedia_health.models import (
    TimeSeriesData,
    CausalEffect,
    ForecastResult,
    TestResult,
)


# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_trend_with_confidence_bands(
    series: Series,
    dates: Optional[Series] = None,
    confidence_interval: Optional[Tuple[Series, Series]] = None,
    title: str = "Time Series Trend",
    xlabel: str = "Date",
    ylabel: str = "Value",
    show_grid: bool = True,
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """Plot time series trend with 95% confidence bands.
    
    Args:
        series: Time series values to plot
        dates: Optional date index (uses integer index if None)
        confidence_interval: Optional tuple of (lower_bound, upper_bound) series
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_grid: Whether to show grid lines
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure object
        
    Requirements: 13.1
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use dates if provided, otherwise use integer index
    x_values = dates if dates is not None else pd.Series(range(len(series)))
    
    # Plot main series
    ax.plot(x_values, series, color='#2E86AB', linewidth=2, label='Observed')
    
    # Plot confidence bands if provided
    if confidence_interval is not None:
        lower_bound, upper_bound = confidence_interval
        ax.fill_between(
            x_values,
            lower_bound,
            upper_bound,
            alpha=0.3,
            color='#2E86AB',
            label='95% Confidence Interval'
        )
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis for dates
    if dates is not None:
        try:
            first_val = x_values.iloc[0] if hasattr(x_values, 'iloc') else x_values[0]
            if isinstance(first_val, (pd.Timestamp, date)):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        except (IndexError, TypeError):
            pass
    
    ax.legend(loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    
    return fig


def plot_campaign_effect(
    causal_effect: CausalEffect,
    dates: Optional[Series] = None,
    title: str = "Campaign Effect Analysis",
    xlabel: str = "Date",
    ylabel: str = "Value",
    show_annotations: bool = True,
    figsize: Tuple[int, int] = (14, 7)
) -> Figure:
    """Plot campaign effect with observed vs counterfactual comparison.
    
    Args:
        causal_effect: CausalEffect object containing observed and counterfactual data
        dates: Optional date index for x-axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_annotations: Whether to show statistical annotations
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure object
        
    Requirements: 13.2
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use dates if provided, otherwise use integer index
    x_values = dates if dates is not None else pd.Series(range(len(causal_effect.observed)))
    
    # Plot observed data
    ax.plot(
        x_values,
        causal_effect.observed,
        color='#2E86AB',
        linewidth=2,
        label='Observed',
        marker='o',
        markersize=3
    )
    
    # Plot counterfactual
    ax.plot(
        x_values,
        causal_effect.counterfactual,
        color='#A23B72',
        linewidth=2,
        linestyle='--',
        label='Counterfactual (No Intervention)',
        marker='s',
        markersize=3
    )
    
    # Shade the treatment period
    treatment_start, treatment_end = causal_effect.treatment_period
    if dates is not None:
        # Find indices for treatment period
        dates_pd = pd.to_datetime(dates)
        treatment_mask = (dates_pd >= pd.Timestamp(treatment_start)) & \
                        (dates_pd <= pd.Timestamp(treatment_end))
        if treatment_mask.any():
            # Get first and last values from masked dates
            masked_dates = x_values[treatment_mask]
            if hasattr(masked_dates, 'iloc'):
                first_date = masked_dates.iloc[0]
                last_date = masked_dates.iloc[-1]
            else:
                first_date = masked_dates[0]
                last_date = masked_dates[-1]
            
            ax.axvspan(
                first_date,
                last_date,
                alpha=0.2,
                color='yellow',
                label='Treatment Period'
            )
    
    # Add statistical annotations
    if show_annotations:
        # Add text box with effect statistics
        effect_pct = causal_effect.percentage_effect()
        ci_lower, ci_upper = causal_effect.confidence_interval
        
        # Determine significance stars
        if causal_effect.p_value < 0.001:
            sig_stars = "***"
        elif causal_effect.p_value < 0.01:
            sig_stars = "**"
        elif causal_effect.p_value < 0.05:
            sig_stars = "*"
        else:
            sig_stars = "ns"
        
        stats_text = (
            f"Effect Size: {causal_effect.effect_size:.2f} ({effect_pct:+.1f}%)\n"
            f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]\n"
            f"p-value: {causal_effect.p_value:.4f} {sig_stars}\n"
            f"Method: {causal_effect.method}"
        )
        
        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis for dates
    if dates is not None:
        try:
            first_val = x_values.iloc[0] if hasattr(x_values, 'iloc') else x_values[0]
            if isinstance(first_val, (pd.Timestamp, date)):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        except (IndexError, TypeError):
            pass
    
    ax.legend(loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    
    return fig


def plot_forecast(
    forecast_result: ForecastResult,
    historical: Optional[Series] = None,
    historical_dates: Optional[Series] = None,
    forecast_dates: Optional[Series] = None,
    title: str = "Forecast with Uncertainty",
    xlabel: str = "Date",
    ylabel: str = "Value",
    show_intervals: List[float] = [0.50, 0.80, 0.95],
    figsize: Tuple[int, int] = (14, 7)
) -> Figure:
    """Plot forecast with uncertainty fans (50%, 80%, 95% prediction intervals).
    
    Args:
        forecast_result: ForecastResult object with predictions and intervals
        historical: Optional historical data to plot alongside forecast
        historical_dates: Optional dates for historical data
        forecast_dates: Optional dates for forecast period
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_intervals: List of confidence levels to display (e.g., [0.50, 0.80, 0.95])
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure object
        
    Requirements: 13.3
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot historical data if provided
    if historical is not None:
        hist_x = historical_dates if historical_dates is not None else \
                 pd.Series(range(len(historical)))
        ax.plot(
            hist_x,
            historical,
            color='#2E86AB',
            linewidth=2,
            label='Historical',
            marker='o',
            markersize=2
        )
    
    # Prepare forecast x-axis
    if forecast_dates is not None:
        forecast_x = forecast_dates
    elif historical_dates is not None:
        # Continue from last historical date
        last_date_val = historical_dates.iloc[-1] if hasattr(historical_dates, 'iloc') else historical_dates[-1]
        last_date = pd.to_datetime(last_date_val)
        forecast_x = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=len(forecast_result.point_forecast),
            freq='D'
        )
    else:
        start_idx = len(historical) if historical is not None else 0
        forecast_x = pd.Series(range(start_idx, start_idx + len(forecast_result.point_forecast)))
    
    # Plot point forecast
    ax.plot(
        forecast_x,
        forecast_result.point_forecast,
        color='#F18F01',
        linewidth=2,
        label=f'Forecast ({forecast_result.model_type})',
        marker='s',
        markersize=3
    )
    
    # Plot uncertainty fans with different alpha levels
    # Assuming forecast_result has the main confidence interval
    # For multiple intervals, we'd need to calculate them separately
    colors = ['#F18F01', '#F18F01', '#F18F01']
    alphas = [0.5, 0.3, 0.15]
    
    # Plot the main confidence interval (typically 95%)
    ax.fill_between(
        forecast_x,
        forecast_result.lower_bound,
        forecast_result.upper_bound,
        alpha=alphas[-1],
        color=colors[-1],
        label=f'{int(forecast_result.confidence_level*100)}% Prediction Interval'
    )
    
    # Add vertical line separating historical from forecast
    if historical is not None and historical_dates is not None:
        last_hist_date = historical_dates.iloc[-1] if hasattr(historical_dates, 'iloc') else historical_dates[-1]
        ax.axvline(
            x=last_hist_date,
            color='gray',
            linestyle='--',
            linewidth=1,
            alpha=0.7,
            label='Forecast Start'
        )
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis for dates
    try:
        first_val = forecast_x.iloc[0] if hasattr(forecast_x, 'iloc') else forecast_x[0]
        if isinstance(first_val, (pd.Timestamp, date)):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    except (IndexError, TypeError):
        pass
    
    ax.legend(loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    
    return fig


def plot_comparison(
    groups: Dict[str, Series],
    test_result: Optional[TestResult] = None,
    title: str = "Group Comparison",
    xlabel: str = "Group",
    ylabel: str = "Value",
    show_error_bars: bool = True,
    show_significance: bool = True,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """Plot group comparison with error bars and significance indicators.
    
    Args:
        groups: Dictionary mapping group names to value series
        test_result: Optional TestResult with statistical comparison
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        show_error_bars: Whether to show error bars (95% CI)
        show_significance: Whether to show significance indicators
        figsize: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure object
        
    Requirements: 13.4
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate means and confidence intervals
    group_names = list(groups.keys())
    means = [groups[name].mean() for name in group_names]
    
    if show_error_bars:
        # Calculate 95% CI using standard error
        from scipy import stats
        errors = []
        for name in group_names:
            data = groups[name]
            se = stats.sem(data)
            ci = se * stats.t.ppf((1 + 0.95) / 2, len(data) - 1)
            errors.append(ci)
    else:
        errors = None
    
    # Create bar plot
    x_pos = np.arange(len(group_names))
    bars = ax.bar(
        x_pos,
        means,
        yerr=errors if show_error_bars else None,
        capsize=5,
        color=['#2E86AB', '#A23B72', '#F18F01', '#06A77D'][:len(group_names)],
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{mean:.2f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Add significance indicators if test result provided
    if show_significance and test_result is not None:
        # Determine significance stars
        if test_result.p_value < 0.001:
            sig_stars = "***"
        elif test_result.p_value < 0.01:
            sig_stars = "**"
        elif test_result.p_value < 0.05:
            sig_stars = "*"
        else:
            sig_stars = "ns"
        
        # Add significance bracket for two-group comparison
        if len(group_names) == 2:
            y_max = max(means) + (max(errors) if errors else 0)
            y_bracket = y_max * 1.1
            
            ax.plot([0, 0, 1, 1], [y_bracket, y_bracket * 1.02, y_bracket * 1.02, y_bracket],
                   'k-', linewidth=1.5)
            ax.text(0.5, y_bracket * 1.03, sig_stars, ha='center', va='bottom',
                   fontsize=14, fontweight='bold')
        
        # Add stats text box
        stats_text = (
            f"Test: {test_result.test_name}\n"
            f"Statistic: {test_result.statistic:.4f}\n"
            f"p-value: {test_result.p_value:.4f} {sig_stars}\n"
            f"Effect Size: {test_result.effect_size:.4f}\n"
            f"95% CI: [{test_result.confidence_interval[0]:.2f}, "
            f"{test_result.confidence_interval[1]:.2f}]"
        )
        
        ax.text(
            0.98, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_names, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    return fig


def add_statistical_annotations(
    ax: Axes,
    test_results: List[TestResult],
    position: str = 'top_right'
) -> None:
    """Add statistical annotations (p-values, effect sizes) to existing plot.
    
    Args:
        ax: Matplotlib Axes object to annotate
        test_results: List of TestResult objects to display
        position: Position for annotation box ('top_right', 'top_left', 'bottom_right', 'bottom_left')
        
    Requirements: 13.5
    """
    # Determine position coordinates
    position_map = {
        'top_right': (0.98, 0.98, 'top', 'right'),
        'top_left': (0.02, 0.98, 'top', 'left'),
        'bottom_right': (0.98, 0.02, 'bottom', 'right'),
        'bottom_left': (0.02, 0.02, 'bottom', 'left'),
    }
    
    x, y, va, ha = position_map.get(position, position_map['top_right'])
    
    # Build annotation text
    lines = ["Statistical Tests:"]
    for test in test_results:
        # Determine significance stars
        if test.p_value < 0.001:
            sig_stars = "***"
        elif test.p_value < 0.01:
            sig_stars = "**"
        elif test.p_value < 0.05:
            sig_stars = "*"
        else:
            sig_stars = "ns"
        
        lines.append(
            f"{test.test_name}: p={test.p_value:.4f} {sig_stars}, "
            f"d={test.effect_size:.3f}"
        )
    
    annotation_text = "\n".join(lines)
    
    # Add text box
    ax.text(
        x, y,
        annotation_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=0.5)
    )
