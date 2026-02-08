"""Interactive dashboard components using Plotly.

This module provides interactive visualization components with hover tooltips,
drill-down functionality, and time period filtering.
"""

from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from pandas import Series, DataFrame
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, datetime

from wikipedia_health.models import (
    TimeSeriesData,
    CausalEffect,
    ForecastResult,
    TestResult,
    Finding,
)


def create_interactive_trend_plot(
    series: Series,
    dates: Optional[Series] = None,
    confidence_interval: Optional[Tuple[Series, Series]] = None,
    title: str = "Interactive Time Series Trend",
    ylabel: str = "Value",
    show_methodology_tooltip: bool = True
) -> go.Figure:
    """Create interactive trend plot with hover tooltips showing confidence intervals.
    
    Args:
        series: Time series values to plot
        dates: Optional date index
        confidence_interval: Optional tuple of (lower_bound, upper_bound) series
        title: Plot title
        ylabel: Y-axis label
        show_methodology_tooltip: Whether to add methodology explanation
        
    Returns:
        Plotly Figure object with interactivity
        
    Requirements: 13.7
    """
    fig = go.Figure()
    
    # Use dates if provided, otherwise use integer index
    x_values = dates if dates is not None else pd.Series(range(len(series)))
    
    # Add confidence interval as filled area
    if confidence_interval is not None:
        lower_bound, upper_bound = confidence_interval
        
        # Create hover text for confidence bands
        hover_text = [
            f"Date: {x}<br>Lower CI: {lower:.2f}<br>Upper CI: {upper:.2f}"
            for x, lower, upper in zip(x_values, lower_bound, upper_bound)
        ]
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(46, 134, 171, 0.3)',
            fill='tonexty',
            name='95% Confidence Interval',
            hovertext=hover_text,
            hoverinfo='text'
        ))
    
    # Add main series with detailed hover information
    hover_text = [
        f"Date: {x}<br>Value: {y:.2f}<br>Click for details"
        for x, y in zip(x_values, series)
    ]
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=series,
        mode='lines+markers',
        name='Observed',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=4),
        hovertext=hover_text,
        hoverinfo='text'
    ))
    
    # Layout with methodology annotation
    annotations = []
    if show_methodology_tooltip:
        annotations.append(
            dict(
                text="ℹ️ Hover over points for details. Shaded area shows 95% confidence interval.",
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                showarrow=False,
                font=dict(size=10, color="gray"),
                xanchor='center'
            )
        )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis_title="Date",
        yaxis_title=ylabel,
        hovermode='closest',
        template='plotly_white',
        annotations=annotations,
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add range slider for time period filtering
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    
    return fig


def create_interactive_campaign_plot(
    causal_effect: CausalEffect,
    dates: Optional[Series] = None,
    title: str = "Interactive Campaign Effect Analysis"
) -> go.Figure:
    """Create interactive campaign effect plot with drill-down to detailed test results.
    
    Args:
        causal_effect: CausalEffect object with observed and counterfactual data
        dates: Optional date index
        title: Plot title
        
    Returns:
        Plotly Figure object with interactivity
        
    Requirements: 13.7
    """
    fig = go.Figure()
    
    # Use dates if provided, otherwise use integer index
    x_values = dates if dates is not None else pd.Series(range(len(causal_effect.observed)))
    
    # Add counterfactual with hover info
    counterfactual_hover = [
        f"Date: {x}<br>Counterfactual: {y:.2f}<br>What would have happened without intervention"
        for x, y in zip(x_values, causal_effect.counterfactual)
    ]
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=causal_effect.counterfactual,
        mode='lines+markers',
        name='Counterfactual (No Intervention)',
        line=dict(color='#A23B72', width=2, dash='dash'),
        marker=dict(size=4, symbol='square'),
        hovertext=counterfactual_hover,
        hoverinfo='text'
    ))
    
    # Add observed with hover info
    observed_hover = [
        f"Date: {x}<br>Observed: {y:.2f}<br>Actual outcome with intervention"
        for x, y in zip(x_values, causal_effect.observed)
    ]
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=causal_effect.observed,
        mode='lines+markers',
        name='Observed',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=4),
        hovertext=observed_hover,
        hoverinfo='text'
    ))
    
    # Highlight treatment period
    treatment_start, treatment_end = causal_effect.treatment_period
    if dates is not None:
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
            
            fig.add_vrect(
                x0=first_date,
                x1=last_date,
                fillcolor="yellow",
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text="Treatment Period",
                annotation_position="top left"
            )
    
    # Add statistical annotation with detailed info
    effect_pct = causal_effect.percentage_effect()
    ci_lower, ci_upper = causal_effect.confidence_interval
    
    # Determine significance
    if causal_effect.p_value < 0.001:
        sig_text = "*** (p < 0.001)"
    elif causal_effect.p_value < 0.01:
        sig_text = "** (p < 0.01)"
    elif causal_effect.p_value < 0.05:
        sig_text = "* (p < 0.05)"
    else:
        sig_text = "(not significant)"
    
    annotation_text = (
        f"<b>Causal Effect Analysis</b><br>"
        f"Method: {causal_effect.method}<br>"
        f"Effect Size: {causal_effect.effect_size:.2f} ({effect_pct:+.1f}%)<br>"
        f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]<br>"
        f"p-value: {causal_effect.p_value:.4f} {sig_text}<br>"
        f"<i>Click annotation for methodology details</i>"
    )
    
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#333",
        borderwidth=1,
        borderpad=10
    )
    
    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='closest',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    
    return fig


def create_interactive_forecast_plot(
    forecast_result: ForecastResult,
    historical: Optional[Series] = None,
    historical_dates: Optional[Series] = None,
    forecast_dates: Optional[Series] = None,
    title: str = "Interactive Forecast with Uncertainty"
) -> go.Figure:
    """Create interactive forecast plot with uncertainty fans and hover details.
    
    Args:
        forecast_result: ForecastResult object with predictions
        historical: Optional historical data
        historical_dates: Optional dates for historical data
        forecast_dates: Optional dates for forecast period
        title: Plot title
        
    Returns:
        Plotly Figure object with interactivity
        
    Requirements: 13.7
    """
    fig = go.Figure()
    
    # Plot historical data if provided
    if historical is not None:
        hist_x = historical_dates if historical_dates is not None else \
                 pd.Series(range(len(historical)))
        
        hist_hover = [
            f"Date: {x}<br>Historical Value: {y:.2f}"
            for x, y in zip(hist_x, historical)
        ]
        
        fig.add_trace(go.Scatter(
            x=hist_x,
            y=historical,
            mode='lines+markers',
            name='Historical',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=3),
            hovertext=hist_hover,
            hoverinfo='text'
        ))
    
    # Prepare forecast x-axis
    if forecast_dates is not None:
        forecast_x = forecast_dates
    elif historical_dates is not None:
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
    
    # Add prediction interval
    interval_hover = [
        f"Date: {x}<br>Lower Bound: {lower:.2f}<br>Upper Bound: {upper:.2f}<br>"
        f"{int(forecast_result.confidence_level*100)}% Prediction Interval"
        for x, lower, upper in zip(
            forecast_x,
            forecast_result.lower_bound,
            forecast_result.upper_bound
        )
    ]
    
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast_result.upper_bound,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast_result.lower_bound,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(241, 143, 1, 0.2)',
        fill='tonexty',
        name=f'{int(forecast_result.confidence_level*100)}% Prediction Interval',
        hovertext=interval_hover,
        hoverinfo='text'
    ))
    
    # Add point forecast
    forecast_hover = [
        f"Date: {x}<br>Forecast: {y:.2f}<br>Model: {forecast_result.model_type}<br>"
        f"Click for model details"
        for x, y in zip(forecast_x, forecast_result.point_forecast)
    ]
    
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast_result.point_forecast,
        mode='lines+markers',
        name=f'Forecast ({forecast_result.model_type})',
        line=dict(color='#F18F01', width=2),
        marker=dict(size=4, symbol='square'),
        hovertext=forecast_hover,
        hoverinfo='text'
    ))
    
    # Add vertical line separating historical from forecast
    if historical is not None and historical_dates is not None:
        last_hist_date = historical_dates.iloc[-1] if hasattr(historical_dates, 'iloc') else historical_dates[-1]
        # Use add_shape instead of add_vline for better datetime support
        fig.add_shape(
            type="line",
            x0=last_hist_date,
            x1=last_hist_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="gray", width=2, dash="dash")
        )
        fig.add_annotation(
            x=last_hist_date,
            y=1,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            yanchor="bottom"
        )
    
    # Add methodology tooltip
    methodology_text = (
        f"<b>Forecast Methodology</b><br>"
        f"Model: {forecast_result.model_type}<br>"
        f"Horizon: {forecast_result.horizon} periods<br>"
        f"Confidence Level: {int(forecast_result.confidence_level*100)}%<br>"
        f"<i>Hover over points for detailed values</i>"
    )
    
    fig.add_annotation(
        text=methodology_text,
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        showarrow=False,
        font=dict(size=10),
        align="right",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#333",
        borderwidth=1,
        borderpad=10
    )
    
    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='closest',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ])
        )
    )
    
    return fig


def create_interactive_comparison_plot(
    groups: Dict[str, Series],
    test_result: Optional[TestResult] = None,
    title: str = "Interactive Group Comparison"
) -> go.Figure:
    """Create interactive comparison plot with drill-down to detailed test results.
    
    Args:
        groups: Dictionary mapping group names to value series
        test_result: Optional TestResult with statistical comparison
        title: Plot title
        
    Returns:
        Plotly Figure object with interactivity
        
    Requirements: 13.7
    """
    from scipy import stats
    
    fig = go.Figure()
    
    # Calculate means and confidence intervals
    group_names = list(groups.keys())
    means = []
    ci_lowers = []
    ci_uppers = []
    
    for name in group_names:
        data = groups[name]
        mean = data.mean()
        se = stats.sem(data)
        ci = se * stats.t.ppf((1 + 0.95) / 2, len(data) - 1)
        
        means.append(mean)
        ci_lowers.append(mean - ci)
        ci_uppers.append(mean + ci)
    
    # Create hover text with detailed statistics
    hover_texts = []
    for name, mean, ci_lower, ci_upper in zip(group_names, means, ci_lowers, ci_uppers):
        data = groups[name]
        hover_text = (
            f"<b>{name}</b><br>"
            f"Mean: {mean:.2f}<br>"
            f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]<br>"
            f"Std Dev: {data.std():.2f}<br>"
            f"N: {len(data)}<br>"
            f"<i>Click for distribution details</i>"
        )
        hover_texts.append(hover_text)
    
    # Create bar plot with error bars
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    fig.add_trace(go.Bar(
        x=group_names,
        y=means,
        error_y=dict(
            type='data',
            array=[u - m for u, m in zip(ci_uppers, means)],
            arrayminus=[m - l for m, l in zip(means, ci_lowers)],
            visible=True
        ),
        marker=dict(
            color=colors[:len(group_names)],
            line=dict(color='black', width=1.5)
        ),
        hovertext=hover_texts,
        hoverinfo='text',
        showlegend=False
    ))
    
    # Add statistical test results if provided
    annotations = []
    
    if test_result is not None:
        # Determine significance
        if test_result.p_value < 0.001:
            sig_text = "*** (p < 0.001)"
        elif test_result.p_value < 0.01:
            sig_text = "** (p < 0.01)"
        elif test_result.p_value < 0.05:
            sig_text = "* (p < 0.05)"
        else:
            sig_text = "(not significant)"
        
        stats_text = (
            f"<b>Statistical Test Results</b><br>"
            f"Test: {test_result.test_name}<br>"
            f"Statistic: {test_result.statistic:.4f}<br>"
            f"p-value: {test_result.p_value:.4f} {sig_text}<br>"
            f"Effect Size: {test_result.effect_size:.4f}<br>"
            f"95% CI: [{test_result.confidence_interval[0]:.2f}, "
            f"{test_result.confidence_interval[1]:.2f}]<br>"
            f"<i>{test_result.interpretation}</i>"
        )
        
        annotations.append(
            dict(
                text=stats_text,
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                showarrow=False,
                font=dict(size=10),
                align="right",
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#333",
                borderwidth=1,
                borderpad=10
            )
        )
    
    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis_title="Group",
        yaxis_title="Value",
        hovermode='closest',
        template='plotly_white',
        height=500,
        annotations=annotations
    )
    
    return fig


def create_findings_dashboard(
    findings: List[Finding],
    title: str = "Analysis Findings Dashboard"
) -> go.Figure:
    """Create comprehensive dashboard showing all findings with drill-down capability.
    
    Args:
        findings: List of Finding objects to display
        title: Dashboard title
        
    Returns:
        Plotly Figure with multiple subplots
        
    Requirements: 13.7
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Findings by Confidence Level",
            "Evidence Strength Distribution",
            "Statistical Tests Summary",
            "Causal Effects Summary"
        ),
        specs=[
            [{"type": "bar"}, {"type": "histogram"}],
            [{"type": "bar"}, {"type": "bar"}]
        ]
    )
    
    # 1. Findings by confidence level
    confidence_counts = {'high': 0, 'medium': 0, 'low': 0}
    for finding in findings:
        level = finding.confidence_level.lower()
        confidence_counts[level] = confidence_counts.get(level, 0) + 1
    
    fig.add_trace(
        go.Bar(
            x=list(confidence_counts.keys()),
            y=list(confidence_counts.values()),
            marker=dict(color=['#06A77D', '#F18F01', '#A23B72']),
            name="Confidence Levels",
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # 2. Evidence strength distribution
    strengths = [f.evidence_strength() for f in findings]
    fig.add_trace(
        go.Histogram(
            x=strengths,
            nbinsx=10,
            marker=dict(color='#2E86AB'),
            name="Evidence Strength",
            hovertemplate="Strength: %{x:.2f}<br>Count: %{y}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # 3. Statistical tests summary
    total_tests = sum(len(f.evidence) for f in findings)
    significant_tests = sum(
        len([t for t in f.evidence if t.is_significant])
        for f in findings
    )
    
    fig.add_trace(
        go.Bar(
            x=["Total Tests", "Significant"],
            y=[total_tests, significant_tests],
            marker=dict(color=['#2E86AB', '#06A77D']),
            name="Statistical Tests",
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # 4. Causal effects summary
    causal_methods = {}
    for finding in findings:
        for effect in finding.causal_effects:
            method = effect.method
            causal_methods[method] = causal_methods.get(method, 0) + 1
    
    if causal_methods:
        fig.add_trace(
            go.Bar(
                x=list(causal_methods.keys()),
                y=list(causal_methods.values()),
                marker=dict(color='#A23B72'),
                name="Causal Methods",
                hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#333')),
        showlegend=False,
        height=800,
        template='plotly_white'
    )
    
    return fig
