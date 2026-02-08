"""Specialized analysis functions for Wikipedia Product Health Analysis.

This module provides high-level analysis functions that orchestrate multiple
components to perform complex analyses like structural shift detection,
platform risk assessment, seasonality analysis, campaign effectiveness,
external event analysis, and forecasting.
"""

from wikipedia_health.specialized_analysis.structural_shifts import (
    analyze_structural_shifts,
    temporal_alignment_test,
    pre_post_comparison
)
from wikipedia_health.specialized_analysis.platform_risk import (
    assess_platform_risk,
    threshold_testing,
    scenario_analysis as platform_scenario_analysis
)
from wikipedia_health.specialized_analysis.seasonality import (
    analyze_seasonality,
    validate_seasonality,
    day_of_week_analysis,
    holiday_effect_modeling,
    utility_vs_leisure_classification
)
from wikipedia_health.specialized_analysis.campaigns import (
    evaluate_campaign,
    duration_analysis,
    cross_campaign_comparison
)
from wikipedia_health.specialized_analysis.external_events import (
    analyze_external_event,
    event_category_comparison
)
from wikipedia_health.specialized_analysis.forecasting import (
    generate_forecast,
    evaluate_forecast_accuracy,
    scenario_analysis as forecast_scenario_analysis
)

__all__ = [
    # Structural shifts
    'analyze_structural_shifts',
    'temporal_alignment_test',
    'pre_post_comparison',
    # Platform risk
    'assess_platform_risk',
    'threshold_testing',
    'platform_scenario_analysis',
    # Seasonality
    'analyze_seasonality',
    'validate_seasonality',
    'day_of_week_analysis',
    'holiday_effect_modeling',
    'utility_vs_leisure_classification',
    # Campaigns
    'evaluate_campaign',
    'duration_analysis',
    'cross_campaign_comparison',
    # External events
    'analyze_external_event',
    'event_category_comparison',
    # Forecasting
    'generate_forecast',
    'evaluate_forecast_accuracy',
    'forecast_scenario_analysis',
]
