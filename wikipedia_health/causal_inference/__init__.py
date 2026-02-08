"""Causal Inference Engine.

Implements causal analysis methodologies including interrupted time series,
difference-in-differences, event study, and synthetic control methods.
"""

from wikipedia_health.causal_inference.interrupted_time_series import (
    InterruptedTimeSeriesAnalyzer,
    ITSAModel
)
from wikipedia_health.causal_inference.difference_in_differences import (
    DifferenceInDifferencesAnalyzer,
    DiDModel
)
from wikipedia_health.causal_inference.event_study import (
    EventStudyAnalyzer,
    BaselineModel,
    EventImpact
)
from wikipedia_health.causal_inference.synthetic_control import (
    SyntheticControlBuilder,
    SyntheticControl,
    PlaceboResult
)

__all__ = [
    'InterruptedTimeSeriesAnalyzer',
    'ITSAModel',
    'DifferenceInDifferencesAnalyzer',
    'DiDModel',
    'EventStudyAnalyzer',
    'BaselineModel',
    'EventImpact',
    'SyntheticControlBuilder',
    'SyntheticControl',
    'PlaceboResult',
]
