"""Property-Based Tests for Fortune 500 Innovation Score and Correlation Analysis

Tests correctness properties for Innovation Score calculation, normalization,
decile rankings, persistence, and Pearson correlation calculations.
"""
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from scipy import stats as scipy_stats

from src.analytics.fortune500_analytics import (
    AnalyticsEngine,
    MetricsRepository,
)


# ============================================================================
# STRATEGIES
# ============================================================================

@st.composite
def company_metrics_strategy(draw):
    """Generate valid company GitHub metrics."""
    stars = draw(st.integers(min_value=0, max_value=1_000_000))
    forks = draw(st.integers(min_value=0, max_value=500_000))
    employee_count = draw(st.integers(min_value=1, max_value=500_000))
    return stars, forks, employee_count


@st.composite
def scores_dict_strategy(draw, min_companies=2, max_companies=50):
    """Gener