"""Unit tests for sector-level aggregation functionality (Task 16.1).

Covers:
- AnalyticsEngine.calculate_sector_averages (Requirement 13.1)
- AnalyticsEngine.identify_sector_extrema (Requirement 13.2)
- AnalyticsEngine.calculate_inter_sector_differences (Requirement 13.3)
- InsightGenerator.identify_best_practices (Requirement 13.5)
"""

import math
import pytest

from fortune500_kg.analytics_engine import AnalyticsEngine, MetricsRepository
from fortune500_kg.data_models import Company
from fortune500_kg.insight_generator import InsightGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return AnalyticsEngine(metrics_repo=MetricsRepository())


COMPANY_METRICS = {
    "C1": {"innovation_score": 8.0, "digital_maturity": 7.0},
    "C2": {"innovation_score": 6.0, "digital_maturity": 5.0},
    "C3": {"innovation_score": 4.0, "digital_maturity": 3.0},
    "C4": {"innovation_score": 2.0, "digital_maturity": 1.0},
}

COMPANY_SECTORS = {
    "C1": "Technology",
    "C2": "Technology",
    "C3": "Finance",
    "C4": "Finance",
}


# ---------------------------------------------------------------------------
# calculate_sector_averages
# ---------------------------------------------------------------------------

class TestCalculateSectorAverages:

    def test_basic_two_sector_averages(self, engine):
        result = engine.calculate_sector_averages(COMPANY_METRICS, COMPANY_SECTORS)
        assert set(result.keys()) == {"Technology", "Finance"}
        assert result["Technology"]["innovation_score"] == pytest.approx(7.0)
        assert result["Technology"]["digital_maturity"] == pytest.approx(6.0)
        assert result["Finance"]["innovation_score"] == pytest.approx(3.0)
        assert result["Finance"]["digital_maturity"] == pytest.approx(2.0)

    def test_single_company_per_sector(self, engine):
        metrics = {"A": {"score": 5.0}, "B": {"score": 9.0}}
        sectors = {"A": "Alpha", "B": "Beta"}
        result = engine.calculate_sector_averages(metrics, sectors)
        assert result["Alpha"]["score"] == pytest.approx(5.0)
        assert result["Beta"]["score"] == pytest.approx(9.0)

    def test_empty_inputs_return_empty(self, engine):
        assert engine.calculate_sector_averages({}, {}) == {}
        assert engine.calculate_sector_averages(COMPANY_METRICS, {}) == {}
        assert engine.calculate_sector_averages({}, COMPANY_SECTORS) == {}

    def test_company_not_in_metrics_is_ignored(self, engine):
        sectors = {"C1": "Technology", "UNKNOWN": "Technology"}
        result = engine.calculate_secto