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


@pytest.fixture
def insight_gen():
    return InsightGenerator()


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
        result = engine.calculate_sector_averages(COMPANY_METRICS, sectors)
        # Only C1 is in both dicts; UNKNOWN is ignored
        assert result["Technology"]["innovation_score"] == pytest.approx(8.0)

    def test_all_sectors_present(self, engine):
        result = engine.calculate_sector_averages(COMPANY_METRICS, COMPANY_SECTORS)
        assert "Technology" in result
        assert "Finance" in result


# ---------------------------------------------------------------------------
# identify_sector_extrema
# ---------------------------------------------------------------------------

class TestIdentifySectorExtrema:

    def test_identifies_highest_and_lowest(self, engine):
        sector_avgs = {
            "Technology": {"innovation_score": 7.0},
            "Finance": {"innovation_score": 3.0},
            "Healthcare": {"innovation_score": 5.0},
        }
        result = engine.identify_sector_extrema(sector_avgs, "innovation_score")
        assert result["highest"] == "Technology"
        assert result["lowest"] == "Finance"

    def test_two_sectors(self, engine):
        sector_avgs = {
            "Technology": {"innovation_score": 7.0},
            "Finance": {"innovation_score": 3.0},
        }
        result = engine.identify_sector_extrema(sector_avgs, "innovation_score")
        assert result["highest"] == "Technology"
        assert result["lowest"] == "Finance"

    def test_missing_metric_returns_empty(self, engine):
        sector_avgs = {"Technology": {"innovation_score": 7.0}}
        result = engine.identify_sector_extrema(sector_avgs, "nonexistent_metric")
        assert result == {}

    def test_empty_sector_averages_returns_empty(self, engine):
        result = engine.identify_sector_extrema({}, "innovation_score")
        assert result == {}

    def test_single_sector_returns_same_for_both(self, engine):
        sector_avgs = {"Technology": {"innovation_score": 7.0}}
        result = engine.identify_sector_extrema(sector_avgs, "innovation_score")
        assert result["highest"] == "Technology"
        assert result["lowest"] == "Technology"


# ---------------------------------------------------------------------------
# calculate_inter_sector_differences
# ---------------------------------------------------------------------------

class TestCalculateInterSectorDifferences:

    def test_basic_percentage_difference(self, engine):
        sector_avgs = {
            "Technology": {"innovation_score": 8.0},
            "Finance": {"innovation_score": 4.0},
        }
        result = engine.calculate_inter_sector_differences(sector_avgs, "innovation_score")
        # Tech vs Finance: (8 - 4) / 4 * 100 = 100%
        assert result["Technology_vs_Finance"] == pytest.approx(100.0)
        # Finance vs Tech: (4 - 8) / 8 * 100 = -50%
        assert result["Finance_vs_Technology"] == pytest.approx(-50.0)

    def test_three_sectors_all_pairs(self, engine):
        sector_avgs = {
            "A": {"score": 10.0},
            "B": {"score": 5.0},
            "C": {"score": 2.0},
        }
        result = engine.calculate_inter_sector_differences(sector_avgs, "score")
        # 3 sectors => 6 ordered pairs
        assert len(result) == 6

    def test_single_sector_returns_empty(self, engine):
        sector_avgs = {"Technology": {"innovation_score": 7.0}}
        result = engine.calculate_inter_sector_differences(sector_avgs, "innovation_score")
        assert result == {}

    def test_empty_returns_empty(self, engine):
        result = engine.calculate_inter_sector_differences({}, "innovation_score")
        assert result == {}

    def test_zero_base_does_not_raise(self, engine):
        sector_avgs = {
            "Technology": {"score": 5.0},
            "Finance": {"score": 0.0},
        }
        result = engine.calculate_inter_sector_differences(sector_avgs, "score")
        # Finance_vs_Technology: (0 - 5) / 5 * 100 = -100%
        assert result["Finance_vs_Technology"] == pytest.approx(-100.0)
        # Technology_vs_Finance: base is 0, should return 0.0 (no division by zero)
        assert result["Technology_vs_Finance"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# InsightGenerator.identify_best_practices
# ---------------------------------------------------------------------------

class TestIdentifyBestPractices:

    def test_high_performer_identified(self, insight_gen):
        sector_avgs = {
            "Technology": {"innovation_score": 8.0},
            "Finance": {"innovation_score": 3.0},
        }
        practices = insight_gen.identify_best_practices(sector_avgs)
        assert len(practices) >= 1
        # Technology is above median, so it should be the source
        assert any(p.sector == "Technology" for p in practices)

    def test_best_practice_from_above_median_sector(self, insight_gen):
        sector_avgs = {
            "Tech": {"score": 9.0},
            "Finance": {"score": 5.0},
            "Healthcare": {"score": 3.0},
        }
        practices = insight_gen.identify_best_practices(sector_avgs, metric_names=["score"])
        # Median of [3, 5, 9] = 5; Tech (9) > 5 is high performer
        high_performer_sectors = {p.sector for p in practices}
        assert "Tech" in high_performer_sectors

    def test_best_practice_targets_below_median_sectors(self, insight_gen):
        sector_avgs = {
            "Tech": {"score": 9.0},
            "Finance": {"score": 5.0},
            "Healthcare": {"score": 3.0},
        }
        practices = insight_gen.identify_best_practices(sector_avgs, metric_names=["score"])
        for p in practices:
            # All target sectors should be at or below median
            assert p.sector_avg > p.overall_median

    def test_empty_sector_averages_returns_empty(self, insight_gen):
        assert insight_gen.identify_best_practices({}) == []

    def test_single_sector_returns_empty(self, insight_gen):
        sector_avgs = {"Technology": {"innovation_score": 7.0}}
        practices = insight_gen.identify_best_practices(sector_avgs)
        assert practices == []

    def test_best_practice_has_required_fields(self, insight_gen):
        sector_avgs = {
            "Technology": {"innovation_score": 8.0},
            "Finance": {"innovation_score": 3.0},
        }
        practices = insight_gen.identify_best_practices(sector_avgs)
        for p in practices:
            assert p.sector
            assert p.metric_name
            assert p.description
            assert isinstance(p.target_sectors, list)
