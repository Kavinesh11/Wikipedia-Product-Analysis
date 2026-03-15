"""Unit tests for InsightGenerator.identify_underperformers (Task 11.1).

Validates: Requirement 9.1
"""

from datetime import datetime

import pytest

from fortune500_kg.analytics_engine import AnalyticsEngine, MetricsRepository
from fortune500_kg.data_models import Company, InnovationScoreRecord
from fortune500_kg.insight_generator import InsightGenerator, UnderperformerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_company(cid: str, name: str, sector: str) -> Company:
    return Company(
        id=cid,
        name=name,
        sector=sector,
        revenue_rank=1,
        employee_count=1000,
    )


def store_score(repo: MetricsRepository, company_id: str, normalized_score: float) -> None:
    """Directly store an InnovationScoreRecord with a given normalized score."""
    record = InnovationScoreRecord(
        company_id=company_id,
        metric_name="innovation_score",
        metric_value=normalized_score,
        timestamp=datetime.now(),
        normalized_score=normalized_score,
    )
    repo.save(record)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIdentifyUnderperformers:
    """Tests for InsightGenerator.identify_underperformers."""

    def test_returns_empty_list_when_no_companies_in_sector(self):
        """No records for the sector → empty result."""
        repo = MetricsRepository()
        companies = [make_company("C1", "Alpha", "Finance")]
        store_score(repo, "C1", 5.0)

        gen = InsightGenerator(metrics_repo=repo, companies=companies)
        result = gen.identify_underperformers("Technology")

        assert result == []

    def test_returns_empty_list_when_repo_is_empty(self):
        """Empty repo → empty result."""
        repo = MetricsRepository()
        gen = InsightGenerator(metrics_repo=repo, companies=[])
        assert gen.identify_underperformers("Technology") == []

    def test_single_company_is_never_underperformer(self):
        """With one company, its score equals the average → not below average."""
        repo = MetricsRepository()
        companies = [make_company("C1", "Solo Corp", "Technology")]
        store_score(repo, "C1", 4.0)

        gen = InsightGenerator(metrics_repo=repo, companies=companies)
        result = gen.identify_underperformers("Technology")

        assert result == []

    def test_identifies_companies_below_sector_average(self):
        """Companies with score < sector average are returned."""
        repo = MetricsRepository()
        companies = [
            make_company("C1", "High Corp", "Technology"),
            make_company("C2", "Low Corp", "Technology"),
            make_company("C3", "Mid Corp", "Technology"),
        ]
        # Scores: 8, 2, 5 → average = 5.0
        store_score(repo, "C1", 8.0)
        store_score(repo, "C2", 2.0)
        store_score(repo, "C3", 5.0)

        gen = InsightGenerator(metrics_repo=repo, companies=companies)
        result = gen.identify_underperformers("Technology")

        # Only C2 (score=2) is strictly below average (5.0)
        assert len(result) == 1
        assert result[0].company.id == "C2"

    def test_gap_is_sector_average_minus_company_score(self):
        """Gap equals sector_average - company_score."""
        repo = MetricsRepository()
        companies = [
            make_company("C1", "High Corp", "Technology"),
            make_company("C2", "Low Corp", "Technology"),
        ]
        # Scores: 8, 2 → average = 5.0
        store_score(repo, "C1", 8.0)
        store_score(repo, "C2", 2.0)

        gen = InsightGenerator(metrics_repo=repo, companies=companies)
        result = gen.identify_underperformers("Technology")

        assert len(result) == 1
        r = result[0]
        assert r.sector_average == pytest.approx(5.0)
        assert r.innovation_score == pytest.approx(2.0)
        assert r.gap == pytest.approx(3.0)

    def test_gap_is_always_positive(self):
        """Gap must be positive for all returned underperformers."""
        repo = MetricsRepository()
        companies = [make_company(f"C{i}", f"Corp{i}", "Finance") for i in range(5)]
        scores = [1.0, 3.0, 5.0, 7.0, 9.0]  # average = 5.0
        for c, s in zip(companies, scores):
            store_score(repo, c.id, s)

        gen = InsightGenerator(metrics_repo=repo, companies=companies)
        result = gen.identify_underperformers("Finance")

        for r in result:
            assert r.gap > 0

    def test_results_sorted_by_gap_descending(self):
        """Worst underperformers (largest gap) appear first."""
        repo = MetricsRepository()
        companies = [
            make_company("C1", "Corp1", "Technology"),
            make_company("C2", "Corp2", "Technology"),
            make_company("C3", "Corp3", "Technology"),
        ]
        # Scores: 9, 3, 1 → average = 4.33...
        store_score(repo, "C1", 9.0)
        store_score(repo, "C2", 3.0)
        store_score(repo, "C3", 1.0)

        gen = InsightGenerator(metrics_repo=repo, companies=companies)
        result = gen.identify_underperformers("Technology")

        gaps = [r.gap for r in result]
        assert gaps == sorted(gaps, reverse=True)

    def test_only_filters_by_requested_sector(self):
        """Companies from other sectors are excluded."""
        repo = MetricsRepository()
        companies = [
            make_company("C1", "Tech Corp", "Technology"),
            make_company("C2", "Finance Corp", "Finance"),
            make_company("C3", "Tech Corp 2", "Technology"),
        ]
        store_score(repo, "C1", 8.0)
        store_score(repo, "C2", 1.0)  # Finance – should be ignored
        store_score(repo, "C3", 2.0)

        gen = InsightGenerator(metrics_repo=repo, companies=companies)
        result = gen.identify_underperformers("Technology")

        # Average for Technology = (8+2)/2 = 5.0; C3 is below
        returned_ids = {r.company.id for r in result}
        assert "C2" not in returned_ids
        assert "C3" in returned_ids

    def test_uses_most_recent_record_per_company(self):
        """When multiple records exist for a company, the latest is used."""
        repo = MetricsRepository()
        companies = [
            make_company("C1", "Corp1", "Technology"),
            make_company("C2", "Corp2", "Technology"),
        ]
        # Store an old low score for C1, then a newer high score
        old_record = InnovationScoreRecord(
            company_id="C1",
            metric_name="innovation_score",
            metric_value=1.0,
            timestamp=datetime(2023, 1, 1),
            normalized_score=1.0,
        )
        new_record = InnovationScoreRecord(
            company_id="C1",
            metric_name="innovation_score",
            metric_value=9.0,
            timestamp=datetime(2024, 1, 1),
            normalized_score=9.0,
        )
        repo.save(old_record)
        repo.save(new_record)
        store_score(repo, "C2", 3.0)

        gen = InsightGenerator(metrics_repo=repo, companies=companies)
        result = gen.identify_underperformers("Technology")

        # C1 latest score = 9.0, C2 = 3.0 → average = 6.0; C2 is underperformer
        returned_ids = {r.company.id for r in result}
        assert "C2" in returned_ids
        assert "C1" not in returned_ids

    def test_result_contains_correct_company_object(self):
        """The Company object in the result matches the original."""
        repo = MetricsRepository()
        company = make_company("C1", "Specific Corp", "Healthcare")
        companies = [company, make_company("C2", "Other Corp", "Healthcare")]
        store_score(repo, "C1", 1.0)
        store_score(repo, "C2", 9.0)

        gen = InsightGenerator(metrics_repo=repo, companies=companies)
        result = gen.identify_underperformers("Healthcare")

        assert len(result) == 1
        assert result[0].company is company
