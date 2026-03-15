"""Property-based tests for ExecutiveReport generation in InsightGenerator.

Covers:
- Property 51: Executive Report Section Completeness  (Req 11.1)
- Property 52: Leaderboard Section Content Requirements  (Req 11.2)
- Property 53: Trends Section Temporal Coverage  (Req 11.3)
- Property 54: Recommendations Section Prioritization  (Req 11.4)
- Property 55: ROI Section Calculation Completeness  (Req 11.5)

**Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5**
"""

from datetime import datetime
from typing import List, Tuple

import pytest
from hypothesis import given, settings, strategies as st, assume

from fortune500_kg.analytics_engine import MetricsRepository
from fortune500_kg.data_models import (
    Company,
    DigitalMaturityRecord,
    EcosystemCentralityRecord,
    ExecutiveReport,
    InnovationScoreRecord,
    LeaderboardEntry,
    MetricsSummary,
    Recommendation,
    ROIAnalysis,
    TrendsAnalysis,
)
from fortune500_kg.insight_generator import InsightGenerator


# ---------------------------------------------------------------------------
# Strategies and helpers
# ---------------------------------------------------------------------------

@st.composite
def companies_with_innovation_scores(draw, min_size=1, max_size=20):
    """Generate a list of (Company, normalized_score) pairs."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    pairs = []
    for i in range(n):
        score = draw(
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        )
        company = Company(
            id=f"C{i:04d}",
            name=f"Corp {i}",
            sector="TechSector",
            revenue_rank=i + 1,
            employee_count=1000,
        )
        pairs.append((company, score))
    return pairs


@st.composite
def companies_with_multi_year_scores(draw, min_size=1, max_size=10):
    """Generate companies with innovation scores across multiple years (for trend tests)."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    n_years = draw(st.integers(min_value=2, max_value=4))
    pairs = []
    for i in range(n):
        company = Company(
            id=f"C{i:04d}",
            name=f"Corp {i}",
            sector="TechSector",
            revenue_rank=i + 1,
            employee_count=1000,
        )
        year_scores = []
        for y in range(n_years):
            score = draw(
                st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
            )
            year = 2020 + y
            year_scores.append((year, score))
        pairs.append((company, year_scores))
    return pairs


def _build_generator(company_score_pairs: List[Tuple[Company, float]]) -> InsightGenerator:
    """Build an InsightGenerator populated with InnovationScoreRecords."""
    repo = MetricsRepository()
    companies = []
    for company, score in company_score_pairs:
        companies.append(company)
        record = InnovationScoreRecord(
            company_id=company.id,
            metric_name="innovation_score",
            metric_value=score,
            timestamp=datetime.now(),
            normalized_score=score,
        )
        repo.save(record)
    return InsightGenerator(metrics_repo=repo, companies=companies)


def _build_generator_multi_year(
    company_year_scores: List[Tuple[Company, List[Tuple[int, float]]]]
) -> InsightGenerator:
    """Build an InsightGenerator with InnovationScoreRecords across multiple years."""
    repo = MetricsRepository()
    companies = []
    for company, year_scores in company_year_scores:
        companies.append(company)
        for year, score in year_scores:
            record = InnovationScoreRecord(
                company_id=company.id,
                metric_name="innovation_score",
                metric_value=score,
                timestamp=datetime(year, 6, 15),
                normalized_score=score,
            )
            repo.save(record)
    return InsightGenerator(metrics_repo=repo, companies=companies)


# ---------------------------------------------------------------------------
# Property 51: Executive Report Section Completeness
# ---------------------------------------------------------------------------

class TestProperty51ExecutiveReportSectionCompleteness:
    """
    Property 51: Executive Report Section Completeness

    For any generated ExecutiveReport, all required sections
    (metrics_summary, leaderboard, trends, recommendations, roi_analysis)
    must be present and non-None.

    **Validates: Requirements 11.1**
    """

    # Feature: fortune500-kg-analytics, Property 51: Executive Report Section Completeness
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_all_required_sections_are_present_and_non_none(self, pairs):
        """All required sections of ExecutiveReport must be present and non-None."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        assert isinstance(report, ExecutiveReport), (
            f"Expected ExecutiveReport, got {type(report)}"
        )
        assert report.metrics_summary is not None, "metrics_summary must not be None"
        assert report.leaderboard is not None, "leaderboard must not be None"
        assert report.trends is not None, "trends must not be None"
        assert report.recommendations is not None, "recommendations must not be None"
        assert report.roi_analysis is not None, "roi_analysis must not be None"

    # Feature: fortune500-kg-analytics, Property 51: Executive Report Section Completeness
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_report_has_valid_report_id_and_generation_date(self, pairs):
        """ExecutiveReport must have a non-empty report_id and a valid generation_date."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        assert isinstance(report.report_id, str) and report.report_id.strip(), (
            f"report_id must be a non-empty string, got {report.report_id!r}"
        )
        assert isinstance(report.generation_date, datetime), (
            f"generation_date must be a datetime, got {type(report.generation_date)}"
        )

    # Feature: fortune500-kg-analytics, Property 51: Executive Report Section Completeness
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_metrics_summary_is_correct_type(self, pairs):
        """metrics_summary must be a MetricsSummary instance."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        assert isinstance(report.metrics_summary, MetricsSummary), (
            f"metrics_summary must be MetricsSummary, got {type(report.metrics_summary)}"
        )

    # Feature: fortune500-kg-analytics, Property 51: Executive Report Section Completeness
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_trends_is_correct_type(self, pairs):
        """trends must be a TrendsAnalysis instance."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        assert isinstance(report.trends, TrendsAnalysis), (
            f"trends must be TrendsAnalysis, got {type(report.trends)}"
        )

    # Feature: fortune500-kg-analytics, Property 51: Executive Report Section Completeness
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_roi_analysis_is_correct_type(self, pairs):
        """roi_analysis must be a ROIAnalysis instance."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        assert isinstance(report.roi_analysis, ROIAnalysis), (
            f"roi_analysis must be ROIAnalysis, got {type(report.roi_analysis)}"
        )

    def test_empty_repository_still_produces_complete_report(self):
        """Even with no stored metrics, all sections must be present and non-None."""
        gen = InsightGenerator(metrics_repo=MetricsRepository(), companies=[])
        report = gen.generate_executive_report()

        assert report.metrics_summary is not None
        assert report.leaderboard is not None
        assert report.trends is not None
        assert report.recommendations is not None
        assert report.roi_analysis is not None


# ---------------------------------------------------------------------------
# Property 52: Leaderboard Section Content Requirements
# ---------------------------------------------------------------------------

class TestProperty52LeaderboardSectionContentRequirements:
    """
    Property 52: Leaderboard Section Content Requirements

    For any leaderboard entry, it must contain company_name, sector,
    innovation_score, and Fortune 500 rank (rank field).

    **Validates: Requirements 11.2**
    """

    # Feature: fortune500-kg-analytics, Property 52: Leaderboard Section Content Requirements
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_every_leaderboard_entry_has_required_fields(self, pairs):
        """Every LeaderboardEntry must have company_name, sector, innovation_score, rank."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        for entry in report.leaderboard:
            assert isinstance(entry, LeaderboardEntry), (
                f"Expected LeaderboardEntry, got {type(entry)}"
            )
            assert isinstance(entry.company_name, str) and entry.company_name, (
                f"company_name must be a non-empty string, got {entry.company_name!r}"
            )
            assert isinstance(entry.sector, str), (
                f"sector must be a string, got {type(entry.sector)}"
            )
            assert isinstance(entry.innovation_score, float), (
                f"innovation_score must be a float, got {type(entry.innovation_score)}"
            )
            assert isinstance(entry.rank, int) and entry.rank >= 1, (
                f"rank must be a positive integer, got {entry.rank!r}"
            )

    # Feature: fortune500-kg-analytics, Property 52: Leaderboard Section Content Requirements
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_leaderboard_ranks_are_unique_and_sequential(self, pairs):
        """Leaderboard ranks must be unique and form a contiguous sequence starting at 1."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        ranks = [entry.rank for entry in report.leaderboard]
        if ranks:
            assert sorted(ranks) == list(range(1, len(ranks) + 1)), (
                f"Leaderboard ranks must be 1..N, got {sorted(ranks)}"
            )

    # Feature: fortune500-kg-analytics, Property 52: Leaderboard Section Content Requirements
    @given(pairs=companies_with_innovation_scores(min_size=2, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_leaderboard_is_sorted_by_innovation_score_descending(self, pairs):
        """Leaderboard entries must be ordered by innovation_score descending (rank 1 = highest)."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        scores = [entry.innovation_score for entry in report.leaderboard]
        assert scores == sorted(scores, reverse=True), (
            f"Leaderboard not sorted by innovation_score descending: {scores}"
        )

    # Feature: fortune500-kg-analytics, Property 52: Leaderboard Section Content Requirements
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_leaderboard_count_matches_companies_with_scores(self, pairs):
        """Leaderboard must have one entry per company that has an innovation score."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        assert len(report.leaderboard) == len(pairs), (
            f"Expected {len(pairs)} leaderboard entries, got {len(report.leaderboard)}"
        )


# ---------------------------------------------------------------------------
# Property 53: Trends Section Temporal Coverage
# ---------------------------------------------------------------------------

class TestProperty53TrendsSectionTemporalCoverage:
    """
    Property 53: Trends Section Temporal Coverage

    For any TrendsAnalysis with multiple data points, the
    innovation_score_trend and digital_maturity_trend lists must be
    ordered chronologically by timestamp.

    **Validates: Requirements 11.3**
    """

    # Feature: fortune500-kg-analytics, Property 53: Trends Section Temporal Coverage
    @given(pairs=companies_with_multi_year_scores(min_size=1, max_size=10))
    @settings(max_examples=100, deadline=None)
    def test_innovation_score_trend_is_chronologically_ordered(self, pairs):
        """innovation_score_trend must be ordered chronologically by timestamp."""
        gen = _build_generator_multi_year(pairs)
        report = gen.generate_executive_report()

        timestamps = [ts for ts, _ in report.trends.innovation_score_trend]
        assert timestamps == sorted(timestamps), (
            f"innovation_score_trend not in chronological order: {timestamps}"
        )

    # Feature: fortune500-kg-analytics, Property 53: Trends Section Temporal Coverage
    @given(pairs=companies_with_multi_year_scores(min_size=1, max_size=10))
    @settings(max_examples=100, deadline=None)
    def test_digital_maturity_trend_is_chronologically_ordered(self, pairs):
        """digital_maturity_trend must be ordered chronologically by timestamp."""
        # Add DMI records alongside innovation scores
        repo = MetricsRepository()
        companies = []
        for company, year_scores in pairs:
            companies.append(company)
            for year, score in year_scores:
                inno_record = InnovationScoreRecord(
                    company_id=company.id,
                    metric_name="innovation_score",
                    metric_value=score,
                    timestamp=datetime(year, 6, 15),
                    normalized_score=score,
                )
                repo.save(inno_record)
                dmi_record = DigitalMaturityRecord(
                    company_id=company.id,
                    metric_name="digital_maturity_index",
                    metric_value=score * 0.5,
                    timestamp=datetime(year, 6, 15),
                    sector=company.sector,
                    sector_avg=score * 0.6,
                    quartile="upper_mid",
                )
                repo.save(dmi_record)
        gen = InsightGenerator(metrics_repo=repo, companies=companies)
        report = gen.generate_executive_report()

        timestamps = [ts for ts, _ in report.trends.digital_maturity_trend]
        assert timestamps == sorted(timestamps), (
            f"digital_maturity_trend not in chronological order: {timestamps}"
        )

    # Feature: fortune500-kg-analytics, Property 53: Trends Section Temporal Coverage
    @given(pairs=companies_with_multi_year_scores(min_size=1, max_size=10))
    @settings(max_examples=100, deadline=None)
    def test_trend_data_points_are_tuples_of_datetime_and_float(self, pairs):
        """Each trend data point must be a (datetime, float) tuple."""
        gen = _build_generator_multi_year(pairs)
        report = gen.generate_executive_report()

        for ts, val in report.trends.innovation_score_trend:
            assert isinstance(ts, datetime), (
                f"Trend timestamp must be datetime, got {type(ts)}"
            )
            assert isinstance(val, float), (
                f"Trend value must be float, got {type(val)}"
            )

    # Feature: fortune500-kg-analytics, Property 53: Trends Section Temporal Coverage
    @given(pairs=companies_with_multi_year_scores(min_size=1, max_size=10))
    @settings(max_examples=100, deadline=None)
    def test_sector_trends_are_chronologically_ordered(self, pairs):
        """Each sector's trend list in sector_trends must be chronologically ordered."""
        gen = _build_generator_multi_year(pairs)
        report = gen.generate_executive_report()

        for sector, trend in report.trends.sector_trends.items():
            timestamps = [ts for ts, _ in trend]
            assert timestamps == sorted(timestamps), (
                f"Sector '{sector}' trend not in chronological order: {timestamps}"
            )


# ---------------------------------------------------------------------------
# Property 54: Recommendations Section Prioritization
# ---------------------------------------------------------------------------

class TestProperty54RecommendationsSectionPrioritization:
    """
    Property 54: Recommendations Section Prioritization

    For any list of recommendations in an ExecutiveReport, they must be
    sorted by priority ascending (priority 1 = highest priority first).

    **Validates: Requirements 11.4**
    """

    # Feature: fortune500-kg-analytics, Property 54: Recommendations Section Prioritization
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_recommendations_sorted_by_priority_ascending(self, pairs):
        """Recommendations must be sorted by priority ascending (1 = highest first)."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        priorities = [rec.priority for rec in report.recommendations]
        assert priorities == sorted(priorities), (
            f"Recommendations not sorted by priority ascending: {priorities}"
        )

    # Feature: fortune500-kg-analytics, Property 54: Recommendations Section Prioritization
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_every_recommendation_has_valid_priority(self, pairs):
        """Every recommendation must have a priority in [1, 5]."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        for rec in report.recommendations:
            assert isinstance(rec.priority, int), (
                f"priority must be an int, got {type(rec.priority)}"
            )
            assert 1 <= rec.priority <= 5, (
                f"priority must be in [1, 5], got {rec.priority}"
            )

    # Feature: fortune500-kg-analytics, Property 54: Recommendations Section Prioritization
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_recommendations_is_a_list(self, pairs):
        """recommendations must be a list (possibly empty)."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        assert isinstance(report.recommendations, list), (
            f"recommendations must be a list, got {type(report.recommendations)}"
        )

    def test_empty_repository_returns_empty_recommendations_list(self):
        """With no stored metrics, recommendations must be an empty list."""
        gen = InsightGenerator(metrics_repo=MetricsRepository(), companies=[])
        report = gen.generate_executive_report()

        assert isinstance(report.recommendations, list)
        assert report.recommendations == []


# ---------------------------------------------------------------------------
# Property 55: ROI Section Calculation Completeness
# ---------------------------------------------------------------------------

class TestProperty55ROISectionCalculationCompleteness:
    """
    Property 55: ROI Section Calculation Completeness

    For any ROIAnalysis, the roi_ratio must equal total_benefits / system_costs
    (when system_costs > 0), and all monetary fields must be non-negative.

    **Validates: Requirements 11.5**
    """

    # Feature: fortune500-kg-analytics, Property 55: ROI Section Calculation Completeness
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_roi_ratio_equals_total_benefits_divided_by_system_costs(self, pairs):
        """roi_ratio must equal total_benefits / system_costs when system_costs > 0."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        roi = report.roi_analysis
        if roi.system_costs > 0:
            expected_ratio = roi.total_benefits / roi.system_costs
            assert abs(roi.roi_ratio - expected_ratio) < 1e-9, (
                f"roi_ratio={roi.roi_ratio} != total_benefits/system_costs="
                f"{roi.total_benefits}/{roi.system_costs}={expected_ratio}"
            )

    # Feature: fortune500-kg-analytics, Property 55: ROI Section Calculation Completeness
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_all_monetary_fields_are_non_negative(self, pairs):
        """All monetary fields in ROIAnalysis must be non-negative."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        roi = report.roi_analysis
        monetary_fields = {
            "time_savings_hours": roi.time_savings_hours,
            "time_savings_value": roi.time_savings_value,
            "revenue_impact_top_quartile": roi.revenue_impact_top_quartile,
            "revenue_impact_bottom_quartile": roi.revenue_impact_bottom_quartile,
            "knowledge_loss_avoidance": roi.knowledge_loss_avoidance,
            "total_benefits": roi.total_benefits,
            "system_costs": roi.system_costs,
        }
        for field_name, value in monetary_fields.items():
            assert value >= 0, (
                f"ROIAnalysis.{field_name} must be non-negative, got {value}"
            )

    # Feature: fortune500-kg-analytics, Property 55: ROI Section Calculation Completeness
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_roi_analysis_has_all_required_fields(self, pairs):
        """ROIAnalysis must have all required fields populated."""
        gen = _build_generator(pairs)
        report = gen.generate_executive_report()

        roi = report.roi_analysis
        assert isinstance(roi.time_savings_hours, float), "time_savings_hours must be float"
        assert isinstance(roi.time_savings_value, float), "time_savings_value must be float"
        assert isinstance(roi.total_benefits, float), "total_benefits must be float"
        assert isinstance(roi.system_costs, float), "system_costs must be float"
        assert isinstance(roi.roi_ratio, float), "roi_ratio must be float"

    # Feature: fortune500-kg-analytics, Property 55: ROI Section Calculation Completeness
    @given(pairs=companies_with_innovation_scores(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_roi_ratio_is_zero_when_system_costs_is_zero(self, pairs):
        """When system_costs == 0, roi_ratio must be 0.0 (no division by zero)."""
        gen = _build_generator(pairs)
        # Override system_costs to 0 by calling calculate_roi directly
        roi_metrics = gen.calculate_roi(system_costs=0.0)
        assert roi_metrics.roi_ratio == 0.0, (
            f"Expected roi_ratio=0.0 when system_costs=0, got {roi_metrics.roi_ratio}"
        )

    def test_empty_repository_roi_analysis_is_complete(self):
        """Even with no stored metrics, ROIAnalysis must have all fields non-negative."""
        gen = InsightGenerator(metrics_repo=MetricsRepository(), companies=[])
        report = gen.generate_executive_report()

        roi = report.roi_analysis
        assert roi.total_benefits >= 0
        assert roi.system_costs >= 0
        assert roi.roi_ratio >= 0
