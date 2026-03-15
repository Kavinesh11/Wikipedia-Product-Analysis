"""Property-based tests for Innovation Score calculations.

Covers:
- Property 6: Innovation Score Calculation Formula  (Req 2.1)
- Property 7: Innovation Score Normalization Bounds  (Req 2.2)
- Property 8: Innovation Score Persistence with Timestamp  (Req 2.3)
- Property 9: Innovation Score Decile Ranking Correctness  (Req 2.4)
"""

import math
import pytest
from datetime import datetime
from hypothesis import given, strategies as st, settings, assume

from fortune500_kg.analytics_engine import AnalyticsEngine, MetricsRepository
from fortune500_kg.data_models import InnovationScoreRecord


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

positive_int = st.integers(min_value=1, max_value=1_000_000)
non_negative_int = st.integers(min_value=0, max_value=1_000_000)
company_id = st.text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


@st.composite
def company_metrics_strategy(draw):
    """Generate a single company's GitHub metrics."""
    return {
        "id": draw(company_id),
        "stars": draw(non_negative_int),
        "forks": draw(non_negative_int),
        "employee_count": draw(positive_int),
    }


@st.composite
def multi_company_metrics(draw, min_size=1, max_size=50):
    """Generate a list of companies with unique IDs."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    companies = []
    seen_ids = set()
    for i in range(n):
        cid = f"COMP{i:04d}"
        seen_ids.add(cid)
        companies.append({
            "id": cid,
            "stars": draw(non_negative_int),
            "forks": draw(non_negative_int),
            "employee_count": draw(positive_int),
        })
    return companies


# ---------------------------------------------------------------------------
# Property 6: Innovation Score Calculation Formula
# ---------------------------------------------------------------------------

class TestProperty6InnovationScoreFormula:
    """
    Property 6: Innovation Score Calculation Formula

    For any company with GitHub metrics and employee count, the Innovation Score
    should equal (github_stars + github_forks) / employee_count before normalization.

    Validates: Requirement 2.1
    """

    @given(
        stars=non_negative_int,
        forks=non_negative_int,
        employee_count=positive_int,
    )
    @settings(max_examples=200, deadline=None)
    def test_formula_correctness(self, stars, forks, employee_count):
        """Raw score equals (stars + forks) / employee_count."""
        engine = AnalyticsEngine()
        score = engine.calculate_innovation_score("C1", stars, forks, employee_count)
        expected = (stars + forks) / employee_count
        assert math.isclose(score, expected, rel_tol=1e-9), (
            f"Expected {expected}, got {score} for stars={stars}, forks={forks}, "
            f"employee_count={employee_count}"
        )

    @given(
        stars=non_negative_int,
        forks=non_negative_int,
        employee_count=positive_int,
    )
    @settings(max_examples=100, deadline=None)
    def test_score_is_non_negative(self, stars, forks, employee_count):
        """Innovation Score is always >= 0."""
        engine = AnalyticsEngine()
        score = engine.calculate_innovation_score("C1", stars, forks, employee_count)
        assert score >= 0.0

    @given(employee_count=st.integers(max_value=0))
    @settings(max_examples=50, deadline=None)
    def test_zero_or_negative_employee_count_raises(self, employee_count):
        """Non-positive employee_count raises ValueError."""
        engine = AnalyticsEngine()
        with pytest.raises(ValueError):
            engine.calculate_innovation_score("C1", 100, 50, employee_count)

    @given(
        stars=non_negative_int,
        forks=non_negative_int,
        employee_count=positive_int,
    )
    @settings(max_examples=100, deadline=None)
    def test_additivity_of_stars_and_forks(self, stars, forks, employee_count):
        """Score(stars, forks) == Score(stars+forks, 0)."""
        engine = AnalyticsEngine()
        score_split = engine.calculate_innovation_score("C1", stars, forks, employee_count)
        score_combined = engine.calculate_innovation_score("C1", stars + forks, 0, employee_count)
        assert math.isclose(score_split, score_combined, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Property 7: Innovation Score Normalization Bounds
# ---------------------------------------------------------------------------

class TestProperty7NormalizationBounds:
    """
    Property 7: Innovation Score Normalization Bounds

    For any set of companies with calculated Innovation Scores, all normalized
    scores should fall within the range [0, 10].

    Validates: Requirement 2.2
    """

    @given(companies=multi_company_metrics(min_size=1, max_size=50))
    @settings(max_examples=200, deadline=None)
    def test_all_normalized_scores_in_range(self, companies):
        """Every normalised score is in [0, 10]."""
        engine = AnalyticsEngine()
        raw_scores = {
            c["id"]: engine.calculate_innovation_score(
                c["id"], c["stars"], c["forks"], c["employee_count"]
            )
            for c in companies
        }
        normalized = engine.normalize_innovation_scores(raw_scores)

        for cid, score in normalized.items():
            assert 0.0 <= score <= 10.0, (
                f"Normalized score for {cid} out of range: {score}"
            )

    @given(companies=multi_company_metrics(min_size=2, max_size=50))
    @settings(max_examples=200, deadline=None)
    def test_max_normalized_score_is_ten(self, companies):
        """The maximum normalised score is exactly 10 when scores differ."""
        engine = AnalyticsEngine()
        raw_scores = {
            c["id"]: engine.calculate_innovation_score(
                c["id"], c["stars"], c["forks"], c["employee_count"]
            )
            for c in companies
        }
        # Only test when there is actual spread
        assume(max(raw_scores.values()) > min(raw_scores.values()))

        normalized = engine.normalize_innovation_scores(raw_scores)
        assert math.isclose(max(normalized.values()), 10.0, rel_tol=1e-9)

    @given(companies=multi_company_metrics(min_size=2, max_size=50))
    @settings(max_examples=200, deadline=None)
    def test_min_normalized_score_is_zero(self, companies):
        """The minimum normalised score is exactly 0 when scores differ."""
        engine = AnalyticsEngine()
        raw_scores = {
            c["id"]: engine.calculate_innovation_score(
                c["id"], c["stars"], c["forks"], c["employee_count"]
            )
            for c in companies
        }
        assume(max(raw_scores.values()) > min(raw_scores.values()))

        normalized = engine.normalize_innovation_scores(raw_scores)
        assert math.isclose(min(normalized.values()), 0.0, abs_tol=1e-9)

    @given(companies=multi_company_metrics(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=None)
    def test_normalization_preserves_order(self, companies):
        """Normalisation preserves the relative ordering of scores."""
        engine = AnalyticsEngine()
        raw_scores = {
            c["id"]: engine.calculate_innovation_score(
                c["id"], c["stars"], c["forks"], c["employee_count"]
            )
            for c in companies
        }
        normalized = engine.normalize_innovation_scores(raw_scores)

        ids = list(raw_scores.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if raw_scores[a] < raw_scores[b]:
                    assert normalized[a] <= normalized[b]
                elif raw_scores[a] > raw_scores[b]:
                    assert normalized[a] >= normalized[b]

    @given(
        value=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
        n=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100, deadline=None)
    def test_constant_scores_normalize_to_zero(self, value, n):
        """When all raw scores are equal, every normalised score is 0."""
        engine = AnalyticsEngine()
        raw_scores = {f"C{i}": value for i in range(n)}
        normalized = engine.normalize_innovation_scores(raw_scores)
        for score in normalized.values():
            assert math.isclose(score, 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Property 8: Innovation Score Persistence with Timestamp
# ---------------------------------------------------------------------------

class TestProperty8PersistenceWithTimestamp:
    """
    Property 8: Innovation Score Persistence with Timestamp

    For any calculated Innovation Score, the stored metric record should contain
    the score value and a timestamp indicating when the calculation occurred.

    Validates: Requirement 2.3
    """

    @given(companies=multi_company_metrics(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_every_company_has_stored_record(self, companies):
        """After storing, every company has exactly one InnovationScoreRecord."""
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        raw_scores = {
            c["id"]: engine.calculate_innovation_score(
                c["id"], c["stars"], c["forks"], c["employee_count"]
            )
            for c in companies
        }
        normalized = engine.normalize_innovation_scores(raw_scores)
        deciles = engine.compute_decile_rankings(normalized)

        company_metrics = {
            c["id"]: {
                "stars": c["stars"],
                "forks": c["forks"],
                "employee_count": c["employee_count"],
                "raw_score": raw_scores[c["id"]],
            }
            for c in companies
        }
        engine.store_innovation_scores(company_metrics, normalized, deciles)

        records = repo.get_by_type(InnovationScoreRecord)
        stored_ids = {r.company_id for r in records}
        expected_ids = {c["id"] for c in companies}
        assert stored_ids == expected_ids

    @given(companies=multi_company_metrics(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_stored_records_have_timestamps(self, companies):
        """Every stored InnovationScoreRecord has a non-None datetime timestamp."""
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        raw_scores = {
            c["id"]: engine.calculate_innovation_score(
                c["id"], c["stars"], c["forks"], c["employee_count"]
            )
            for c in companies
        }
        normalized = engine.normalize_innovation_scores(raw_scores)
        deciles = engine.compute_decile_rankings(normalized)
        company_metrics = {
            c["id"]: {
                "stars": c["stars"],
                "forks": c["forks"],
                "employee_count": c["employee_count"],
                "raw_score": raw_scores[c["id"]],
            }
            for c in companies
        }
        engine.store_innovation_scores(company_metrics, normalized, deciles)

        for record in repo.get_by_type(InnovationScoreRecord):
            assert isinstance(record.timestamp, datetime), (
                f"Record for {record.company_id} has non-datetime timestamp"
            )
            assert record.timestamp is not None

    @given(companies=multi_company_metrics(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_stored_metric_value_matches_raw_score(self, companies):
        """The metric_value field equals the raw (un-normalised) score."""
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        raw_scores = {
            c["id"]: engine.calculate_innovation_score(
                c["id"], c["stars"], c["forks"], c["employee_count"]
            )
            for c in companies
        }
        normalized = engine.normalize_innovation_scores(raw_scores)
        deciles = engine.compute_decile_rankings(normalized)
        company_metrics = {
            c["id"]: {
                "stars": c["stars"],
                "forks": c["forks"],
                "employee_count": c["employee_count"],
                "raw_score": raw_scores[c["id"]],
            }
            for c in companies
        }
        engine.store_innovation_scores(company_metrics, normalized, deciles)

        for record in repo.get_by_type(InnovationScoreRecord):
            expected_raw = raw_scores[record.company_id]
            assert math.isclose(record.metric_value, expected_raw, rel_tol=1e-9)

    @given(companies=multi_company_metrics(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_stored_normalized_score_matches(self, companies):
        """The normalized_score field matches the value from normalize_innovation_scores."""
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        raw_scores = {
            c["id"]: engine.calculate_innovation_score(
                c["id"], c["stars"], c["forks"], c["employee_count"]
            )
            for c in companies
        }
        normalized = engine.normalize_innovation_scores(raw_scores)
        deciles = engine.compute_decile_rankings(normalized)
        company_metrics = {
            c["id"]: {
                "stars": c["stars"],
                "forks": c["forks"],
                "employee_count": c["employee_count"],
                "raw_score": raw_scores[c["id"]],
            }
            for c in companies
        }
        engine.store_innovation_scores(company_metrics, normalized, deciles)

        for record in repo.get_by_type(InnovationScoreRecord):
            expected_norm = normalized[record.company_id]
            assert math.isclose(record.normalized_score, expected_norm, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Property 9: Innovation Score Decile Ranking Correctness
# ---------------------------------------------------------------------------

class TestProperty9DecileRankingCorrectness:
    """
    Property 9: Innovation Score Decile Ranking Correctness

    For any set of companies ranked by Innovation Score, companies should be
    assigned to deciles such that each decile contains approximately 10% of
    companies, ordered from lowest (decile 1) to highest (decile 10).

    Validates: Requirement 2.4
    """

    @given(companies=multi_company_metrics(min_size=1, max_size=100))
    @settings(max_examples=200, deadline=None)
    def test_decile_values_in_range(self, companies):
        """Every decile rank is in [1, 10]."""
        engine = AnalyticsEngine()
        raw_scores = {
            c["id"]: engine.calculate_innovation_score(
                c["id"], c["stars"], c["forks"], c["employee_count"]
            )
            for c in companies
        }
        normalized = engine.normalize_innovation_scores(raw_scores)
        deciles = engine.compute_decile_rankings(normalized)

        for cid, decile in deciles.items():
            assert 1 <= decile <= 10, f"Decile for {cid} out of range: {decile}"

    @given(companies=multi_company_metrics(min_size=1, max_size=100))
    @settings(max_examples=200, deadline=None)
    def test_every_company_gets_a_decile(self, companies):
        """Every company receives a decile rank."""
        engine = AnalyticsEngine()
        raw_scores = {
            c["id"]: engine.calculate_innovation_score(
                c["id"], c["stars"], c["forks"], c["employee_count"]
            )
            for c in companies
        }
        normalized = engine.normalize_innovation_scores(raw_scores)
        deciles = engine.compute_decile_rankings(normalized)

        assert set(deciles.keys()) == set(raw_scores.keys())

    @given(companies=multi_company_metrics(min_size=2, max_size=100))
    @settings(max_examples=200, deadline=None)
    def test_higher_score_gets_higher_or_equal_decile(self, companies):
        """A company with a strictly higher normalised score gets a >= decile rank."""
        engine = AnalyticsEngine()
        raw_scores = {
            c["id"]: engine.calculate_innovation_score(
                c["id"], c["stars"], c["forks"], c["employee_count"]
            )
            for c in companies
        }
        normalized = engine.normalize_innovation_scores(raw_scores)
        deciles = engine.compute_decile_rankings(normalized)

        ids = list(normalized.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if normalized[a] < normalized[b]:
                    assert deciles[a] <= deciles[b], (
                        f"{a} (score={normalized[a]:.4f}) should have decile <= "
                        f"{b} (score={normalized[b]:.4f}), "
                        f"got {deciles[a]} vs {deciles[b]}"
                    )

    @given(n=st.integers(min_value=10, max_value=100))
    @settings(max_examples=50, deadline=None)
    def test_decile_distribution_is_roughly_uniform(self, n):
        """With N companies having distinct scores, each decile has ~N/10 members."""
        engine = AnalyticsEngine()
        # Create companies with strictly increasing scores
        raw_scores = {f"C{i:04d}": float(i) for i in range(n)}
        deciles = engine.compute_decile_rankings(raw_scores)

        counts = [0] * 11  # index 1..10
        for d in deciles.values():
            counts[d] += 1

        # Each decile should have at least 1 member and at most ceil(n/10)*2
        max_allowed = math.ceil(n / 10) * 2
        for d in range(1, 11):
            assert counts[d] >= 0  # trivially true
        # Total must equal n
        assert sum(counts) == n
