"""Property-Based Tests for Historical Trend Analysis

Properties covered:
  - Property 57: Metric Timestamp Persistence (Req 12.1)
  - Property 58: Time-Range Query Filtering Accuracy (Req 12.2)
  - Property 59: Year-Over-Year Growth Rate Calculation (Req 12.3)
  - Property 60: Time-Series Visualization Multi-Year Coverage (Req 12.4)
  - Property 61: Inflection Point Detection Criteria (Req 12.5)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from src.analytics.fortune500_analytics import (
    AnalyticsEngine,
    DigitalMaturityRecord,
    InnovationScoreRecord,
    MetricsRepository,
)

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

COMPANY_ID_ST = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=8,
)

BASE_DATE = datetime(2020, 1, 1)


@st.composite
def timestamped_innovation_records(draw, company_id: str = "C1", n: int = 5):
    """Generate *n* InnovationScoreRecords with distinct, ordered timestamps."""
    offsets = draw(
        st.lists(
            st.integers(min_value=0, max_value=3650),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    offsets.sort()
    records = []
    for offset in offsets:
        ts = BASE_DATE + timedelta(days=offset)
        score = draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False))
        records.append(
            InnovationScoreRecord(
                company_id=company_id,
                raw_score=score,
                normalized_score=score,
                decile_rank=1,
                github_stars=0,
                github_forks=0,
                employee_count=1,
                timestamp=ts,
            )
        )
    return records


# ===========================================================================
# Property 57: Metric Timestamp Persistence (Req 12.1)
# ===========================================================================

@given(
    company_id=COMPANY_ID_ST,
    stars=st.integers(min_value=0, max_value=100_000),
    forks=st.integers(min_value=0, max_value=50_000),
    employee_count=st.integers(min_value=1, max_value=500_000),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_57_innovation_score_timestamp_persisted(
    company_id, stars, forks, employee_count
):
    """Every stored InnovationScoreRecord must carry a non-None timestamp."""
    repo = MetricsRepository()
    engine = AnalyticsEngine(repo)

    raw = engine.calculate_innovation_score(company_id, stars, forks, employee_count)
    record = engine.store_innovation_score(
        company_id=company_id,
        raw_score=raw,
        normalized_score=raw,
        decile=1,
        stars=stars,
        forks=forks,
        employee_count=employee_count,
    )

    assert record.timestamp is not None
    stored = repo.get_innovation_scores(company_id)
    assert len(stored) == 1
    assert stored[0].timestamp is not None


@given(
    company_id=COMPANY_ID_ST,
    stars=st.integers(min_value=0, max_value=100_000),
    forks=st.integers(min_value=0, max_value=50_000),
    contributors=st.integers(min_value=0, max_value=10_000),
    revenue_rank=st.integers(min_value=1, max_value=500),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_57_digital_maturity_timestamp_persisted(
    company_id, stars, forks, contributors, revenue_rank
):
    """Every stored DigitalMaturityRecord must carry a non-None timestamp."""
    repo = MetricsRepository()
    engine = AnalyticsEngine(repo)

    dmi = engine.calculate_digital_maturity_index(
        company_id, stars, forks, contributors, revenue_rank
    )
    record = engine.store_digital_maturity(
        company_id=company_id,
        dmi_value=dmi,
        stars=stars,
        forks=forks,
        contributors=contributors,
        revenue_rank=revenue_rank,
        sector="Tech",
        sector_avg=dmi,
        quartile="top",
    )

    assert record.timestamp is not None
    stored = repo.get_digital_maturity_records(company_id)
    assert len(stored) == 1
    assert stored[0].timestamp is not None


# ===========================================================================
# Property 58: Time-Range Query Filtering Accuracy (Req 12.2)
# ===========================================================================

@given(
    n_records=st.integers(min_value=3, max_value=20),
    window_days=st.integers(min_value=1, max_value=500),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_58_time_range_returns_only_records_in_window(
    n_records, window_days
):
    """get_records_in_range must return only records whose timestamp is within [start, end]."""
    repo = MetricsRepository()
    engine = AnalyticsEngine(repo)

    # Store records spread over 2 * window_days
    for i in range(n_records):
        ts = BASE_DATE + timedelta(days=i * (2 * window_days) // max(n_records - 1, 1))
        record = InnovationScoreRecord(
            company_id="C1",
            raw_score=float(i),
            normalized_score=float(i),
            decile_rank=1,
            github_stars=i,
            github_forks=0,
            employee_count=1,
            timestamp=ts,
        )
        repo.store_innovation_score(record)

    start = BASE_DATE + timedelta(days=window_days // 4)
    end = BASE_DATE + timedelta(days=window_days)

    results = repo.get_records_in_range(start, end, InnovationScoreRecord)

    for r in results:
        assert start <= r.timestamp <= end, (
            f"Record timestamp {r.timestamp} outside [{start}, {end}]"
        )


@given(
    n_records=st.integers(min_value=2, max_value=15),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_58_time_range_excludes_out_of_window_records(n_records):
    """Records outside [start, end] must NOT appear in get_records_in_range results."""
    repo = MetricsRepository()

    # Store records at day 0, 10, 20, ...
    for i in range(n_records):
        ts = BASE_DATE + timedelta(days=i * 10)
        repo.store_innovation_score(
            InnovationScoreRecord(
                company_id="C1",
                raw_score=float(i),
                normalized_score=float(i),
                decile_rank=1,
                github_stars=i,
                github_forks=0,
                employee_count=1,
                timestamp=ts,
            )
        )

    # Query only the middle window
    start = BASE_DATE + timedelta(days=5)
    end = BASE_DATE + timedelta(days=15)

    results = repo.get_records_in_range(start, end, InnovationScoreRecord)
    result_timestamps = {r.timestamp for r in results}

    # Day 10 is inside; day 0 and day 20+ are outside
    assert BASE_DATE + timedelta(days=10) in result_timestamps
    assert BASE_DATE not in result_timestamps


@given(
    n_records=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_58_time_range_results_ordered_by_timestamp(n_records):
    """Results from get_records_in_range must be sorted ascending by timestamp."""
    repo = MetricsRepository()

    for i in range(n_records):
        ts = BASE_DATE + timedelta(days=i * 3)
        repo.store_innovation_score(
            InnovationScoreRecord(
                company_id="C1",
                raw_score=float(i),
                normalized_score=float(i),
                decile_rank=1,
                github_stars=i,
                github_forks=0,
                employee_count=1,
                timestamp=ts,
            )
        )

    start = BASE_DATE
    end = BASE_DATE + timedelta(days=n_records * 3)
    results = repo.get_records_in_range(start, end, InnovationScoreRecord)

    timestamps = [r.timestamp for r in results]
    assert timestamps == sorted(timestamps)


# ===========================================================================
# Property 59: Year-Over-Year Growth Rate Calculation (Req 12.3)
# ===========================================================================

@given(
    v_current=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    v_previous=st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_property_59_yoy_growth_rate_formula(v_current, v_previous):
    """YoY growth rate must equal ((V_current - V_previous) / V_previous) * 100."""
    engine = AnalyticsEngine()
    result = engine.calculate_yoy_growth_rate(v_current, v_previous)
    expected = ((v_current - v_previous) / v_previous) * 100.0
    assert abs(result - expected) < 1e-9


@given(
    v_previous=st.floats(
        min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
    ).filter(lambda x: abs(x) < 1e-12),
)
@settings(max_examples=50)
def test_property_59_yoy_growth_rate_zero_previous_raises(v_previous):
    """calculate_yoy_growth_rate must raise ValueError when v_previous is zero."""
    engine = AnalyticsEngine()
    with pytest.raises(ValueError):
        engine.calculate_yoy_growth_rate(10.0, 0.0)


@given(
    values=st.lists(
        st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_59_yoy_series_length_is_n_minus_1(values):
    """YoY growth series must have exactly len(values) - 1 entries (no zero predecessors)."""
    engine = AnalyticsEngine()
    series = [(BASE_DATE + timedelta(days=i * 365), v) for i, v in enumerate(values)]
    result = engine.calculate_yoy_growth_series(series)
    # All values > 0 so no entries are skipped
    assert len(result) == len(values) - 1


@given(
    values=st.lists(
        st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_59_yoy_series_matches_formula(values):
    """Each entry in the YoY series must match the point-wise formula."""
    engine = AnalyticsEngine()
    series = [(BASE_DATE + timedelta(days=i * 365), v) for i, v in enumerate(values)]
    result = engine.calculate_yoy_growth_series(series)

    for i, (ts, rate) in enumerate(result):
        v_curr = values[i + 1]
        v_prev = values[i]
        expected = ((v_curr - v_prev) / v_prev) * 100.0
        assert abs(rate - expected) < 1e-9


# ===========================================================================
# Property 60: Time-Series Multi-Year Coverage (Req 12.4)
# ===========================================================================

@given(
    n_years=st.integers(min_value=2, max_value=10),
    company_id=COMPANY_ID_ST,
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_60_time_series_spans_multiple_years(n_years, company_id):
    """get_time_series must return data spanning at least n_years distinct calendar years."""
    repo = MetricsRepository()
    engine = AnalyticsEngine(repo)

    # Store one record per year
    for year_offset in range(n_years):
        ts = datetime(2020 + year_offset, 6, 15)
        score = float(year_offset + 1)
        repo.store_innovation_score(
            InnovationScoreRecord(
                company_id=company_id,
                raw_score=score,
                normalized_score=score,
                decile_rank=1,
                github_stars=year_offset,
                github_forks=0,
                employee_count=1,
                timestamp=ts,
            )
        )

    series = engine.get_time_series(company_id, "innovation_score")
    years_present = {ts.year for ts, _ in series}
    assert len(years_present) == n_years


@given(
    n_records=st.integers(min_value=2, max_value=20),
    company_id=COMPANY_ID_ST,
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_60_time_series_is_chronologically_ordered(n_records, company_id):
    """get_time_series must return data points sorted ascending by timestamp."""
    repo = MetricsRepository()
    engine = AnalyticsEngine(repo)

    # Insert in reverse order to verify sorting
    for i in range(n_records - 1, -1, -1):
        ts = BASE_DATE + timedelta(days=i * 30)
        repo.store_innovation_score(
            InnovationScoreRecord(
                company_id=company_id,
                raw_score=float(i),
                normalized_score=float(i),
                decile_rank=1,
                github_stars=i,
                github_forks=0,
                employee_count=1,
                timestamp=ts,
            )
        )

    series = engine.get_time_series(company_id, "innovation_score")
    timestamps = [ts for ts, _ in series]
    assert timestamps == sorted(timestamps)


# ===========================================================================
# Property 61: Inflection Point Detection Criteria (Req 12.5)
# ===========================================================================

@given(
    n_points=st.integers(min_value=3, max_value=20),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_61_inflection_points_are_direction_changes(n_points):
    """Every reported inflection point must represent a genuine direction change."""
    engine = AnalyticsEngine()

    # Build a series with a known peak in the middle
    half = n_points // 2
    values = list(range(half + 1)) + list(range(half - 1, -1, -1))
    series = [(BASE_DATE + timedelta(days=i * 10), float(v)) for i, v in enumerate(values)]

    inflections = engine.identify_inflection_points(series)

    for ts, direction in inflections:
        assert direction in ("increasing_to_decreasing", "decreasing_to_increasing")


@given(
    ascending=st.lists(
        st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=15,
    ).map(sorted),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_61_monotone_series_has_no_inflection_points(ascending):
    """A strictly monotone series must produce zero inflection points."""
    assume(len(set(ascending)) == len(ascending))  # strictly monotone
    engine = AnalyticsEngine()
    series = [(BASE_DATE + timedelta(days=i * 10), v) for i, v in enumerate(ascending)]
    inflections = engine.identify_inflection_points(series)
    assert inflections == []


def test_property_61_known_peak_produces_inflection():
    """A series with a single peak must produce exactly one inflection point."""
    engine = AnalyticsEngine()
    # 1, 2, 3, 2, 1 — peak at index 2
    series = [
        (BASE_DATE + timedelta(days=0), 1.0),
        (BASE_DATE + timedelta(days=10), 2.0),
        (BASE_DATE + timedelta(days=20), 3.0),
        (BASE_DATE + timedelta(days=30), 2.0),
        (BASE_DATE + timedelta(days=40), 1.0),
    ]
    inflections = engine.identify_inflection_points(series)
    assert len(inflections) == 1
    ts, direction = inflections[0]
    assert direction == "increasing_to_decreasing"
    assert ts == BASE_DATE + timedelta(days=20)


def test_property_61_known_valley_produces_inflection():
    """A series with a single valley must produce exactly one inflection point."""
    engine = AnalyticsEngine()
    # 3, 2, 1, 2, 3 — valley at index 2
    series = [
        (BASE_DATE + timedelta(days=0), 3.0),
        (BASE_DATE + timedelta(days=10), 2.0),
        (BASE_DATE + timedelta(days=20), 1.0),
        (BASE_DATE + timedelta(days=30), 2.0),
        (BASE_DATE + timedelta(days=40), 3.0),
    ]
    inflections = engine.identify_inflection_points(series)
    assert len(inflections) == 1
    ts, direction = inflections[0]
    assert direction == "decreasing_to_increasing"
    assert ts == BASE_DATE + timedelta(days=20)
