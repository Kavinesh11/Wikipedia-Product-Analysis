"""Property-Based Tests for Digital Maturity Index

Tests correctness properties for DMI calculation, sector aggregation,
sector gap percentage, bottom quartile identification, and persistence.

Properties covered:
  - Property 16: Digital Maturity Index Calculation Formula (Req 4.1)
  - Property 17: Sector-Level Digital Maturity Aggregation (Req 4.2)
  - Property 18: Sector Gap Percentage Calculation (Req 4.3)
  - Property 19: Bottom Quartile Identification (Req 4.4)
  - Property 20: Digital Maturity Persistence with Metadata (Req 4.5)
"""
import math
from datetime import datetime

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from src.analytics.fortune500_analytics import (
    AnalyticsEngine,
    MetricsRepository,
)

# ============================================================================
# STRATEGIES
# ============================================================================

SECTORS = ["Tech", "Finance", "Healthcare", "Energy", "Retail"]

QUARTILES = ["top", "upper_mid", "lower_mid", "bottom"]


@st.composite
def dmi_inputs_strategy(draw):
    """Generate valid inputs for DMI calculation."""
    stars = draw(st.integers(min_value=0, max_value=1_000_000))
    forks = draw(st.integers(min_value=0, max_value=500_000))
    contributors = draw(st.integers(min_value=0, max_value=100_000))
    revenue_rank = draw(st.integers(min_value=1, max_value=500))
    return stars, forks, contributors, revenue_rank


@st.composite
def company_metrics_dict_strategy(draw, min_companies=2, max_companies=30):
    """Generate a dict of company_id -> metrics for sector-level tests."""
    n = draw(st.integers(min_value=min_companies, max_value=max_companies))
    companies = {}
    for i in range(n):
        company_id = f"company_{i}"
        companies[company_id] = {
            "stars": draw(st.integers(min_value=0, max_value=100_000)),
            "forks": draw(st.integers(min_value=0, max_value=50_000)),
            "contributors": draw(st.integers(min_value=0, max_value=10_000)),
            "revenue_rank": draw(st.integers(min_value=1, max_value=500)),
            "sector": draw(st.sampled_from(SECTORS)),
        }
    return companies


@st.composite
def dmi_values_dict_strategy(draw, min_companies=4, max_companies=40):
    """Generate a dict of company_id -> DMI float value."""
    n = draw(st.integers(min_value=min_companies, max_value=max_companies))
    return {
        f"company_{i}": draw(st.floats(min_value=0.0, max_value=10_000.0,
                                        allow_nan=False, allow_infinity=False))
        for i in range(n)
    }


# ============================================================================
# Property 16: Digital Maturity Index Calculation Formula (Req 4.1)
# ============================================================================

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(dmi_inputs_strategy())
def test_property_16_dmi_formula(inputs):
    """DMI must equal (stars + forks + contributors) / revenue_rank."""
    stars, forks, contributors, revenue_rank = inputs
    engine = AnalyticsEngine(MetricsRepository())

    result = engine.calculate_digital_maturity_index(
        company_id="test_co",
        stars=stars,
        forks=forks,
        contributors=contributors,
        revenue_rank=revenue_rank,
    )

    expected = (stars + forks + contributors) / revenue_rank
    assert math.isclose(result, expected, rel_tol=1e-9), (
        f"Expected {expected}, got {result}"
    )


@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(min_value=0, max_value=100_000),
       st.integers(min_value=0, max_value=50_000),
       st.integers(min_value=0, max_value=10_000))
def test_property_16_dmi_rejects_zero_revenue_rank(stars, forks, contributors):
    """DMI must raise ValueError when revenue_rank <= 0."""
    engine = AnalyticsEngine(MetricsRepository())

    with pytest.raises(ValueError):
        engine.calculate_digital_maturity_index(
            company_id="test_co",
            stars=stars,
            forks=forks,
            contributors=contributors,
            revenue_rank=0,
        )


# ============================================================================
# Property 17: Sector-Level Digital Maturity Aggregation (Req 4.2)
# ============================================================================

@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(company_metrics_dict_strategy())
def test_property_17_sector_avg_is_arithmetic_mean(company_metrics):
    """Sector average DMI must equal the arithmetic mean of company DMIs in that sector."""
    engine = AnalyticsEngine(MetricsRepository())

    sector_avgs = engine.calculate_sector_digital_maturity(company_metrics)

    # Manually compute expected averages
    from collections import defaultdict
    sector_buckets = defaultdict(list)
    for company_id, metrics in company_metrics.items():
        dmi = (metrics["stars"] + metrics["forks"] + metrics["contributors"]) / metrics["revenue_rank"]
        sector_buckets[metrics["sector"]].append(dmi)

    expected_avgs = {
        sector: sum(vals) / len(vals)
        for sector, vals in sector_buckets.items()
    }

    assert set(sector_avgs.keys()) == set(expected_avgs.keys())
    for sector, expected in expected_avgs.items():
        assert math.isclose(sector_avgs[sector], expected, rel_tol=1e-9), (
            f"Sector {sector}: expected {expected}, got {sector_avgs[sector]}"
        )


@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(company_metrics_dict_strategy(min_companies=1, max_companies=1))
def test_property_17_single_company_sector_avg_equals_dmi(company_metrics):
    """With one company per sector, sector avg must equal that company's DMI."""
    engine = AnalyticsEngine(MetricsRepository())

    sector_avgs = engine.calculate_sector_digital_maturity(company_metrics)

    for company_id, metrics in company_metrics.items():
        expected_dmi = (metrics["stars"] + metrics["forks"] + metrics["contributors"]) / metrics["revenue_rank"]
        sector = metrics["sector"]
        assert math.isclose(sector_avgs[sector], expected_dmi, rel_tol=1e-9)


# ============================================================================
# Property 18: Sector Gap Percentage Calculation (Req 4.3)
# ============================================================================

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False),
)
def test_property_18_sector_gap_formula(sector_a_avg, sector_b_avg):
    """Gap percentage must equal ((A - B) / B) * 100."""
    engine = AnalyticsEngine(MetricsRepository())

    result = engine.calculate_sector_gap_percentage(sector_a_avg, sector_b_avg)

    expected = ((sector_a_avg - sector_b_avg) / sector_b_avg) * 100.0
    assert math.isclose(result, expected, rel_tol=1e-9), (
        f"Expected {expected}, got {result}"
    )


@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False))
def test_property_18_gap_zero_when_equal(avg):
    """Gap percentage must be 0 when both sector averages are equal."""
    engine = AnalyticsEngine(MetricsRepository())

    result = engine.calculate_sector_gap_percentage(avg, avg)

    assert math.isclose(result, 0.0, abs_tol=1e-9)


@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False))
def test_property_18_gap_rejects_zero_denominator(sector_a_avg):
    """calculate_sector_gap_percentage must raise ValueError when sector_b_avg == 0."""
    engine = AnalyticsEngine(MetricsRepository())

    with pytest.raises(ValueError):
        engine.calculate_sector_gap_percentage(sector_a_avg, 0.0)


# ============================================================================
# Property 19: Bottom Quartile Identification (Req 4.4)
# ============================================================================

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(dmi_values_dict_strategy())
def test_property_19_bottom_quartile_count(dmi_values):
    """Bottom quartile must contain exactly floor(N/4) companies."""
    engine = AnalyticsEngine(MetricsRepository())

    result = engine.identify_bottom_quartile(dmi_values)

    expected_count = len(dmi_values) // 4
    assert len(result) == expected_count, (
        f"Expected {expected_count} companies in bottom quartile, got {len(result)}"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(dmi_values_dict_strategy())
def test_property_19_bottom_quartile_has_lowest_values(dmi_values):
    """All companies in the bottom quartile must have DMI <= any company not in it."""
    engine = AnalyticsEngine(MetricsRepository())

    bottom = set(engine.identify_bottom_quartile(dmi_values))
    not_bottom = set(dmi_values.keys()) - bottom

    if not bottom or not not_bottom:
        return  # Nothing to compare

    max_bottom_dmi = max(dmi_values[cid] for cid in bottom)
    min_not_bottom_dmi = min(dmi_values[cid] for cid in not_bottom)

    assert max_bottom_dmi <= min_not_bottom_dmi, (
        f"Bottom quartile max DMI {max_bottom_dmi} exceeds "
        f"non-bottom min DMI {min_not_bottom_dmi}"
    )


@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(dmi_values_dict_strategy())
def test_property_19_bottom_quartile_ids_are_valid(dmi_values):
    """All returned company IDs must exist in the input dict."""
    engine = AnalyticsEngine(MetricsRepository())

    result = engine.identify_bottom_quartile(dmi_values)

    for cid in result:
        assert cid in dmi_values


def test_property_19_empty_input_returns_empty():
    """identify_bottom_quartile must return empty list for empty input."""
    engine = AnalyticsEngine(MetricsRepository())
    assert engine.identify_bottom_quartile({}) == []


# ============================================================================
# Property 20: Digital Maturity Persistence with Metadata (Req 4.5)
# ============================================================================

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    dmi_inputs_strategy(),
    st.sampled_from(SECTORS),
    st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
    st.sampled_from(QUARTILES),
)
def test_property_20_dmi_record_persisted_with_metadata(inputs, sector, sector_avg, quartile):
    """Stored DigitalMaturityRecord must contain dmi_value, sector, and timestamp."""
    stars, forks, contributors, revenue_rank = inputs
    repo = MetricsRepository()
    engine = AnalyticsEngine(repo)

    dmi_value = engine.calculate_digital_maturity_index(
        company_id="test_co",
        stars=stars,
        forks=forks,
        contributors=contributors,
        revenue_rank=revenue_rank,
    )

    before = datetime.utcnow()
    record = engine.store_digital_maturity(
        company_id="test_co",
        dmi_value=dmi_value,
        stars=stars,
        forks=forks,
        contributors=contributors,
        revenue_rank=revenue_rank,
        sector=sector,
        sector_avg=sector_avg,
        quartile=quartile,
    )
    after = datetime.utcnow()

    # Record must be in the repository
    stored = repo.get_digital_maturity_records(company_id="test_co")
    assert len(stored) == 1
    stored_record = stored[0]

    # Must contain dmi_value
    assert math.isclose(stored_record.dmi_value, dmi_value, rel_tol=1e-9)

    # Must contain sector
    assert stored_record.sector == sector

    # Must contain timestamp within the test window
    assert before <= stored_record.timestamp <= after


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    dmi_inputs_strategy(),
    st.sampled_from(SECTORS),
    st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
    st.sampled_from(QUARTILES),
)
def test_property_20_dmi_record_contains_all_fields(inputs, sector, sector_avg, quartile):
    """Stored record must have all required fields populated correctly."""
    stars, forks, contributors, revenue_rank = inputs
    repo = MetricsRepository()
    engine = AnalyticsEngine(repo)

    dmi_value = engine.calculate_digital_maturity_index(
        company_id="co_1",
        stars=stars,
        forks=forks,
        contributors=contributors,
        revenue_rank=revenue_rank,
    )

    engine.store_digital_maturity(
        company_id="co_1",
        dmi_value=dmi_value,
        stars=stars,
        forks=forks,
        contributors=contributors,
        revenue_rank=revenue_rank,
        sector=sector,
        sector_avg=sector_avg,
        quartile=quartile,
    )

    records = repo.get_digital_maturity_records(company_id="co_1")
    assert len(records) == 1
    r = records[0]

    assert r.company_id == "co_1"
    assert r.stars == stars
    assert r.forks == forks
    assert r.contributors == contributors
    assert r.revenue_rank == revenue_rank
    assert r.sector == sector
    assert math.isclose(r.sector_avg, sector_avg, rel_tol=1e-9)
    assert r.quartile == quartile
    assert isinstance(r.timestamp, datetime)
