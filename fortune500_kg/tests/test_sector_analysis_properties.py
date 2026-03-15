"""Property-based tests for cross-sector comparative analysis (Task 16.2).

Properties:
- Property 62: Sector-Level Metric Aggregation Completeness
- Property 63: Sector Extrema Identification
- Property 64: Inter-Sector Percentage Difference Calculation
- Property 66: Best Practice Identification from High Performers

Validates: Requirements 13.1, 13.2, 13.3, 13.5
"""

from hypothesis import given, settings, assume
import hypothesis.strategies as st
import pytest

from fortune500_kg.analytics_engine import AnalyticsEngine, MetricsRepository
from fortune500_kg.insight_generator import InsightGenerator


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

_metric_value = st.floats(
    min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False
)

_sector_names = st.lists(
    st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        min_size=1,
        max_size=8,
    ),
    min_size=2,
    max_size=6,
    unique=True,
)

_metric_names = st.lists(
    st.sampled_from(["innovation_score", "digital_maturity", "ecosystem_centrality", "revenue_growth"]),
    min_size=1,
    max_size=3,
    unique=True,
)


@st.composite
def company_dataset(draw, min_companies=4, min_sectors=2):
    """Generate a consistent (company_metrics, company_sectors) dataset."""
    sectors = draw(_sector_names)
    assume(len(sectors) >= min_sectors)
    metric_names = draw(_metric_names)

    n_companies = draw(st.integers(min_value=min_companies, max_value=20))
    company_ids = [f"C{i}" for i in range(n_companies)]

    company_sectors = {
        cid: draw(st.sampled_from(sectors))
        for cid in company_ids
    }
    company_metrics = {
        cid: {m: draw(_metric_value) for m in metric_names}
        for cid in company_ids
    }
    return company_metrics, company_sectors, metric_names, sectors


# ---------------------------------------------------------------------------
# Property 62: Sector-Level Metric Aggregation Completeness
# Feature: fortune500-kg-analytics, Property 62
# Validates: Requirements 13.1
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(company_dataset())
def test_property_62_sector_aggregation_completeness(dataset):
    """
    For any key metric, sector-level averages should be calculated for all
    distinct sectors present in the dataset.

    **Validates: Requirements 13.1**
    """
    company_metrics, company_sectors, metric_names, _ = dataset
    engine = AnalyticsEngine(metrics_repo=MetricsRepository())
    result = engine.calculate_sector_averages(company_metrics, company_sectors)

    # All sectors that have at least one company in company_metrics must appear
    expected_sectors = set(
        sector for cid, sector in company_sectors.items()
        if cid in company_metrics
    )
    assert set(result.keys()) == expected_sectors

    # Each sector result must contain all metric names
    for sector in result:
        for metric in metric_names:
            assert metric in result[sector]


# ---------------------------------------------------------------------------
# Property 63: Sector Extrema Identification
# Feature: fortune500-kg-analytics, Property 63
# Validates: Requirements 13.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(company_dataset())
def test_property_63_sector_extrema_identification(dataset):
    """
    For any metric with sector-level averages, the identified highest sector
    should have the maximum average value and the lowest sector should have
    the minimum average value.

    **Validates: Requirements 13.2**
    """
    company_metrics, company_sectors, metric_names, _ = dataset
    engine = AnalyticsEngine(metrics_repo=MetricsRepository())
    sector_avgs = engine.calculate_sector_averages(company_metrics, company_sectors)

    assume(len(sector_avgs) >= 2)

    for metric in metric_names:
        extrema = engine.identify_sector_extrema(sector_avgs, metric)
        if not extrema:
            continue

        # Collect all sector averages for this metric
        metric_sector_values = {
            s: avgs[metric]
            for s, avgs in sector_avgs.items()
            if metric in avgs
        }

        highest_sector = extrema["highest"]
        lowest_sector = extrema["lowest"]

        # The highest sector must have the maximum value
        assert metric_sector_values[highest_sector] == max(metric_sector_values.values())
        # The lowest sector must have the minimum value
        assert metric_sector_values[lowest_sector] == min(metric_sector_values.values())


# ---------------------------------------------------------------------------
# Property 64: Inter-Sector Percentage Difference Calculation
# Feature: fortune500-kg-analytics, Property 64
# Validates: Requirements 13.3
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(company_dataset())
def test_property_64_inter_sector_percentage_difference(dataset):
    """
    For any two sectors A and B with average metric values M_A and M_B,
    the percentage difference should equal ((M_A - M_B) / M_B) × 100.

    **Validates: Requirements 13.3**
    """
    company_metrics, company_sectors, metric_names, _ = dataset
    engine = AnalyticsEngine(metrics_repo=MetricsRepository())
    sector_avgs = engine.calculate_sector_averages(company_metrics, company_sectors)

    assume(len(sector_avgs) >= 2)

    for metric in metric_names:
        differences = engine.calculate_inter_sector_differences(sector_avgs, metric)
        if not differences:
            continue

        # Collect sector averages for this metric
        metric_sector_values = {
            s: avgs[metric]
            for s, avgs in sector_avgs.items()
            if metric in avgs
        }

        # Verify each computed difference matches the formula
        for key, pct_diff in differences.items():
            sector_a, sector_b = key.split("_vs_")
            m_a = metric_sector_values[sector_a]
            m_b = metric_sector_values[sector_b]

            if m_b == 0.0:
                # When base is zero, implementation returns 0.0
                assert pct_diff == pytest.approx(0.0)
            else:
                expected = ((m_a - m_b) / m_b) * 100.0
                assert pct_diff == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Property 66: Best Practice Identification from High Performers
# Feature: fortune500-kg-analytics, Property 66
# Validates: Requirements 13.5
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(company_dataset(min_sectors=2))
def test_property_66_best_practice_from_high_performers(dataset):
    """
    For any generated best practice recommendation, the referenced practices
    should come from sectors with above-median average performance on the
    relevant metric.

    **Validates: Requirements 13.5**
    """
    company_metrics, company_sectors, metric_names, _ = dataset
    engine = AnalyticsEngine(metrics_repo=MetricsRepository())
    sector_avgs = engine.calculate_sector_averages(company_metrics, company_sectors)

    assume(len(sector_avgs) >= 2)

    insight_gen = InsightGenerator()
    practices = insight_gen.identify_best_practices(sector_avgs, metric_names=metric_names)

    for practice in practices:
        metric = practice.metric_name
        # Collect all sector averages for this metric
        metric_values = {
            s: avgs[metric]
            for s, avgs in sector_avgs.items()
            if metric in avgs
        }
        sorted_vals = sorted(metric_values.values())
        n = len(sorted_vals)
        if n % 2 == 1:
            median = sorted_vals[n // 2]
        else:
            median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

        # The practice's source sector must be above the median
        assert practice.sector_avg > median, (
            f"Best practice sector '{practice.sector}' has avg {practice.sector_avg} "
            f"which is not above median {median}"
        )
        # The stored overall_median must match the computed median
        assert practice.overall_median == pytest.approx(median)
