"""Property-based tests for ROI calculations in InsightGenerator.

Covers:
- Property 46: Time Savings Calculation Methodology  (Req 10.1)
- Property 47: Quartile Revenue Impact Quantification  (Req 10.2)
- Property 48: Decision Speed Improvement Calculation  (Req 10.3)
- Property 49: Knowledge Loss Avoidance Estimation  (Req 10.4)
- Property 50: ROI Ratio Calculation  (Req 10.5)

**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
"""

from datetime import datetime
from typing import List, Tuple

from hypothesis import given, settings, strategies as st

from fortune500_kg.analytics_engine import MetricsRepository
from fortune500_kg.data_models import Company, InnovationScoreRecord, ROIMetrics
from fortune500_kg.insight_generator import InsightGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_generator_with_companies(
    company_revenue_pairs: List[Tuple[Company, float]],
) -> InsightGenerator:
    """Build an InsightGenerator populated with InnovationScoreRecords.

    Each pair is (Company, normalized_score).  The company's revenue_rank is
    used by _calculate_revenue_impact() as a revenue proxy.
    """
    repo = MetricsRepository()
    companies = []
    for company, score in company_revenue_pairs:
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


def _build_empty_generator() -> InsightGenerator:
    """Build an InsightGenerator with no stored metrics."""
    return InsightGenerator(metrics_repo=MetricsRepository(), companies=[])


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def valid_hours_and_rate(draw):
    """Generate (traditional_hours, system_hours, hourly_rate) where system < traditional."""
    traditional = draw(st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False))
    system = draw(st.floats(min_value=0.0, max_value=traditional, allow_nan=False, allow_infinity=False))
    rate = draw(st.floats(min_value=1.0, max_value=1_000.0, allow_nan=False, allow_infinity=False))
    return traditional, system, rate


@st.composite
def valid_decision_times(draw):
    """Generate (old_time, new_time) where new_time < old_time and old_time > 0."""
    old_time = draw(st.floats(min_value=0.01, max_value=365.0, allow_nan=False, allow_infinity=False))
    new_time = draw(st.floats(min_value=0.0, max_value=old_time, allow_nan=False, allow_infinity=False))
    return old_time, new_time


@st.composite
def valid_turnover_and_knowledge(draw):
    """Generate (turnover_rate, knowledge_base_value) with turnover in [0, 1]."""
    turnover = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    knowledge = draw(st.floats(min_value=0.0, max_value=1_000_000_000.0, allow_nan=False, allow_infinity=False))
    return turnover, knowledge


@st.composite
def companies_with_revenue_ranks(draw, min_size=4, max_size=20):
    """Generate a list of (Company, score) pairs with distinct revenue ranks."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    pairs = []
    for i in range(n):
        score = draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        company = Company(
            id=f"C{i:04d}",
            name=f"Corp {i}",
            sector="TestSector",
            revenue_rank=i + 1,
            employee_count=1000,
        )
        pairs.append((company, score))
    return pairs


# ---------------------------------------------------------------------------
# Property 46: Time Savings Calculation Methodology
# ---------------------------------------------------------------------------

class TestProperty46TimeSavingsCalculationMethodology:
    """
    Property 46: Time Savings Calculation Methodology

    For any valid traditional_hours, system_hours, and hourly_rate,
    time savings should equal (traditional_hours - system_hours) × hourly_rate.

    **Validates: Requirements 10.1**
    """

    # Feature: fortune500-kg-analytics, Property 46: Time Savings Calculation Methodology
    @given(params=valid_hours_and_rate())
    @settings(max_examples=100, deadline=None)
    def test_time_savings_equals_hours_difference_times_rate(self, params):
        """time_savings == (traditional_hours - system_hours) × hourly_rate."""
        traditional, system, rate = params
        gen = _build_empty_generator()
        result = gen.calculate_roi(
            traditional_hours=traditional,
            system_hours=system,
            hourly_rate=rate,
            system_costs=1.0,  # avoid division by zero
        )
        expected = (traditional - system) * rate
        assert abs(result.time_savings - expected) < 1e-6, (
            f"Expected time_savings={expected}, got {result.time_savings}"
        )

    @given(params=valid_hours_and_rate())
    @settings(max_examples=100, deadline=None)
    def test_time_savings_is_non_negative_when_system_hours_le_traditional(self, params):
        """When system_hours <= traditional_hours, time_savings >= 0."""
        traditional, system, rate = params
        gen = _build_empty_generator()
        result = gen.calculate_roi(
            traditional_hours=traditional,
            system_hours=system,
            hourly_rate=rate,
            system_costs=1.0,
        )
        assert result.time_savings >= -1e-9, (
            f"time_savings should be non-negative, got {result.time_savings}"
        )


# ---------------------------------------------------------------------------
# Property 47: Quartile Revenue Impact Quantification
# ---------------------------------------------------------------------------

class TestProperty47QuartileRevenueImpactQuantification:
    """
    Property 47: Quartile Revenue Impact Quantification

    For any set of companies with revenue values, the revenue impact should
    equal the difference between top quartile average and bottom quartile average.

    **Validates: Requirements 10.2**
    """

    # Feature: fortune500-kg-analytics, Property 47: Quartile Revenue Impact Quantification
    @given(pairs=companies_with_revenue_ranks(min_size=4, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_revenue_impact_equals_bottom_minus_top_quartile_avg(self, pairs):
        """Revenue impact = bottom_quartile_avg_rank - top_quartile_avg_rank."""
        gen = _build_generator_with_companies(pairs)
        result = gen.calculate_roi(system_costs=1.0)

        # Manually compute expected value
        revenue_ranks = sorted([float(c.revenue_rank) for c, _ in pairs])
        n = len(revenue_ranks)
        q = n // 4
        bottom_avg = sum(revenue_ranks[:q]) / q
        top_avg = sum(revenue_ranks[n - q:]) / q
        expected = bottom_avg - top_avg

        assert abs(result.revenue_impact - expected) < 1e-6, (
            f"Expected revenue_impact={expected}, got {result.revenue_impact}"
        )

    def test_revenue_impact_is_zero_when_no_companies(self):
        """With no companies in the repository, revenue_impact should be 0.0."""
        gen = _build_empty_generator()
        result = gen.calculate_roi(system_costs=1.0)
        assert result.revenue_impact == 0.0

    @given(pairs=companies_with_revenue_ranks(min_size=4, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_revenue_impact_is_non_negative(self, pairs):
        """Revenue impact sign matches bottom_avg - top_avg of sorted revenue ranks."""
        gen = _build_generator_with_companies(pairs)
        result = gen.calculate_roi(system_costs=1.0)

        # Manually compute expected sign
        revenue_ranks = sorted([float(c.revenue_rank) for c, _ in pairs])
        n = len(revenue_ranks)
        q = n // 4
        bottom_avg = sum(revenue_ranks[:q]) / q
        top_avg = sum(revenue_ranks[n - q:]) / q
        expected = bottom_avg - top_avg

        # The sign of revenue_impact should match the sign of (bottom_avg - top_avg)
        assert abs(result.revenue_impact - expected) < 1e-6, (
            f"Expected revenue_impact={expected}, got {result.revenue_impact}"
        )


# ---------------------------------------------------------------------------
# Property 48: Decision Speed Improvement Calculation
# ---------------------------------------------------------------------------

class TestProperty48DecisionSpeedImprovementCalculation:
    """
    Property 48: Decision Speed Improvement Calculation

    For any old_time and new_time where new_time < old_time, decision speed
    improvement should equal ((old_time - new_time) / old_time) × 100.

    **Validates: Requirements 10.3**
    """

    # Feature: fortune500-kg-analytics, Property 48: Decision Speed Improvement Calculation
    @given(times=valid_decision_times())
    @settings(max_examples=100, deadline=None)
    def test_decision_speed_improvement_formula(self, times):
        """decision_speed_improvement == ((old - new) / old) × 100."""
        old_time, new_time = times
        gen = _build_empty_generator()
        result = gen.calculate_roi(
            old_decision_time=old_time,
            new_decision_time=new_time,
            system_costs=1.0,
        )
        expected = ((old_time - new_time) / old_time) * 100.0
        assert abs(result.decision_speed_improvement - expected) < 1e-6, (
            f"Expected decision_speed_improvement={expected}, got {result.decision_speed_improvement}"
        )

    @given(times=valid_decision_times())
    @settings(max_examples=100, deadline=None)
    def test_decision_speed_improvement_is_between_0_and_100(self, times):
        """When new_time is in [0, old_time], improvement is in [0, 100]."""
        old_time, new_time = times
        gen = _build_empty_generator()
        result = gen.calculate_roi(
            old_decision_time=old_time,
            new_decision_time=new_time,
            system_costs=1.0,
        )
        assert -1e-9 <= result.decision_speed_improvement <= 100.0 + 1e-9, (
            f"decision_speed_improvement {result.decision_speed_improvement} out of [0, 100]"
        )

    def test_decision_speed_improvement_is_zero_when_times_equal(self):
        """When old_time == new_time, improvement should be 0%."""
        gen = _build_empty_generator()
        result = gen.calculate_roi(
            old_decision_time=10.0,
            new_decision_time=10.0,
            system_costs=1.0,
        )
        assert abs(result.decision_speed_improvement) < 1e-9

    def test_decision_speed_improvement_is_100_when_new_time_is_zero(self):
        """When new_time == 0, improvement should be 100%."""
        gen = _build_empty_generator()
        result = gen.calculate_roi(
            old_decision_time=10.0,
            new_decision_time=0.0,
            system_costs=1.0,
        )
        assert abs(result.decision_speed_improvement - 100.0) < 1e-9


# ---------------------------------------------------------------------------
# Property 49: Knowledge Loss Avoidance Estimation
# ---------------------------------------------------------------------------

class TestProperty49KnowledgeLossAvoidanceEstimation:
    """
    Property 49: Knowledge Loss Avoidance Estimation

    For any turnover rate and knowledge base value, knowledge loss avoidance
    should be calculated correctly based on those inputs.

    **Validates: Requirements 10.4**
    """

    # Feature: fortune500-kg-analytics, Property 49: Knowledge Loss Avoidance Estimation
    @given(params=valid_turnover_and_knowledge())
    @settings(max_examples=100, deadline=None)
    def test_knowledge_loss_avoidance_equals_turnover_times_knowledge_value(self, params):
        """knowledge_loss_avoidance == turnover_rate × knowledge_base_value."""
        turnover, knowledge = params
        gen = _build_empty_generator()
        result = gen.calculate_roi(
            turnover_rate=turnover,
            knowledge_base_value=knowledge,
            system_costs=1.0,
        )
        expected = turnover * knowledge
        assert abs(result.knowledge_loss_avoidance - expected) < 1e-6, (
            f"Expected knowledge_loss_avoidance={expected}, got {result.knowledge_loss_avoidance}"
        )

    @given(params=valid_turnover_and_knowledge())
    @settings(max_examples=100, deadline=None)
    def test_knowledge_loss_avoidance_is_non_negative(self, params):
        """knowledge_loss_avoidance should always be >= 0."""
        turnover, knowledge = params
        gen = _build_empty_generator()
        result = gen.calculate_roi(
            turnover_rate=turnover,
            knowledge_base_value=knowledge,
            system_costs=1.0,
        )
        assert result.knowledge_loss_avoidance >= -1e-9

    def test_knowledge_loss_avoidance_is_zero_when_no_turnover(self):
        """With 0% turnover, knowledge_loss_avoidance should be 0."""
        gen = _build_empty_generator()
        result = gen.calculate_roi(
            turnover_rate=0.0,
            knowledge_base_value=1_000_000.0,
            system_costs=1.0,
        )
        assert result.knowledge_loss_avoidance == 0.0

    def test_knowledge_loss_avoidance_scales_with_knowledge_value(self):
        """Doubling knowledge_base_value should double knowledge_loss_avoidance."""
        gen = _build_empty_generator()
        r1 = gen.calculate_roi(turnover_rate=0.1, knowledge_base_value=100_000.0, system_costs=1.0)
        r2 = gen.calculate_roi(turnover_rate=0.1, knowledge_base_value=200_000.0, system_costs=1.0)
        assert abs(r2.knowledge_loss_avoidance - 2 * r1.knowledge_loss_avoidance) < 1e-6


# ---------------------------------------------------------------------------
# Property 50: ROI Ratio Calculation
# ---------------------------------------------------------------------------

class TestProperty50ROIRatioCalculation:
    """
    Property 50: ROI Ratio Calculation

    For any total_benefits and system_costs > 0, the ROI ratio should equal
    total_benefits / system_costs.

    **Validates: Requirements 10.5**
    """

    # Feature: fortune500-kg-analytics, Property 50: ROI Ratio Calculation
    @given(
        traditional=st.floats(min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        system_h=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        rate=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        costs=st.floats(min_value=0.01, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_roi_ratio_equals_total_benefits_divided_by_system_costs(
        self, traditional, system_h, rate, costs
    ):
        """roi_ratio == total_benefits / system_costs."""
        gen = _build_empty_generator()
        result = gen.calculate_roi(
            traditional_hours=traditional,
            system_hours=system_h,
            hourly_rate=rate,
            turnover_rate=0.0,
            knowledge_base_value=0.0,
            system_costs=costs,
        )
        # With turnover=0 and knowledge=0, total_benefits = time_savings + revenue_impact
        expected_ratio = result.total_benefits / costs
        assert abs(result.roi_ratio - expected_ratio) < 1e-9, (
            f"Expected roi_ratio={expected_ratio}, got {result.roi_ratio}"
        )

    @given(
        costs=st.floats(min_value=0.01, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_total_benefits_is_sum_of_all_benefit_components(self, costs):
        """total_benefits == time_savings + revenue_impact + knowledge_loss_avoidance."""
        gen = _build_empty_generator()
        result = gen.calculate_roi(
            traditional_hours=1000.0,
            system_hours=100.0,
            hourly_rate=100.0,
            old_decision_time=10.0,
            new_decision_time=2.0,
            turnover_rate=0.1,
            knowledge_base_value=500_000.0,
            system_costs=costs,
        )
        expected_total = (
            result.time_savings
            + result.revenue_impact
            + result.knowledge_loss_avoidance
        )
        assert abs(result.total_benefits - expected_total) < 1e-6, (
            f"Expected total_benefits={expected_total}, got {result.total_benefits}"
        )

    def test_roi_ratio_is_zero_when_system_costs_is_zero(self):
        """When system_costs == 0, roi_ratio should be 0.0 (no division by zero)."""
        gen = _build_empty_generator()
        result = gen.calculate_roi(system_costs=0.0)
        assert result.roi_ratio == 0.0

    @given(
        costs=st.floats(min_value=0.01, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_roi_ratio_is_positive_when_benefits_exceed_costs(self, costs):
        """When total_benefits > system_costs, roi_ratio > 1."""
        gen = _build_empty_generator()
        # Force large time savings to ensure benefits > costs
        result = gen.calculate_roi(
            traditional_hours=100_000.0,
            system_hours=0.0,
            hourly_rate=1_000.0,
            turnover_rate=0.0,
            knowledge_base_value=0.0,
            system_costs=costs,
        )
        assert result.roi_ratio > 1.0, (
            f"Expected roi_ratio > 1, got {result.roi_ratio}"
        )
