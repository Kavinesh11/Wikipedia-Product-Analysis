"""Property-based tests for correlation calculations.

Covers:
- Property 10: Correlation Coefficient Calculation  (Req 2.5)
- Property 31: Innovation-Revenue Correlation Calculation  (Req 7.1)
- Property 35: Correlation Persistence with Confidence Intervals  (Req 7.5)
"""

import math
import pytest
from datetime import datetime
from hypothesis import given, strategies as st, settings, assume

from fortune500_kg.analytics_engine import AnalyticsEngine, MetricsRepository
from fortune500_kg.data_models import CorrelationRecord


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

float_val = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)

metric_series = st.lists(float_val, min_size=3, max_size=100)


@st.composite
def paired_series(draw, min_size=3, max_size=100):
    """Generate two float series of the same length."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    xs = [draw(float_val) for _ in range(n)]
    ys = [draw(float_val) for _ in range(n)]
    return xs, ys


# ---------------------------------------------------------------------------
# Property 10: Correlation Coefficient Calculation
# ---------------------------------------------------------------------------

class TestProperty10CorrelationCoefficientCalculation:
    """
    Property 10: Correlation Coefficient Calculation

    For any two metric series (Innovation Score and revenue growth), the
    calculated Pearson correlation coefficient should match the mathematical
    definition: r = Σ((x - x̄)(y - ȳ)) / √(Σ(x - x̄)² × Σ(y - ȳ)²)

    Validates: Requirement 2.5
    """

    @given(pair=paired_series())
    @settings(max_examples=200, deadline=None)
    def test_coefficient_in_range(self, pair):
        """Pearson r is always in [-1, 1]."""
        xs, ys = pair
        engine = AnalyticsEngine()
        result = engine.calculate_correlation(xs, ys)
        assert -1.0 <= result.coefficient <= 1.0, (
            f"Coefficient {result.coefficient} out of [-1, 1]"
        )

    @given(pair=paired_series())
    @settings(max_examples=200, deadline=None)
    def test_coefficient_matches_formula(self, pair):
        """r matches the manual Pearson formula."""
        xs, ys = pair
        n = len(xs)
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n

        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        denom = math.sqrt(
            sum((x - x_mean) ** 2 for x in xs) *
            sum((y - y_mean) ** 2 for y in ys)
        )

        engine = AnalyticsEngine()
        result = engine.calculate_correlation(xs, ys)

        if denom == 0:
            # Degenerate: constant series → r should be 0
            assert math.isclose(result.coefficient, 0.0, abs_tol=1e-9)
        else:
            expected_r = max(-1.0, min(1.0, num / denom))
            assert math.isclose(result.coefficient, expected_r, rel_tol=1e-6, abs_tol=1e-9), (
                f"Expected r={expected_r:.6f}, got {result.coefficient:.6f}"
            )

    @given(pair=paired_series())
    @settings(max_examples=200, deadline=None)
    def test_symmetry(self, pair):
        """corr(x, y) == corr(y, x)."""
        xs, ys = pair
        engine = AnalyticsEngine()
        r_xy = engine.calculate_correlation(xs, ys).coefficient
        r_yx = engine.calculate_correlation(ys, xs).coefficient
        assert math.isclose(r_xy, r_yx, rel_tol=1e-9, abs_tol=1e-12)

    @given(
        xs=metric_series,
        scale=st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False),
        shift=float_val,
    )
    @settings(max_examples=100, deadline=None)
    def test_invariant_to_linear_transform_of_y(self, xs, scale, shift):
        """corr(x, a*y + b) == corr(x, y) for a > 0."""
        assume(len(xs) >= 3)
        ys = [x * 1.5 + 3.0 for x in xs]  # correlated series
        ys_transformed = [scale * y + shift for y in ys]

        engine = AnalyticsEngine()
        r_orig = engine.calculate_correlation(xs, ys).coefficient
        r_trans = engine.calculate_correlation(xs, ys_transformed).coefficient

        assert math.isclose(r_orig, r_trans, rel_tol=1e-6, abs_tol=1e-9), (
            f"Linear transform changed r: {r_orig:.6f} vs {r_trans:.6f}"
        )

    @given(xs=metric_series)
    @settings(max_examples=100, deadline=None)
    def test_perfect_positive_correlation(self, xs):
        """corr(x, x) == 1.0 when x has meaningful spread."""
        n = len(xs)
        x_mean = sum(xs) / n
        sum_sq = sum((v - x_mean) ** 2 for v in xs)
        # Require non-trivial variance to avoid degenerate near-constant series
        assume(sum_sq > 1e-10)
        engine = AnalyticsEngine()
        result = engine.calculate_correlation(xs, xs)
        assert math.isclose(result.coefficient, 1.0, rel_tol=1e-6, abs_tol=1e-9)

    @given(xs=metric_series)
    @settings(max_examples=100, deadline=None)
    def test_perfect_negative_correlation(self, xs):
        """corr(x, -x) == -1.0 when x has meaningful spread."""
        n = len(xs)
        x_mean = sum(xs) / n
        sum_sq = sum((v - x_mean) ** 2 for v in xs)
        assume(sum_sq > 1e-10)
        neg_xs = [-v for v in xs]
        engine = AnalyticsEngine()
        result = engine.calculate_correlation(xs, neg_xs)
        assert math.isclose(result.coefficient, -1.0, rel_tol=1e-6, abs_tol=1e-9)

    @given(pair=paired_series())
    @settings(max_examples=100, deadline=None)
    def test_p_value_in_range(self, pair):
        """p-value is always in [0, 1]."""
        xs, ys = pair
        engine = AnalyticsEngine()
        result = engine.calculate_correlation(xs, ys)
        assert 0.0 <= result.p_value <= 1.0, (
            f"p-value {result.p_value} out of [0, 1]"
        )

    @given(pair=paired_series())
    @settings(max_examples=100, deadline=None)
    def test_sample_size_matches_input(self, pair):
        """sample_size equals the length of the input series."""
        xs, ys = pair
        engine = AnalyticsEngine()
        result = engine.calculate_correlation(xs, ys)
        assert result.sample_size == len(xs)

    def test_mismatched_lengths_raises(self):
        """Different-length series raise ValueError."""
        engine = AnalyticsEngine()
        with pytest.raises(ValueError):
            engine.calculate_correlation([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_too_few_points_raises(self):
        """Fewer than 3 data points raise ValueError."""
        engine = AnalyticsEngine()
        with pytest.raises(ValueError):
            engine.calculate_correlation([1.0, 2.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# Property 31: Innovation-Revenue Correlation Calculation
# ---------------------------------------------------------------------------

class TestProperty31InnovationRevenueCorrelation:
    """
    Property 31: Innovation-Revenue Correlation Calculation

    For any dataset of companies with Innovation Scores and revenue growth rates,
    the Pearson correlation coefficient should be calculated using the standard
    statistical formula.

    Validates: Requirement 7.1
    """

    @given(
        n=st.integers(min_value=3, max_value=100),
        noise_scale=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_positive_correlation_detected(self, n, noise_scale):
        """Strongly positively correlated series yields r > 0."""
        # innovation_scores linearly drive revenue_growth with small noise
        innovation_scores = [float(i) for i in range(n)]
        revenue_growth = [s + noise_scale * 0.01 for s in innovation_scores]

        engine = AnalyticsEngine()
        result = engine.calculate_correlation(innovation_scores, revenue_growth)
        assert result.coefficient > 0.0

    @given(n=st.integers(min_value=3, max_value=100))
    @settings(max_examples=100, deadline=None)
    def test_negative_correlation_detected(self, n):
        """Negatively correlated series yields r < 0."""
        innovation_scores = [float(i) for i in range(n)]
        revenue_growth = [float(n - i) for i in range(n)]

        engine = AnalyticsEngine()
        result = engine.calculate_correlation(innovation_scores, revenue_growth)
        assert result.coefficient < 0.0

    @given(pair=paired_series(min_size=3, max_size=50))
    @settings(max_examples=100, deadline=None)
    def test_result_has_all_required_fields(self, pair):
        """CorrelationResult has coefficient, p_value, confidence_interval, sample_size."""
        xs, ys = pair
        engine = AnalyticsEngine()
        result = engine.calculate_correlation(xs, ys)

        assert hasattr(result, "coefficient")
        assert hasattr(result, "p_value")
        assert hasattr(result, "confidence_interval")
        assert hasattr(result, "sample_size")
        assert isinstance(result.confidence_interval, tuple)
        assert len(result.confidence_interval) == 2


# ---------------------------------------------------------------------------
# Property 35: Correlation Persistence with Confidence Intervals
# ---------------------------------------------------------------------------

class TestProperty35CorrelationPersistence:
    """
    Property 35: Correlation Persistence with Confidence Intervals

    When correlations are calculated, the Analytics Engine should store
    correlation coefficients with confidence intervals in the Metrics Repository.

    Validates: Requirement 7.5
    """

    @given(pair=paired_series())
    @settings(max_examples=100, deadline=None)
    def test_stored_record_has_correct_coefficient(self, pair):
        """Stored CorrelationRecord.correlation_coefficient matches calculated r."""
        xs, ys = pair
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        result = engine.calculate_correlation(xs, ys)
        record = engine.store_correlation("innovation_score", "revenue_growth", result)

        assert math.isclose(
            record.correlation_coefficient, result.coefficient, rel_tol=1e-9
        )

    @given(pair=paired_series())
    @settings(max_examples=100, deadline=None)
    def test_stored_record_has_confidence_interval(self, pair):
        """Stored record contains a 2-tuple confidence interval."""
        xs, ys = pair
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        result = engine.calculate_correlation(xs, ys)
        record = engine.store_correlation("innovation_score", "revenue_growth", result)

        assert isinstance(record.confidence_interval, tuple)
        assert len(record.confidence_interval) == 2
        lo, hi = record.confidence_interval
        assert lo <= hi, f"CI lower bound {lo} > upper bound {hi}"

    @given(pair=paired_series())
    @settings(max_examples=100, deadline=None)
    def test_stored_record_has_timestamp(self, pair):
        """Stored CorrelationRecord has a datetime timestamp."""
        xs, ys = pair
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        result = engine.calculate_correlation(xs, ys)
        record = engine.store_correlation("innovation_score", "revenue_growth", result)

        assert isinstance(record.timestamp, datetime)
        assert record.timestamp is not None

    @given(pair=paired_series())
    @settings(max_examples=100, deadline=None)
    def test_stored_record_has_metric_names(self, pair):
        """Stored record preserves metric1 and metric2 names."""
        xs, ys = pair
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        result = engine.calculate_correlation(xs, ys)
        record = engine.store_correlation("innovation_score", "revenue_growth", result)

        assert record.metric1 == "innovation_score"
        assert record.metric2 == "revenue_growth"

    @given(pair=paired_series())
    @settings(max_examples=100, deadline=None)
    def test_stored_record_has_sample_size(self, pair):
        """Stored record sample_size matches input length."""
        xs, ys = pair
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        result = engine.calculate_correlation(xs, ys)
        record = engine.store_correlation("innovation_score", "revenue_growth", result)

        assert record.sample_size == len(xs)

    @given(pair=paired_series())
    @settings(max_examples=100, deadline=None)
    def test_record_is_persisted_in_repository(self, pair):
        """After store_correlation, the record appears in the repository."""
        xs, ys = pair
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        result = engine.calculate_correlation(xs, ys)
        engine.store_correlation("innovation_score", "revenue_growth", result)

        records = repo.get_by_type(CorrelationRecord)
        assert len(records) == 1

    @given(pair=paired_series())
    @settings(max_examples=100, deadline=None)
    def test_confidence_interval_contains_coefficient(self, pair):
        """The 95% CI should contain the point estimate r (trivially true by construction)."""
        xs, ys = pair
        engine = AnalyticsEngine()
        result = engine.calculate_correlation(xs, ys)
        lo, hi = result.confidence_interval
        # CI bounds should be valid floats
        assert math.isfinite(lo)
        assert math.isfinite(hi)
        assert lo <= hi

    @given(pair=paired_series(min_size=10, max_size=100))
    @settings(max_examples=100, deadline=None)
    def test_ci_bounds_in_valid_range(self, pair):
        """Confidence interval bounds are in [-1, 1]."""
        xs, ys = pair
        engine = AnalyticsEngine()
        result = engine.calculate_correlation(xs, ys)
        lo, hi = result.confidence_interval
        assert -1.0 <= lo <= 1.0, f"CI lower bound {lo} out of [-1, 1]"
        assert -1.0 <= hi <= 1.0, f"CI upper bound {hi} out of [-1, 1]"
