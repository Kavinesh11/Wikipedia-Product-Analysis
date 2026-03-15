"""Property-based tests for business outcome correlation analysis.

Covers:
- Property 32: Centrality-M&A Correlation Calculation  (Req 7.2)
- Property 33: Top Quartile Revenue Growth Aggregation  (Req 7.3)
- Property 34: Quartile Revenue Growth Comparison       (Req 7.4)
"""

import math
import pytest
from hypothesis import given, strategies as st, settings, assume

from fortune500_kg.analytics_engine import AnalyticsEngine, MetricsRepository


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

float_val = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def paired_float_series(draw, min_size=3, max_size=100):
    """Two float lists of the same length."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    xs = [draw(float_val) for _ in range(n)]
    ys = [draw(float_val) for _ in range(n)]
    return xs, ys


@st.composite
def company_scores_and_growth(draw, min_companies=4, max_companies=100):
    """Dict of company_id -> innovation_score and company_id -> revenue_growth."""
    n = draw(st.integers(min_value=min_companies, max_value=max_companies))
    company_ids = [f"company_{i}" for i in range(n)]
    scores = {cid: draw(float_val) for cid in company_ids}
    growth = {cid: draw(float_val) for cid in company_ids}
    return company_ids, scores, growth


# ---------------------------------------------------------------------------
# Property 32: Centrality-M&A Correlation Calculation
# ---------------------------------------------------------------------------

class TestProperty32CentralityMACorrelation:
    """
    Property 32: Centrality-M&A Correlation Calculation

    For any dataset of companies with Ecosystem Centrality and M&A activity
    frequency, the correlation coefficient should be calculated correctly.

    **Validates: Requirements 7.2**
    """

    @given(pair=paired_float_series())
    @settings(max_examples=100, deadline=None)
    def test_coefficient_in_valid_range(self, pair):
        """Correlation coefficient is always in [-1, 1]."""
        centrality, ma_activity = pair
        engine = AnalyticsEngine()
        result = engine.calculate_centrality_ma_correlation(centrality, ma_activity)
        assert -1.0 <= result.coefficient <= 1.0

    @given(pair=paired_float_series())
    @settings(max_examples=100, deadline=None)
    def test_coefficient_matches_pearson_formula(self, pair):
        """Centrality-M&A r matches the manual Pearson formula."""
        centrality, ma_activity = pair
        n = len(centrality)
        x_mean = sum(centrality) / n
        y_mean = sum(ma_activity) / n

        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(centrality, ma_activity))
        denom = math.sqrt(
            sum((x - x_mean) ** 2 for x in centrality) *
            sum((y - y_mean) ** 2 for y in ma_activity)
        )

        engine = AnalyticsEngine()
        result = engine.calculate_centrality_ma_correlation(centrality, ma_activity)

        if denom == 0:
            assert math.isclose(result.coefficient, 0.0, abs_tol=1e-9)
        else:
            expected_r = max(-1.0, min(1.0, num / denom))
            assert math.isclose(result.coefficient, expected_r, rel_tol=1e-6, abs_tol=1e-9)

    @given(pair=paired_float_series())
    @settings(max_examples=100, deadline=None)
    def test_result_has_required_fields(self, pair):
        """CorrelationResult has all required fields."""
        centrality, ma_activity = pair
        engine = AnalyticsEngine()
        result = engine.calculate_centrality_ma_correlation(centrality, ma_activity)

        assert hasattr(result, "coefficient")
        assert hasattr(result, "p_value")
        assert hasattr(result, "confidence_interval")
        assert hasattr(result, "sample_size")
        assert result.sample_size == len(centrality)

    @given(pair=paired_float_series())
    @settings(max_examples=100, deadline=None)
    def test_p_value_in_valid_range(self, pair):
        """p-value is always in [0, 1]."""
        centrality, ma_activity = pair
        engine = AnalyticsEngine()
        result = engine.calculate_centrality_ma_correlation(centrality, ma_activity)
        assert 0.0 <= result.p_value <= 1.0

    @given(pair=paired_float_series())
    @settings(max_examples=100, deadline=None)
    def test_symmetry(self, pair):
        """corr(centrality, ma) == corr(ma, centrality)."""
        centrality, ma_activity = pair
        engine = AnalyticsEngine()
        r_fwd = engine.calculate_centrality_ma_correlation(centrality, ma_activity).coefficient
        r_rev = engine.calculate_centrality_ma_correlation(ma_activity, centrality).coefficient
        assert math.isclose(r_fwd, r_rev, rel_tol=1e-9, abs_tol=1e-12)

    def test_mismatched_lengths_raises(self):
        """Different-length series raise ValueError."""
        engine = AnalyticsEngine()
        with pytest.raises(ValueError):
            engine.calculate_centrality_ma_correlation([0.1, 0.2, 0.3], [0.1, 0.2])

    def test_too_few_points_raises(self):
        """Fewer than 3 data points raise ValueError."""
        engine = AnalyticsEngine()
        with pytest.raises(ValueError):
            engine.calculate_centrality_ma_correlation([0.1, 0.2], [0.1, 0.2])


# ---------------------------------------------------------------------------
# Property 33: Top Quartile Revenue Growth Aggregation
# ---------------------------------------------------------------------------

class TestProperty33TopQuartileRevenueGrowth:
    """
    Property 33: Top Quartile Revenue Growth Aggregation

    For any set of companies ranked by Innovation Score, the top quartile
    average revenue growth should equal the arithmetic mean of the top 25%
    of companies by Innovation Score.

    **Validates: Requirements 7.3**
    """

    @given(data=company_scores_and_growth())
    @settings(max_examples=100, deadline=None)
    def test_top_quartile_size_is_floor_n_over_4(self, data):
        """Top quartile contains exactly ⌊N/4⌋ companies (min 1)."""
        company_ids, scores, _ = data
        engine = AnalyticsEngine()
        top_ids = engine.get_top_quartile_companies(scores)
        n = len(company_ids)
        expected_size = max(1, n // 4)
        assert len(top_ids) == expected_size

    @given(data=company_scores_and_growth())
    @settings(max_examples=100, deadline=None)
    def test_top_quartile_has_highest_scores(self, data):
        """Every company in the top quartile has a score >= every company outside it."""
        company_ids, scores, _ = data
        engine = AnalyticsEngine()
        top_ids = set(engine.get_top_quartile_companies(scores))
        non_top_ids = set(company_ids) - top_ids

        if not non_top_ids:
            return  # All companies in quartile, nothing to compare

        min_top_score = min(scores[cid] for cid in top_ids)
        max_non_top_score = max(scores[cid] for cid in non_top_ids)
        assert min_top_score >= max_non_top_score

    @given(data=company_scores_and_growth())
    @settings(max_examples=100, deadline=None)
    def test_top_quartile_avg_equals_arithmetic_mean(self, data):
        """Average revenue growth for top quartile equals arithmetic mean of their growth rates."""
        _, scores, growth = data
        engine = AnalyticsEngine()
        top_ids = engine.get_top_quartile_companies(scores)

        expected_avg = sum(growth[cid] for cid in top_ids) / len(top_ids)
        actual_avg = engine.calculate_quartile_revenue_growth(top_ids, growth)

        assert math.isclose(actual_avg, expected_avg, rel_tol=1e-9, abs_tol=1e-12)

    @given(data=company_scores_and_growth())
    @settings(max_examples=100, deadline=None)
    def test_top_quartile_ids_are_subset_of_all_companies(self, data):
        """Top quartile company IDs are a subset of all company IDs."""
        company_ids, scores, _ = data
        engine = AnalyticsEngine()
        top_ids = engine.get_top_quartile_companies(scores)
        assert set(top_ids).issubset(set(company_ids))

    def test_empty_scores_returns_empty(self):
        """Empty input returns empty list."""
        engine = AnalyticsEngine()
        assert engine.get_top_quartile_companies({}) == []

    def test_single_company_returns_that_company(self):
        """Single company is always in the top quartile."""
        engine = AnalyticsEngine()
        result = engine.get_top_quartile_companies({"c1": 5.0})
        assert result == ["c1"]

    def test_calculate_quartile_growth_empty_returns_zero(self):
        """No companies in quartile returns 0.0."""
        engine = AnalyticsEngine()
        assert engine.calculate_quartile_revenue_growth([], {"c1": 0.1}) == 0.0


# ---------------------------------------------------------------------------
# Property 34: Quartile Revenue Growth Comparison
# ---------------------------------------------------------------------------

class TestProperty34QuartileRevenueGrowthComparison:
    """
    Property 34: Quartile Revenue Growth Comparison

    For any set of companies divided into high-score and low-score quartiles,
    the comparison should calculate the difference between the two quartile
    average growth rates.

    **Validates: Requirements 7.4**
    """

    @given(data=company_scores_and_growth())
    @settings(max_examples=100, deadline=None)
    def test_comparison_result_has_required_keys(self, data):
        """compare_quartile_revenue_growth returns dict with required keys."""
        _, scores, growth = data
        engine = AnalyticsEngine()
        result = engine.compare_quartile_revenue_growth(scores, growth)

        assert "top_quartile_avg" in result
        assert "bottom_quartile_avg" in result
        assert "difference" in result

    @given(data=company_scores_and_growth())
    @settings(max_examples=100, deadline=None)
    def test_difference_equals_top_minus_bottom(self, data):
        """difference == top_quartile_avg - bottom_quartile_avg."""
        _, scores, growth = data
        engine = AnalyticsEngine()
        result = engine.compare_quartile_revenue_growth(scores, growth)

        expected_diff = result["top_quartile_avg"] - result["bottom_quartile_avg"]
        assert math.isclose(result["difference"], expected_diff, rel_tol=1e-9, abs_tol=1e-12)

    @given(data=company_scores_and_growth())
    @settings(max_examples=100, deadline=None)
    def test_top_avg_matches_manual_calculation(self, data):
        """top_quartile_avg matches manually computed mean of top quartile growth."""
        _, scores, growth = data
        engine = AnalyticsEngine()
        top_ids = engine.get_top_quartile_companies(scores)
        expected_top_avg = sum(growth[cid] for cid in top_ids) / len(top_ids)

        result = engine.compare_quartile_revenue_growth(scores, growth)
        assert math.isclose(result["top_quartile_avg"], expected_top_avg, rel_tol=1e-9, abs_tol=1e-12)

    @given(data=company_scores_and_growth())
    @settings(max_examples=100, deadline=None)
    def test_bottom_avg_matches_manual_calculation(self, data):
        """bottom_quartile_avg matches manually computed mean of bottom quartile growth."""
        _, scores, growth = data
        engine = AnalyticsEngine()
        bottom_ids = engine.get_bottom_quartile_companies(scores)
        expected_bottom_avg = sum(growth[cid] for cid in bottom_ids) / len(bottom_ids)

        result = engine.compare_quartile_revenue_growth(scores, growth)
        assert math.isclose(result["bottom_quartile_avg"], expected_bottom_avg, rel_tol=1e-9, abs_tol=1e-12)

    @given(data=company_scores_and_growth())
    @settings(max_examples=100, deadline=None)
    def test_top_and_bottom_quartiles_are_disjoint(self, data):
        """Top and bottom quartile company sets do not overlap when all scores are distinct."""
        company_ids, scores, _ = data
        assume(len(company_ids) >= 8)
        # Require all scores to be distinct so top/bottom quartile selection is unambiguous
        assume(len(set(scores.values())) == len(scores))
        engine = AnalyticsEngine()
        top_ids = set(engine.get_top_quartile_companies(scores))
        bottom_ids = set(engine.get_bottom_quartile_companies(scores))
        assert top_ids.isdisjoint(bottom_ids)

    def test_when_top_quartile_has_higher_growth_difference_is_positive(self):
        """When top quartile companies have higher growth, difference > 0."""
        scores = {f"c{i}": float(i) for i in range(8)}
        # Top quartile (c6, c7) have growth 60, 70; bottom (c0, c1) have 0, 10
        growth = {f"c{i}": float(i * 10) for i in range(8)}

        engine = AnalyticsEngine()
        result = engine.compare_quartile_revenue_growth(scores, growth)
        assert result["difference"] > 0

    def test_when_top_and_bottom_have_equal_growth_difference_is_zero(self):
        """When all companies have the same growth rate, difference == 0."""
        scores = {f"c{i}": float(i) for i in range(8)}
        growth = {f"c{i}": 5.0 for i in range(8)}

        engine = AnalyticsEngine()
        result = engine.compare_quartile_revenue_growth(scores, growth)
        assert math.isclose(result["difference"], 0.0, abs_tol=1e-9)
