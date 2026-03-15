"""Property-based tests for InsightGenerator.

Covers:
- Property 41: Underperformer Identification Correctness  (Req 9.1)

**Validates: Requirements 9.1**
"""

from datetime import datetime

from hypothesis import given, settings, strategies as st

from fortune500_kg.analytics_engine import MetricsRepository
from fortune500_kg.data_models import Company, InnovationScoreRecord
from fortune500_kg.insight_generator import InsightGenerator


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

SECTOR = "TestSector"


@st.composite
def companies_with_scores(draw, min_size=1, max_size=30):
    """Generate a list of (Company, innovation_score) pairs in the same sector.

    Scores are floats in [0, 10] to match the normalised Innovation Score range.
    Company IDs are unique.
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    companies = []
    for i in range(n):
        score = draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        company = Company(
            id=f"C{i:04d}",
            name=f"Corp {i}",
            sector=SECTOR,
            revenue_rank=i + 1,
            employee_count=1000,
        )
        companies.append((company, score))
    return companies


def _build_insight_generator(company_score_pairs):
    """Build a MetricsRepository and InsightGenerator from (Company, score) pairs."""
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


# ---------------------------------------------------------------------------
# Property 41: Underperformer Identification Correctness
# ---------------------------------------------------------------------------

class TestProperty41UnderperformerIdentificationCorrectness:
    """
    Property 41: Underperformer Identification Correctness

    For any sector with average Innovation Score S_avg, identified underperforming
    companies should have Innovation Score < S_avg.

    **Validates: Requirements 9.1**
    """

    @given(pairs=companies_with_scores(min_size=1, max_size=30))
    @settings(max_examples=100, deadline=None)
    def test_every_returned_company_is_strictly_below_average(self, pairs):
        """Every returned underperformer has score strictly below the sector average."""
        gen = _build_insight_generator(pairs)
        results = gen.identify_underperformers(SECTOR)

        if not results:
            return  # nothing to check

        sector_average = results[0].sector_average
        for r in results:
            assert r.innovation_score < sector_average, (
                f"Company {r.company.id} score {r.innovation_score} is not "
                f"strictly below sector average {sector_average}"
            )

    @given(pairs=companies_with_scores(min_size=1, max_size=30))
    @settings(max_examples=100, deadline=None)
    def test_no_company_at_or_above_average_is_returned(self, pairs):
        """Companies with score >= sector average must NOT appear in results."""
        gen = _build_insight_generator(pairs)
        results = gen.identify_underperformers(SECTOR)

        if not pairs:
            return

        scores = {company.id: score for company, score in pairs}
        sector_average = sum(scores.values()) / len(scores)
        returned_ids = {r.company.id for r in results}

        for company, score in pairs:
            if score >= sector_average:
                assert company.id not in returned_ids, (
                    f"Company {company.id} with score {score} >= average "
                    f"{sector_average} should not be in results"
                )

    @given(pairs=companies_with_scores(min_size=1, max_size=30))
    @settings(max_examples=100, deadline=None)
    def test_gap_equals_sector_average_minus_company_score(self, pairs):
        """Gap value equals sector_average - company_score for every result."""
        gen = _build_insight_generator(pairs)
        results = gen.identify_underperformers(SECTOR)

        for r in results:
            expected_gap = r.sector_average - r.innovation_score
            assert abs(r.gap - expected_gap) < 1e-9, (
                f"Gap mismatch for {r.company.id}: expected {expected_gap}, got {r.gap}"
            )

    @given(pairs=companies_with_scores(min_size=1, max_size=30))
    @settings(max_examples=100, deadline=None)
    def test_results_sorted_by_gap_descending(self, pairs):
        """Results are sorted by gap descending (worst underperformers first)."""
        gen = _build_insight_generator(pairs)
        results = gen.identify_underperformers(SECTOR)

        gaps = [r.gap for r in results]
        assert gaps == sorted(gaps, reverse=True), (
            f"Results not sorted by gap descending: {gaps}"
        )

    @given(pairs=companies_with_scores(min_size=1, max_size=30))
    @settings(max_examples=100, deadline=None)
    def test_all_underperformers_are_returned(self, pairs):
        """Every company strictly below the sector average appears in results."""
        gen = _build_insight_generator(pairs)
        results = gen.identify_underperformers(SECTOR)

        scores = {company.id: score for company, score in pairs}
        sector_average = sum(scores.values()) / len(scores)
        returned_ids = {r.company.id for r in results}

        for company, score in pairs:
            if score < sector_average:
                assert company.id in returned_ids, (
                    f"Company {company.id} with score {score} < average "
                    f"{sector_average} should be in results but is missing"
                )


# ---------------------------------------------------------------------------
# Helpers for Properties 42 & 44
# ---------------------------------------------------------------------------

@st.composite
def digital_maturity_records_bottom(draw, min_size=1, max_size=20):
    """Generate a list of (company_id, DigitalMaturityRecord) for bottom-quartile companies.

    Each record has quartile='bottom' and a positive sector_avg > metric_value
    so that the gap is always positive and confidence / talent improvement are
    well-defined.
    """
    from fortune500_kg.data_models import DigitalMaturityRecord

    n = draw(st.integers(min_value=min_size, max_value=max_size))
    records = []
    for i in range(n):
        sector_avg = draw(st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False))
        # dmi is strictly below sector_avg so gap > 0
        dmi = draw(st.floats(min_value=0.0, max_value=sector_avg - 0.01, allow_nan=False, allow_infinity=False))
        company_id = f"BQ{i:04d}"
        record = DigitalMaturityRecord(
            company_id=company_id,
            metric_name="digital_maturity_index",
            metric_value=dmi,
            timestamp=datetime.now(),
            sector="TestSector",
            sector_avg=sector_avg,
            quartile="bottom",
        )
        records.append((company_id, record))
    return records


def _build_insight_generator_dmi(bottom_records):
    """Build InsightGenerator populated with DigitalMaturityRecord entries."""
    from fortune500_kg.data_models import DigitalMaturityRecord

    repo = MetricsRepository()
    companies = []
    for company_id, record in bottom_records:
        company = Company(
            id=company_id,
            name=f"Corp {company_id}",
            sector="TestSector",
            revenue_rank=1,
            employee_count=1000,
        )
        companies.append(company)
        repo.save(record)
    return InsightGenerator(metrics_repo=repo, companies=companies)


# ---------------------------------------------------------------------------
# Property 42: Bottom Quartile Investment Recommendation Coverage
# ---------------------------------------------------------------------------

class TestProperty42BottomQuartileInvestmentRecommendationCoverage:
    """
    Property 42: Bottom Quartile Investment Recommendation Coverage

    For any set of companies with DigitalMaturityRecord entries in the 'bottom'
    quartile, recommend_investments() should return exactly one Recommendation
    per bottom-quartile company (no more, no less).

    **Validates: Requirements 9.2**
    """

    @given(records=digital_maturity_records_bottom(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_one_recommendation_per_bottom_quartile_company(self, records):
        """Exactly one Recommendation is returned for each bottom-quartile company."""
        gen = _build_insight_generator_dmi(records)
        recommendations = gen.recommend_investments(quartile="bottom")

        expected_ids = {company_id for company_id, _ in records}
        assert len(recommendations) == len(expected_ids), (
            f"Expected {len(expected_ids)} recommendations, got {len(recommendations)}"
        )

    @given(records=digital_maturity_records_bottom(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_every_bottom_quartile_company_has_recommendation(self, records):
        """Every bottom-quartile company appears in at least one recommendation."""
        gen = _build_insight_generator_dmi(records)
        recommendations = gen.recommend_investments(quartile="bottom")

        expected_ids = {company_id for company_id, _ in records}
        covered_ids = {cid for rec in recommendations for cid in rec.target_companies}

        for company_id in expected_ids:
            assert company_id in covered_ids, (
                f"Bottom-quartile company {company_id} has no recommendation"
            )

    @given(records=digital_maturity_records_bottom(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_no_extra_recommendations_beyond_bottom_quartile(self, records):
        """No recommendation targets a company not in the bottom quartile."""
        gen = _build_insight_generator_dmi(records)
        recommendations = gen.recommend_investments(quartile="bottom")

        expected_ids = {company_id for company_id, _ in records}
        for rec in recommendations:
            for cid in rec.target_companies:
                assert cid in expected_ids, (
                    f"Recommendation targets {cid} which is not a bottom-quartile company"
                )

    @given(records=digital_maturity_records_bottom(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_recommendations_are_investment_category(self, records):
        """All returned recommendations have category='investment'."""
        gen = _build_insight_generator_dmi(records)
        recommendations = gen.recommend_investments(quartile="bottom")

        for rec in recommendations:
            assert rec.category == "investment", (
                f"Expected category 'investment', got '{rec.category}'"
            )

    def test_empty_repository_returns_no_recommendations(self):
        """When no bottom-quartile records exist, returns empty list."""
        gen = InsightGenerator(metrics_repo=MetricsRepository(), companies=[])
        recommendations = gen.recommend_investments(quartile="bottom")
        assert recommendations == []


# ---------------------------------------------------------------------------
# Property 44: Talent Attraction Quantification Presence
# ---------------------------------------------------------------------------

class TestProperty44TalentAttractionQuantificationPresence:
    """
    Property 44: Talent Attraction Quantification Presence

    For any Recommendation returned by recommend_investments(), the
    expected_outcome field must contain a quantified talent attraction
    improvement (i.e., a percentage value and the phrase "talent attraction").

    **Validates: Requirements 9.4**
    """

    @given(records=digital_maturity_records_bottom(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_expected_outcome_contains_talent_attraction_phrase(self, records):
        """expected_outcome contains the phrase 'talent attraction'."""
        gen = _build_insight_generator_dmi(records)
        recommendations = gen.recommend_investments(quartile="bottom")

        for rec in recommendations:
            assert "talent attraction" in rec.expected_outcome.lower(), (
                f"expected_outcome missing 'talent attraction': {rec.expected_outcome!r}"
            )

    @given(records=digital_maturity_records_bottom(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_expected_outcome_contains_percentage_value(self, records):
        """expected_outcome contains a percentage value (a number followed by '%')."""
        import re
        gen = _build_insight_generator_dmi(records)
        recommendations = gen.recommend_investments(quartile="bottom")

        percentage_pattern = re.compile(r'\d+(\.\d+)?%')
        for rec in recommendations:
            assert percentage_pattern.search(rec.expected_outcome), (
                f"expected_outcome missing a percentage value: {rec.expected_outcome!r}"
            )

    @given(records=digital_maturity_records_bottom(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_expected_outcome_is_non_empty_string(self, records):
        """expected_outcome is a non-empty string for every recommendation."""
        gen = _build_insight_generator_dmi(records)
        recommendations = gen.recommend_investments(quartile="bottom")

        for rec in recommendations:
            assert isinstance(rec.expected_outcome, str) and rec.expected_outcome.strip(), (
                f"expected_outcome is empty or not a string: {rec.expected_outcome!r}"
            )


# ---------------------------------------------------------------------------
# Strategies for Property 43
# ---------------------------------------------------------------------------

@st.composite
def ecosystem_centrality_dataset(draw, min_size=2, max_size=20):
    """Generate a list of (Company, EcosystemCentralityRecord) pairs in one sector.

    Ensures at least 2 companies so that a meaningful median can be computed.
    betweenness_centrality values are floats in [0, 1].
    revenue_rank values are unique positive integers.
    """
    from fortune500_kg.data_models import EcosystemCentralityRecord

    n = draw(st.integers(min_value=min_size, max_value=max_size))
    # Generate unique revenue ranks
    revenue_ranks = draw(
        st.lists(
            st.integers(min_value=1, max_value=500),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    centralities = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )

    pairs = []
    for i in range(n):
        company = Company(
            id=f"AT{i:04d}",
            name=f"AcqCorp {i}",
            sector=SECTOR,
            revenue_rank=revenue_ranks[i],
            employee_count=1000,
        )
        record = EcosystemCentralityRecord(
            company_id=company.id,
            metric_name="ecosystem_centrality",
            metric_value=centralities[i],
            timestamp=datetime.now(),
            betweenness_centrality=centralities[i],
            pagerank_score=0.0,
            sector_avg_centrality=0.0,
        )
        pairs.append((company, record))
    return pairs


def _build_insight_generator_centrality(pairs):
    """Build InsightGenerator populated with EcosystemCentralityRecord entries."""
    from fortune500_kg.data_models import EcosystemCentralityRecord

    repo = MetricsRepository()
    companies = []
    for company, record in pairs:
        companies.append(company)
        repo.save(record)
    return InsightGenerator(metrics_repo=repo, companies=companies)


def _compute_median(values):
    """Compute median of a list of floats."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


# ---------------------------------------------------------------------------
# Property 43: Acquisition Target Multi-Criteria Filtering
# ---------------------------------------------------------------------------

class TestProperty43AcquisitionTargetMultiCriteriaFiltering:
    """
    Property 43: Acquisition Target Multi-Criteria Filtering

    For any set of companies with EcosystemCentralityRecord entries,
    identify_acquisition_targets() should return only companies that satisfy
    BOTH criteria:
      1. betweenness_centrality > sector median centrality
      2. revenue_rank > sector median revenue_rank

    **Validates: Requirements 9.3**
    """

    @given(pairs=ecosystem_centrality_dataset(min_size=2, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_every_target_has_centrality_above_sector_median(self, pairs):
        """Every returned target has betweenness_centrality above the sector median."""
        gen = _build_insight_generator_centrality(pairs)
        targets = gen.identify_acquisition_targets()

        if not targets:
            return

        centralities = [record.betweenness_centrality for _, record in pairs]
        median_centrality = _compute_median(centralities)

        for target in targets:
            actual_centrality = target.metrics["ecosystem_centrality"]
            assert actual_centrality > median_centrality, (
                f"Target {target.company_id} centrality {actual_centrality} is not "
                f"above sector median {median_centrality}"
            )

    @given(pairs=ecosystem_centrality_dataset(min_size=2, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_every_target_has_revenue_rank_above_sector_median(self, pairs):
        """Every returned target has revenue_rank above the sector median."""
        gen = _build_insight_generator_centrality(pairs)
        targets = gen.identify_acquisition_targets()

        if not targets:
            return

        revenue_ranks = [float(company.revenue_rank) for company, _ in pairs]
        median_revenue_rank = _compute_median(revenue_ranks)

        # Build a lookup from company_id to revenue_rank
        rank_map = {company.id: company.revenue_rank for company, _ in pairs}

        for target in targets:
            actual_rank = target.metrics["revenue_rank"]
            assert actual_rank > median_revenue_rank, (
                f"Target {target.company_id} revenue_rank {actual_rank} is not "
                f"above sector median {median_revenue_rank}"
            )

    @given(pairs=ecosystem_centrality_dataset(min_size=2, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_no_company_failing_either_criterion_appears_in_results(self, pairs):
        """No company that fails centrality OR revenue_rank criterion is in results."""
        gen = _build_insight_generator_centrality(pairs)
        targets = gen.identify_acquisition_targets()

        centralities = [record.betweenness_centrality for _, record in pairs]
        median_centrality = _compute_median(centralities)

        revenue_ranks = [float(company.revenue_rank) for company, _ in pairs]
        median_revenue_rank = _compute_median(revenue_ranks)

        returned_ids = {t.company_id for t in targets}

        for company, record in pairs:
            fails_centrality = record.betweenness_centrality <= median_centrality
            fails_revenue_rank = float(company.revenue_rank) <= median_revenue_rank
            if fails_centrality or fails_revenue_rank:
                assert company.id not in returned_ids, (
                    f"Company {company.id} fails at least one criterion "
                    f"(centrality={record.betweenness_centrality} vs median={median_centrality}, "
                    f"revenue_rank={company.revenue_rank} vs median={median_revenue_rank}) "
                    f"but appears in results"
                )

    @given(pairs=ecosystem_centrality_dataset(min_size=2, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_results_sorted_by_ecosystem_centrality_descending(self, pairs):
        """Results are sorted by ecosystem_centrality descending."""
        gen = _build_insight_generator_centrality(pairs)
        targets = gen.identify_acquisition_targets()

        centrality_values = [t.metrics["ecosystem_centrality"] for t in targets]
        assert centrality_values == sorted(centrality_values, reverse=True), (
            f"Results not sorted by ecosystem_centrality descending: {centrality_values}"
        )

    def test_empty_repository_returns_no_targets(self):
        """When no EcosystemCentralityRecord entries exist, returns empty list."""
        gen = InsightGenerator(metrics_repo=MetricsRepository(), companies=[])
        targets = gen.identify_acquisition_targets()
        assert targets == []


# ---------------------------------------------------------------------------
# Property 45: Recommendation Structure Completeness
# ---------------------------------------------------------------------------

class TestProperty45RecommendationStructureCompleteness:
    """
    Property 45: Recommendation Structure Completeness

    For any Recommendation object returned by recommend_investments(), the object
    must contain:
    - supporting_metrics: non-empty dict with at least one key-value pair
    - confidence_level: float in [0.0, 1.0]
    - expected_outcome: non-empty string

    **Validates: Requirements 9.5**
    """

    @given(records=digital_maturity_records_bottom(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_every_recommendation_has_non_empty_supporting_metrics(self, records):
        """Every recommendation has a non-empty supporting_metrics dict."""
        gen = _build_insight_generator_dmi(records)
        recommendations = gen.recommend_investments(quartile="bottom")

        for rec in recommendations:
            assert isinstance(rec.supporting_metrics, dict), (
                f"supporting_metrics is not a dict: {type(rec.supporting_metrics)}"
            )
            assert len(rec.supporting_metrics) > 0, (
                f"supporting_metrics is empty for recommendation '{rec.title}'"
            )

    @given(records=digital_maturity_records_bottom(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_every_recommendation_has_confidence_level_in_range(self, records):
        """Every recommendation has confidence_level in [0.0, 1.0]."""
        gen = _build_insight_generator_dmi(records)
        recommendations = gen.recommend_investments(quartile="bottom")

        for rec in recommendations:
            assert isinstance(rec.confidence_level, float), (
                f"confidence_level is not a float: {type(rec.confidence_level)}"
            )
            assert 0.0 <= rec.confidence_level <= 1.0, (
                f"confidence_level {rec.confidence_level} is not in [0.0, 1.0] "
                f"for recommendation '{rec.title}'"
            )

    @given(records=digital_maturity_records_bottom(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_every_recommendation_has_non_empty_expected_outcome(self, records):
        """Every recommendation has a non-empty expected_outcome string."""
        gen = _build_insight_generator_dmi(records)
        recommendations = gen.recommend_investments(quartile="bottom")

        for rec in recommendations:
            assert isinstance(rec.expected_outcome, str), (
                f"expected_outcome is not a string: {type(rec.expected_outcome)}"
            )
            assert rec.expected_outcome.strip(), (
                f"expected_outcome is empty or whitespace for recommendation '{rec.title}'"
            )

    @given(records=digital_maturity_records_bottom(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_validate_recommendation_passes_for_all_returned_recommendations(self, records):
        """validate_recommendation() does not raise for any returned recommendation."""
        from fortune500_kg.insight_generator import validate_recommendation

        gen = _build_insight_generator_dmi(records)
        recommendations = gen.recommend_investments(quartile="bottom")

        for rec in recommendations:
            # Should not raise
            validate_recommendation(rec)

    def test_validate_recommendation_raises_for_empty_supporting_metrics(self):
        """validate_recommendation() raises ValueError when supporting_metrics is empty."""
        from fortune500_kg.data_models import Recommendation
        from fortune500_kg.insight_generator import validate_recommendation

        bad_rec = Recommendation(
            priority=1,
            category="investment",
            title="Test",
            description="Test description",
            target_companies=["C001"],
            expected_outcome="Improve talent attraction by 15%",
            confidence_level=0.8,
            supporting_metrics={},  # empty — invalid
        )
        try:
            validate_recommendation(bad_rec)
            assert False, "Expected ValueError was not raised"
        except ValueError:
            pass  # expected

    def test_validate_recommendation_raises_for_confidence_out_of_range(self):
        """validate_recommendation() raises ValueError when confidence_level is out of [0.0, 1.0]."""
        from fortune500_kg.data_models import Recommendation
        from fortune500_kg.insight_generator import validate_recommendation

        for bad_confidence in [-0.1, 1.1, 2.0]:
            bad_rec = Recommendation(
                priority=1,
                category="investment",
                title="Test",
                description="Test description",
                target_companies=["C001"],
                expected_outcome="Improve talent attraction by 15%",
                confidence_level=bad_confidence,
                supporting_metrics={"dmi_gap": 5.0},
            )
            try:
                validate_recommendation(bad_rec)
                assert False, f"Expected ValueError for confidence_level={bad_confidence}"
            except ValueError:
                pass  # expected

    def test_validate_recommendation_raises_for_empty_expected_outcome(self):
        """validate_recommendation() raises ValueError when expected_outcome is empty."""
        from fortune500_kg.data_models import Recommendation
        from fortune500_kg.insight_generator import validate_recommendation

        for bad_outcome in ["", "   "]:
            bad_rec = Recommendation(
                priority=1,
                category="investment",
                title="Test",
                description="Test description",
                target_companies=["C001"],
                expected_outcome=bad_outcome,
                confidence_level=0.8,
                supporting_metrics={"dmi_gap": 5.0},
            )
            try:
                validate_recommendation(bad_rec)
                assert False, f"Expected ValueError for expected_outcome={bad_outcome!r}"
            except ValueError:
                pass  # expected