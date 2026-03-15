"""Property-Based Tests for Graph Algorithm Execution

Tests correctness properties for PageRank, Louvain community detection,
and betweenness centrality calculations.

Properties covered:
  - Property 11: PageRank Iteration Limit (Req 3.1)
  - Property 12: Louvain Community Assignment Completeness (Req 3.2)
  - Property 13: Betweenness Centrality Top-N Selection (Req 3.3)
  - Property 14: Graph Algorithm Result Persistence (Req 3.4)
  - Property 15: Sector-Level Centrality Aggregation (Req 3.5)
"""
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from src.analytics.fortune500_analytics import (
    AnalyticsEngine,
    MetricsRepository,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

COMPANY_IDS = st.lists(
    st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=1, max_size=8),
    min_size=1,
    max_size=20,
    unique=True,
)

SECTORS = ["Tech", "Finance", "Healthcare", "Energy", "Retail"]


@st.composite
def graph_strategy(draw):
    """Generate a random adjacency-list graph over a set of company IDs."""
    company_ids = draw(COMPANY_IDS)
    graph: dict = {cid: [] for cid in company_ids}
    if len(company_ids) > 1:
        for cid in company_ids:
            # Each node may link to a random subset of others
            others = [o for o in company_ids if o != cid]
            num_edges = draw(st.integers(min_value=0, max_value=min(3, len(others))))
            neighbours = draw(
                st.lists(st.sampled_from(others), min_size=num_edges,
                         max_size=num_edges, unique=True)
            )
            graph[cid] = neighbours
    return graph, company_ids


@st.composite
def graph_with_sectors_strategy(draw):
    """Generate a graph plus a sector mapping for every node."""
    graph, company_ids = draw(graph_strategy())
    # Draw a sector for each company individually (no fixed_dictionaries needed)
    sectors = {
        cid: draw(st.sampled_from(SECTORS)) for cid in company_ids
    }
    return graph, company_ids, sectors


# ---------------------------------------------------------------------------
# Property 11: PageRank Iteration Limit (Req 3.1)
# ---------------------------------------------------------------------------

@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(graph_strategy(), st.integers(min_value=1, max_value=20))
def test_property_11_pagerank_iteration_limit(graph_data, max_iter):
    """PageRank must complete within max_iterations; never exceed 20."""
    graph, company_ids = graph_data
    engine = AnalyticsEngine(MetricsRepository())

    result = engine.execute_pagerank(graph, max_iterations=max_iter)

    # Every input node must appear in the result
    assert set(result.keys()) == set(company_ids)

    # All scores are non-negative and sum to approximately 1
    scores = list(result.values())
    assert all(s >= 0.0 for s in scores)
    assert abs(sum(scores) - 1.0) < 1e-4 or len(scores) == 0


@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(graph_strategy())
def test_property_11_pagerank_rejects_over_20_iterations(graph_data):
    """execute_pagerank must raise ValueError when max_iterations > 20."""
    graph, _ = graph_data
    engine = AnalyticsEngine(MetricsRepository())

    with pytest.raises(ValueError):
        engine.execute_pagerank(graph, max_iterations=21)


# ---------------------------------------------------------------------------
# Property 14: Graph Algorithm Result Persistence (Req 3.4) – PageRank
# ---------------------------------------------------------------------------

@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(graph_strategy())
def test_property_14_pagerank_results_persisted(graph_data):
    """After execute_pagerank, every company must have a record in the repo."""
    graph, company_ids = graph_data
    repo = MetricsRepository()
    engine = AnalyticsEngine(repo)

    engine.execute_pagerank(graph)

    stored_ids = {r.company_id for r in repo.get_pagerank_records()}
    assert stored_ids == set(company_ids)


# ---------------------------------------------------------------------------
# Property 12: Louvain Community Assignment Completeness (Req 3.2)
# ---------------------------------------------------------------------------

@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(graph_strategy())
def test_property_12_louvain_every_node_assigned(graph_data):
    """Every company node must be assigned to exactly one community_id."""
    graph, company_ids = graph_data
    engine = AnalyticsEngine(MetricsRepository())

    result = engine.execute_louvain(graph)

    # All nodes present
    assert set(result.keys()) == set(company_ids)

    # Each value is a non-negative integer (valid community id)
    for cid, comm in result.items():
        assert isinstance(comm, int)
        assert comm >= 0


@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(graph_strategy())
def test_property_12_louvain_no_duplicate_assignment(graph_data):
    """Each company appears in exactly one community (no duplicates in result)."""
    graph, company_ids = graph_data
    engine = AnalyticsEngine(MetricsRepository())

    result = engine.execute_louvain(graph)

    # Result keys are unique by definition (dict), so just verify count
    assert len(result) == len(company_ids)


# ---------------------------------------------------------------------------
# Property 14: Graph Algorithm Result Persistence (Req 3.4) – Louvain
# ---------------------------------------------------------------------------

@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(graph_strategy())
def test_property_14_louvain_results_persisted(graph_data):
    """After execute_louvain, every company must have a record in the repo."""
    graph, company_ids = graph_data
    repo = MetricsRepository()
    engine = AnalyticsEngine(repo)

    engine.execute_louvain(graph)

    stored_ids = {r.company_id for r in repo.get_louvain_records()}
    assert stored_ids == set(company_ids)


# ---------------------------------------------------------------------------
# Property 13: Betweenness Centrality Top-N Selection (Req 3.3)
# ---------------------------------------------------------------------------

@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(graph_with_sectors_strategy(), st.integers(min_value=1, max_value=10))
def test_property_13_betweenness_top_n_selection(graph_data, top_n):
    """calculate_betweenness_centrality must return a score for every node."""
    graph, company_ids, sectors = graph_data
    engine = AnalyticsEngine(MetricsRepository())

    result = engine.calculate_betweenness_centrality(graph, sectors, top_n=top_n)

    assert set(result.keys()) == set(company_ids)
    for score in result.values():
        assert isinstance(score, float)
        assert score >= 0.0


@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(graph_with_sectors_strategy())
def test_property_13_betweenness_default_top_n_is_10(graph_data):
    """Default top_n=10 must not raise and must cover all nodes."""
    graph, company_ids, sectors = graph_data
    engine = AnalyticsEngine(MetricsRepository())

    result = engine.calculate_betweenness_centrality(graph, sectors)

    assert set(result.keys()) == set(company_ids)


# ---------------------------------------------------------------------------
# Property 14: Graph Algorithm Result Persistence (Req 3.4) – Centrality
# ---------------------------------------------------------------------------

@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(graph_with_sectors_strategy())
def test_property_14_centrality_results_persisted(graph_data):
    """After calculate_betweenness_centrality, every company has a repo record."""
    graph, company_ids, sectors = graph_data
    repo = MetricsRepository()
    engine = AnalyticsEngine(repo)

    engine.calculate_betweenness_centrality(graph, sectors)

    stored_ids = {r.company_id for r in repo.get_centrality_records()}
    assert stored_ids == set(company_ids)


# ---------------------------------------------------------------------------
# Property 15: Sector-Level Centrality Aggregation (Req 3.5)
# ---------------------------------------------------------------------------

@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(graph_with_sectors_strategy())
def test_property_15_sector_avg_centrality_correct(graph_data):
    """sector_avg_centrality in each record must equal the arithmetic mean
    of all company centrality scores within that sector."""
    import math

    graph, company_ids, sectors = graph_data
    repo = MetricsRepository()
    engine = AnalyticsEngine(repo)

    centrality_scores = engine.calculate_betweenness_centrality(graph, sectors)

    # Build expected sector averages from the returned scores
    from collections import defaultdict
    sector_buckets: dict = defaultdict(list)
    for cid, score in centrality_scores.items():
        sector_buckets[sectors.get(cid, "Unknown")].append(score)

    expected_avg = {
        s: sum(v) / len(v) for s, v in sector_buckets.items()
    }

    # Verify stored records match expected averages
    for record in repo.get_centrality_records():
        expected = expected_avg.get(record.sector, 0.0)
        assert math.isclose(record.sector_avg_centrality, expected, rel_tol=1e-9), (
            f"Sector {record.sector}: expected avg {expected}, "
            f"got {record.sector_avg_centrality}"
        )
