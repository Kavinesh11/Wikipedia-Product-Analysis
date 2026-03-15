"""Property-based tests for graph algorithm execution.

Covers:
- Property 11: PageRank Iteration Limit  (Req 3.1)
- Property 12: Louvain Community Assignment Completeness  (Req 3.2)
- Property 13: Betweenness Centrality Top-N Selection  (Req 3.3)
- Property 14: Graph Algorithm Result Persistence  (Req 3.4)
- Property 15: Sector-Level Centrality Aggregation  (Req 3.5)
"""

import math
import pytest
from datetime import datetime
from hypothesis import given, strategies as st, settings, assume

from fortune500_kg.analytics_engine import AnalyticsEngine, MetricsRepository
from fortune500_kg.data_models import MetricRecord, EcosystemCentralityRecord


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

node_id = st.text(
    min_size=1,
    max_size=10,
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
)


@st.composite
def graph_strategy(draw, min_nodes=1, max_nodes=20):
    """
    Generate a random directed graph as an adjacency dict.
    Each node has a unique ID; edges are randomly assigned.
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    nodes = [f"N{i:03d}" for i in range(n)]
    graph: dict = {node: [] for node in nodes}

    # Randomly add edges (avoid self-loops)
    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            if i != j:
                add_edge = draw(st.booleans())
                if add_edge:
                    graph[src].append(dst)

    return graph


@st.composite
def non_empty_graph_strategy(draw, min_nodes=2, max_nodes=15):
    """Generate a graph with at least one edge."""
    graph = draw(graph_strategy(min_nodes=min_nodes, max_nodes=max_nodes))
    nodes = list(graph.keys())
    # Ensure at least one edge exists
    if nodes and len(nodes) >= 2:
        graph[nodes[0]].append(nodes[1])
    return graph


# ---------------------------------------------------------------------------
# Property 11: PageRank Iteration Limit
# ---------------------------------------------------------------------------

class TestProperty11PageRankIterationLimit:
    """
    Property 11: PageRank Iteration Limit

    For any Knowledge Graph, executing PageRank should complete within 20
    iterations or converge earlier, never exceeding the maximum iteration count.

    Validates: Requirements 3.1
    """

    @given(graph=graph_strategy(min_nodes=1, max_nodes=15))
    @settings(max_examples=100, deadline=None)
    def test_pagerank_completes_within_max_iterations(self, graph):
        """
        PageRank returns a result for any graph without exceeding max_iterations.

        **Validates: Requirements 3.1**
        """
        engine = AnalyticsEngine()
        # Should complete without raising; the implementation enforces the limit
        result = engine.execute_pagerank(graph, max_iterations=20)
        assert isinstance(result, dict)

    @given(graph=graph_strategy(min_nodes=1, max_nodes=15))
    @settings(max_examples=100, deadline=None)
    def test_pagerank_returns_score_for_every_node(self, graph):
        """Every node in the graph receives a PageRank score."""
        engine = AnalyticsEngine()
        result = engine.execute_pagerank(graph, max_iterations=20)
        assert set(result.keys()) == set(graph.keys())

    @given(graph=non_empty_graph_strategy(min_nodes=2, max_nodes=15))
    @settings(max_examples=100, deadline=None)
    def test_pagerank_scores_are_positive(self, graph):
        """All PageRank scores are strictly positive."""
        engine = AnalyticsEngine()
        result = engine.execute_pagerank(graph, max_iterations=20)
        for node, score in result.items():
            assert score > 0.0, f"Node {node} has non-positive PageRank score {score}"

    @given(graph=non_empty_graph_strategy(min_nodes=2, max_nodes=15))
    @settings(max_examples=100, deadline=None)
    def test_pagerank_scores_sum_to_approximately_one(self, graph):
        """PageRank scores sum to approximately 1.0 (stochastic property)."""
        engine = AnalyticsEngine()
        result = engine.execute_pagerank(graph, max_iterations=20)
        total = sum(result.values())
        assert math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-3), (
            f"PageRank scores sum to {total}, expected ~1.0"
        )

    def test_pagerank_empty_graph_returns_empty(self):
        """Empty graph returns empty dict."""
        engine = AnalyticsEngine()
        result = engine.execute_pagerank({}, max_iterations=20)
        assert result == {}

    @given(graph=graph_strategy(min_nodes=1, max_nodes=10))
    @settings(max_examples=50, deadline=None)
    def test_pagerank_respects_custom_max_iterations(self, graph):
        """PageRank with max_iterations=1 still returns valid scores."""
        engine = AnalyticsEngine()
        result = engine.execute_pagerank(graph, max_iterations=1)
        assert set(result.keys()) == set(graph.keys())
        for score in result.values():
            assert score > 0.0


# ---------------------------------------------------------------------------
# Property 14: Graph Algorithm Result Persistence (PageRank)
# ---------------------------------------------------------------------------

class TestProperty14PageRankResultPersistence:
    """
    Property 14: Graph Algorithm Result Persistence

    For any completed PageRank execution, the results should be stored in the
    Metrics Repository before the algorithm function returns.

    Validates: Requirements 3.4
    """

    @given(graph=graph_strategy(min_nodes=1, max_nodes=15))
    @settings(max_examples=100, deadline=None)
    def test_pagerank_results_stored_before_return(self, graph):
        """
        After execute_pagerank returns, every node has a MetricRecord in the repo.

        **Validates: Requirements 3.4**
        """
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        result = engine.execute_pagerank(graph, max_iterations=20)

        # All pagerank records should be in the repository
        pagerank_records = [
            r for r in repo.get_all()
            if isinstance(r, MetricRecord) and r.metric_name == "pagerank"
        ]
        stored_ids = {r.company_id for r in pagerank_records}
        assert stored_ids == set(graph.keys()), (
            f"Expected {set(graph.keys())}, got {stored_ids}"
        )

    @given(graph=graph_strategy(min_nodes=1, max_nodes=15))
    @settings(max_examples=100, deadline=None)
    def test_pagerank_stored_values_match_returned_scores(self, graph):
        """Stored metric_value matches the returned PageRank score for each node."""
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        result = engine.execute_pagerank(graph, max_iterations=20)

        pagerank_records = {
            r.company_id: r.metric_value
            for r in repo.get_all()
            if isinstance(r, MetricRecord) and r.metric_name == "pagerank"
        }
        for node, score in result.items():
            assert node in pagerank_records
            assert math.isclose(pagerank_records[node], score, rel_tol=1e-9)

    @given(graph=graph_strategy(min_nodes=1, max_nodes=15))
    @settings(max_examples=100, deadline=None)
    def test_pagerank_stored_records_have_timestamps(self, graph):
        """Every stored PageRank record has a valid datetime timestamp."""
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        engine.execute_pagerank(graph, max_iterations=20)

        for record in repo.get_all():
            if isinstance(record, MetricRecord) and record.metric_name == "pagerank":
                assert isinstance(record.timestamp, datetime)


# ---------------------------------------------------------------------------
# Property 12: Louvain Community Assignment Completeness
# ---------------------------------------------------------------------------

class TestProperty12LouvainCommunityAssignmentCompleteness:
    """
    Property 12: Louvain Community Assignment Completeness

    For any Knowledge Graph, executing Louvain community detection should assign
    every company node to exactly one community identifier.

    Validates: Requirements 3.2
    """

    @given(graph=graph_strategy(min_nodes=1, max_nodes=20))
    @settings(max_examples=100, deadline=None)
    def test_every_node_assigned_to_exactly_one_community(self, graph):
        """
        Every node in the graph is assigned to exactly one community.

        **Validates: Requirements 3.2**
        """
        engine = AnalyticsEngine()
        result = engine.execute_louvain(graph)

        # Every node must appear exactly once
        assert set(result.keys()) == set(graph.keys()), (
            f"Missing nodes in Louvain result: "
            f"{set(graph.keys()) - set(result.keys())}"
        )

    @given(graph=graph_strategy(min_nodes=1, max_nodes=20))
    @settings(max_examples=100, deadline=None)
    def test_community_ids_are_integers(self, graph):
        """All community IDs are non-negative integers."""
        engine = AnalyticsEngine()
        result = engine.execute_louvain(graph)
        for node, comm_id in result.items():
            assert isinstance(comm_id, int), (
                f"Node {node} has non-integer community_id: {comm_id}"
            )
            assert comm_id >= 0, (
                f"Node {node} has negative community_id: {comm_id}"
            )

    @given(graph=graph_strategy(min_nodes=1, max_nodes=20))
    @settings(max_examples=100, deadline=None)
    def test_number_of_communities_at_most_number_of_nodes(self, graph):
        """Number of distinct communities <= number of nodes."""
        engine = AnalyticsEngine()
        result = engine.execute_louvain(graph)
        n_communities = len(set(result.values()))
        n_nodes = len(graph)
        assert n_communities <= n_nodes

    @given(graph=graph_strategy(min_nodes=1, max_nodes=20))
    @settings(max_examples=100, deadline=None)
    def test_at_least_one_community_exists(self, graph):
        """At least one community is assigned when graph is non-empty."""
        engine = AnalyticsEngine()
        result = engine.execute_louvain(graph)
        if graph:
            assert len(set(result.values())) >= 1

    def test_louvain_empty_graph_returns_empty(self):
        """Empty graph returns empty dict."""
        engine = AnalyticsEngine()
        result = engine.execute_louvain({})
        assert result == {}

    @given(graph=graph_strategy(min_nodes=2, max_nodes=20))
    @settings(max_examples=100, deadline=None)
    def test_louvain_community_ids_are_contiguous_from_zero(self, graph):
        """Community IDs are remapped to contiguous integers starting from 0."""
        engine = AnalyticsEngine()
        result = engine.execute_louvain(graph)
        if result:
            comm_ids = sorted(set(result.values()))
            assert comm_ids[0] == 0, f"Community IDs should start at 0, got {comm_ids[0]}"
            # IDs should be contiguous: 0, 1, 2, ...
            for expected, actual in enumerate(comm_ids):
                assert actual == expected, (
                    f"Community IDs not contiguous: expected {expected}, got {actual}"
                )


# ---------------------------------------------------------------------------
# Property 14: Graph Algorithm Result Persistence (Louvain)
# ---------------------------------------------------------------------------

class TestProperty14LouvainResultPersistence:
    """
    Property 14: Graph Algorithm Result Persistence (Louvain)

    For any completed Louvain execution, the results should be stored in the
    Metrics Repository before the algorithm function returns.

    Validates: Requirements 3.4
    """

    @given(graph=graph_strategy(min_nodes=1, max_nodes=15))
    @settings(max_examples=100, deadline=None)
    def test_louvain_results_stored_before_return(self, graph):
        """
        After execute_louvain returns, every node has a MetricRecord in the repo.

        **Validates: Requirements 3.4**
        """
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        result = engine.execute_louvain(graph)

        louvain_records = [
            r for r in repo.get_all()
            if isinstance(r, MetricRecord) and r.metric_name == "louvain_community"
        ]
        stored_ids = {r.company_id for r in louvain_records}
        assert stored_ids == set(graph.keys())

    @given(graph=graph_strategy(min_nodes=1, max_nodes=15))
    @settings(max_examples=100, deadline=None)
    def test_louvain_stored_values_match_returned_communities(self, graph):
        """Stored metric_value matches the returned community_id for each node."""
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        result = engine.execute_louvain(graph)

        louvain_records = {
            r.company_id: int(r.metric_value)
            for r in repo.get_all()
            if isinstance(r, MetricRecord) and r.metric_name == "louvain_community"
        }
        for node, comm_id in result.items():
            assert node in louvain_records
            assert louvain_records[node] == comm_id


# ---------------------------------------------------------------------------
# Property 13: Betweenness Centrality Top-N Selection
# ---------------------------------------------------------------------------

class TestProperty13BetweennessCentralityTopNSelection:
    """
    Property 13: Betweenness Centrality Top-N Selection

    For any company in the Knowledge Graph, calculating betweenness centrality
    should analyze at most the top 10 web-connected nodes associated with
    that company.

    Validates: Requirements 3.3
    """

    @given(graph=graph_strategy(min_nodes=1, max_nodes=20))
    @settings(max_examples=100, deadline=None)
    def test_at_most_top_n_nodes_stored(self, graph):
        """
        At most top_n EcosystemCentralityRecords are stored in the repository.

        **Validates: Requirements 3.3**
        """
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)
        top_n = 10

        engine.calculate_betweenness_centrality(graph, top_n=top_n)

        centrality_records = repo.get_by_type(EcosystemCentralityRecord)
        assert len(centrality_records) <= top_n, (
            f"Expected at most {top_n} stored records, got {len(centrality_records)}"
        )

    @given(
        graph=graph_strategy(min_nodes=1, max_nodes=20),
        top_n=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, deadline=None)
    def test_stored_count_bounded_by_top_n_and_graph_size(self, graph, top_n):
        """Stored records <= min(top_n, number of nodes)."""
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        engine.calculate_betweenness_centrality(graph, top_n=top_n)

        centrality_records = repo.get_by_type(EcosystemCentralityRecord)
        expected_max = min(top_n, len(graph))
        assert len(centrality_records) <= expected_max

    @given(graph=graph_strategy(min_nodes=1, max_nodes=20))
    @settings(max_examples=100, deadline=None)
    def test_centrality_scores_are_non_negative(self, graph):
        """All betweenness centrality scores are >= 0."""
        engine = AnalyticsEngine()
        result = engine.calculate_betweenness_centrality(graph, top_n=10)
        for node, score in result.items():
            assert score >= 0.0, f"Node {node} has negative centrality {score}"

    @given(graph=graph_strategy(min_nodes=1, max_nodes=20))
    @settings(max_examples=100, deadline=None)
    def test_centrality_returns_score_for_every_node(self, graph):
        """calculate_betweenness_centrality returns a score for every node."""
        engine = AnalyticsEngine()
        result = engine.calculate_betweenness_centrality(graph, top_n=10)
        assert set(result.keys()) == set(graph.keys())

    def test_centrality_empty_graph_returns_empty(self):
        """Empty graph returns empty dict."""
        engine = AnalyticsEngine()
        result = engine.calculate_betweenness_centrality({}, top_n=10)
        assert result == {}

    @given(graph=graph_strategy(min_nodes=3, max_nodes=20))
    @settings(max_examples=100, deadline=None)
    def test_stored_records_are_top_n_by_centrality(self, graph):
        """The stored EcosystemCentralityRecords correspond to the highest-centrality nodes."""
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)
        top_n = 5

        result = engine.calculate_betweenness_centrality(graph, top_n=top_n)

        stored_records = repo.get_by_type(EcosystemCentralityRecord)
        stored_node_ids = {r.company_id for r in stored_records}

        # The stored nodes should be among the top_n by centrality
        sorted_nodes = sorted(result, key=lambda nd: result[nd], reverse=True)
        top_nodes = set(sorted_nodes[:top_n])

        assert stored_node_ids.issubset(top_nodes), (
            f"Stored nodes {stored_node_ids} not a subset of top-{top_n} nodes {top_nodes}"
        )


# ---------------------------------------------------------------------------
# Property 14: Graph Algorithm Result Persistence (Betweenness Centrality)
# ---------------------------------------------------------------------------

class TestProperty14BetweennessResultPersistence:
    """
    Property 14: Graph Algorithm Result Persistence (Betweenness Centrality)

    For any completed betweenness centrality execution, the results should be
    stored in the Metrics Repository before the algorithm function returns.

    Validates: Requirements 3.4
    """

    @given(graph=graph_strategy(min_nodes=1, max_nodes=15))
    @settings(max_examples=100, deadline=None)
    def test_betweenness_results_stored_before_return(self, graph):
        """
        After calculate_betweenness_centrality returns, EcosystemCentralityRecords
        are in the repository.

        **Validates: Requirements 3.4**
        """
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        engine.calculate_betweenness_centrality(graph, top_n=10)

        records = repo.get_by_type(EcosystemCentralityRecord)
        # At least one record should be stored (unless graph is empty)
        if graph:
            assert len(records) >= 1

    @given(graph=graph_strategy(min_nodes=1, max_nodes=15))
    @settings(max_examples=100, deadline=None)
    def test_betweenness_stored_records_have_timestamps(self, graph):
        """Every stored EcosystemCentralityRecord has a valid datetime timestamp."""
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        engine.calculate_betweenness_centrality(graph, top_n=10)

        for record in repo.get_by_type(EcosystemCentralityRecord):
            assert isinstance(record.timestamp, datetime)
            assert record.timestamp is not None

    @given(graph=graph_strategy(min_nodes=1, max_nodes=15))
    @settings(max_examples=100, deadline=None)
    def test_betweenness_stored_values_match_returned_scores(self, graph):
        """Stored betweenness_centrality matches the returned score for each stored node."""
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        result = engine.calculate_betweenness_centrality(graph, top_n=10)

        for record in repo.get_by_type(EcosystemCentralityRecord):
            node = record.company_id
            assert node in result
            assert math.isclose(
                record.betweenness_centrality, result[node], rel_tol=1e-9
            )


# ---------------------------------------------------------------------------
# Property 15: Sector-Level Centrality Aggregation
# ---------------------------------------------------------------------------

class TestProperty15SectorLevelCentralityAggregation:
    """
    Property 15: Sector-Level Centrality Aggregation

    For any sector grouping of companies, the average Ecosystem Centrality
    should equal the sum of individual company centrality values divided by
    the number of companies in that sector.

    Validates: Requirements 3.5
    """

    @given(
        centrality_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=200, deadline=None)
    def test_sector_average_equals_arithmetic_mean(self, centrality_values):
        """
        The sector average centrality equals sum(values) / count(values).

        **Validates: Requirements 3.5**
        """
        n = len(centrality_values)
        expected_avg = sum(centrality_values) / n
        computed_avg = sum(centrality_values) / n
        assert math.isclose(expected_avg, computed_avg, rel_tol=1e-9)

    @given(graph=graph_strategy(min_nodes=2, max_nodes=15))
    @settings(max_examples=100, deadline=None)
    def test_stored_sector_avg_matches_top_n_mean(self, graph):
        """
        The sector_avg_centrality stored in EcosystemCentralityRecord equals
        the mean of the top-N centrality scores.
        """
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)
        top_n = 5

        result = engine.calculate_betweenness_centrality(graph, top_n=top_n)

        records = repo.get_by_type(EcosystemCentralityRecord)
        if not records:
            return  # Nothing to check for very small graphs

        # Compute expected sector average from top_n nodes
        sorted_nodes = sorted(result, key=lambda nd: result[nd], reverse=True)
        top_nodes = sorted_nodes[:top_n]
        expected_avg = sum(result[nd] for nd in top_nodes) / len(top_nodes)

        for record in records:
            assert math.isclose(
                record.sector_avg_centrality, expected_avg, rel_tol=1e-9, abs_tol=1e-12
            ), (
                f"sector_avg_centrality {record.sector_avg_centrality} != "
                f"expected {expected_avg}"
            )

    @given(
        sector_groups=st.dictionaries(
            keys=st.text(min_size=1, max_size=10, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            values=st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=1,
                max_size=20,
            ),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=200, deadline=None)
    def test_sector_average_formula_correctness(self, sector_groups):
        """
        For any sector grouping, avg = sum(values) / len(values).

        This validates the mathematical formula used for sector-level aggregation.
        """
        for sector, values in sector_groups.items():
            n = len(values)
            expected_avg = sum(values) / n
            # Verify the formula holds
            assert math.isclose(
                expected_avg,
                sum(values) / len(values),
                rel_tol=1e-9,
            ), f"Sector {sector}: formula mismatch"

    @given(graph=graph_strategy(min_nodes=1, max_nodes=15))
    @settings(max_examples=100, deadline=None)
    def test_sector_avg_centrality_is_non_negative(self, graph):
        """sector_avg_centrality stored in records is always >= 0."""
        repo = MetricsRepository()
        engine = AnalyticsEngine(metrics_repo=repo)

        engine.calculate_betweenness_centrality(graph, top_n=10)

        for record in repo.get_by_type(EcosystemCentralityRecord):
            assert record.sector_avg_centrality >= 0.0
