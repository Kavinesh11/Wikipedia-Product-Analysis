"""Property-based tests for competitor cluster detection (Task 17.2).

Properties:
- Property 67: Cluster Identification from Louvain Results
- Property 68: Network Density Calculation per Cluster
- Property 69: Density Gap Identification Threshold
- Property 71: Low-Density Cluster Opportunity Flagging

Validates: Requirements 14.1, 14.2, 14.3, 14.5
"""

import math
from hypothesis import given, settings, assume
import hypothesis.strategies as st
import pytest

from fortune500_kg.analytics_engine import AnalyticsEngine, MetricsRepository


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def louvain_result(draw, min_companies=2, min_clusters=1):
    """Generate a valid Louvain result mapping company_id -> community_id."""
    n_clusters = draw(st.integers(min_value=min_clusters, max_value=8))
    n_companies = draw(st.integers(min_value=max(min_companies, n_clusters), max_value=30))
    company_ids = [f"C{i}" for i in range(n_companies)]
    return {
        cid: draw(st.integers(min_value=0, max_value=n_clusters - 1))
        for cid in company_ids
    }


@st.composite
def cluster_with_graph(draw):
    """Generate a cluster (list of company_ids) and a graph adjacency dict."""
    n = draw(st.integers(min_value=2, max_value=15))
    company_ids = [f"N{i}" for i in range(n)]

    # Build a random undirected graph
    graph = {cid: [] for cid in company_ids}
    for i in range(n):
        for j in range(i + 1, n):
            if draw(st.booleans()):
                graph[company_ids[i]].append(company_ids[j])
                graph[company_ids[j]].append(company_ids[i])

    return company_ids, graph


@st.composite
def cluster_densities(draw, min_clusters=2):
    """Generate a mapping of cluster_id -> density."""
    n = draw(st.integers(min_value=min_clusters, max_value=10))
    return {
        i: draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        for i in range(n)
    }


# ---------------------------------------------------------------------------
# Property 67: Cluster Identification from Louvain Results
# Feature: fortune500-kg-analytics, Property 67
# Validates: Requirements 14.1
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(louvain_result())
def test_property_67_cluster_identification_from_louvain(louvain):
    """
    For any Louvain community detection result, each identified cluster should
    correspond to a unique community_id from the algorithm output.

    **Validates: Requirements 14.1**
    """
    engine = AnalyticsEngine(metrics_repo=MetricsRepository())
    clusters = engine.identify_clusters_from_louvain(louvain)

    # Each cluster_id must be a unique community_id from the Louvain output
    expected_community_ids = set(louvain.values())
    assert set(clusters.keys()) == expected_community_ids

    # Every company must appear in exactly one cluster
    all_companies_in_clusters = [c for companies in clusters.values() for c in companies]
    assert len(all_companies_in_clusters) == len(louvain)
    assert set(all_companies_in_clusters) == set(louvain.keys())

    # Each company's cluster must match its community_id
    for cluster_id, companies in clusters.items():
        for company in companies:
            assert louvain[company] == cluster_id


# ---------------------------------------------------------------------------
# Property 68: Network Density Calculation per Cluster
# Feature: fortune500-kg-analytics, Property 68
# Validates: Requirements 14.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(cluster_with_graph())
def test_property_68_network_density_calculation(cluster_data):
    """
    For any identified cluster with N nodes and E edges, the network density
    should equal E / (N × (N-1) / 2) for undirected graphs.

    **Validates: Requirements 14.2**
    """
    company_ids, graph = cluster_data
    engine = AnalyticsEngine(metrics_repo=MetricsRepository())
    density = engine.calculate_network_density(company_ids, graph)

    n = len(company_ids)
    if n <= 1:
        assert density == pytest.approx(0.0)
        return

    # Count actual intra-cluster edges
    company_set = set(company_ids)
    seen_edges = set()
    for company in company_ids:
        for neighbour in graph.get(company, []):
            if neighbour in company_set:
                edge = tuple(sorted([company, neighbour]))
                seen_edges.add(edge)
    e = len(seen_edges)

    max_edges = n * (n - 1) / 2
    expected_density = e / max_edges if max_edges > 0 else 0.0

    assert density == pytest.approx(expected_density, abs=1e-9)

    # Density must be in [0, 1]
    assert 0.0 <= density <= 1.0


# ---------------------------------------------------------------------------
# Property 69: Density Gap Identification Threshold
# Feature: fortune500-kg-analytics, Property 69
# Validates: Requirements 14.3
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(cluster_densities())
def test_property_69_density_gap_threshold(densities):
    """
    For any identified density gap, the gap should represent a difference
    between cluster densities that exceeds a statistically significant threshold.

    **Validates: Requirements 14.3**
    """
    engine = AnalyticsEngine(metrics_repo=MetricsRepository())
    gaps = engine.identify_density_gaps(densities)

    if len(densities) < 2:
        assert gaps == []
        return

    # Compute threshold: 1.5 * std_dev
    density_vals = list(densities.values())
    mean_d = sum(density_vals) / len(density_vals)
    variance = sum((d - mean_d) ** 2 for d in density_vals) / len(density_vals)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0
    threshold = 1.5 * std_dev

    for gap_info in gaps:
        cid_a = gap_info["cluster_a"]
        cid_b = gap_info["cluster_b"]
        gap = gap_info["gap"]
        significant = gap_info["significant"]

        # Gap value must equal absolute difference between the two densities
        expected_gap = abs(densities[cid_a] - densities[cid_b])
        assert gap == pytest.approx(expected_gap, abs=1e-9)

        # Significance flag must match threshold comparison
        assert significant == (gap > threshold)


# ---------------------------------------------------------------------------
# Property 71: Low-Density Cluster Opportunity Flagging
# Feature: fortune500-kg-analytics, Property 71
# Validates: Requirements 14.5
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(cluster_densities())
def test_property_71_low_density_cluster_flagging(densities):
    """
    For any cluster with network density below the median density across all
    clusters, the cluster should be flagged as a potential acquisition or
    partnership opportunity.

    **Validates: Requirements 14.5**
    """
    engine = AnalyticsEngine(metrics_repo=MetricsRepository())
    low_density_clusters = engine.flag_low_density_clusters(densities)

    if not densities:
        assert low_density_clusters == []
        return

    # Compute median
    sorted_vals = sorted(densities.values())
    n = len(sorted_vals)
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    # Every flagged cluster must have density strictly below median
    for cluster_id in low_density_clusters:
        assert densities[cluster_id] < median, (
            f"Cluster {cluster_id} with density {densities[cluster_id]} "
            f"was flagged but is not below median {median}"
        )

    # Every cluster with density strictly below median must be flagged
    for cluster_id, density in densities.items():
        if density < median:
            assert cluster_id in low_density_clusters, (
                f"Cluster {cluster_id} with density {density} "
                f"should be flagged (below median {median})"
            )
