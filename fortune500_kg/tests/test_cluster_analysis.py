"""Unit tests for competitor cluster detection (Task 17.1).

Covers:
- AnalyticsEngine.identify_clusters_from_louvain (Requirement 14.1)
- AnalyticsEngine.calculate_network_density (Requirement 14.2)
- AnalyticsEngine.identify_density_gaps (Requirement 14.3)
- AnalyticsEngine.flag_low_density_clusters (Requirement 14.5)
"""

import pytest

from fortune500_kg.analytics_engine import AnalyticsEngine, MetricsRepository


@pytest.fixture
def engine():
    return AnalyticsEngine(metrics_repo=MetricsRepository())


# ---------------------------------------------------------------------------
# identify_clusters_from_louvain
# ---------------------------------------------------------------------------

class TestIdentifyClustersFromLouvain:

    def test_basic_cluster_mapping(self, engine):
        louvain = {"C1": 0, "C2": 0, "C3": 1, "C4": 1}
        result = engine.identify_clusters_from_louvain(louvain)
        assert set(result.keys()) == {0, 1}
        assert set(result[0]) == {"C1", "C2"}
        assert set(result[1]) == {"C3", "C4"}

    def test_each_company_in_exactly_one_cluster(self, engine):
        louvain = {"A": 0, "B": 1, "C": 2, "D": 0, "E": 1}
        result = engine.identify_clusters_from_louvain(louvain)
        all_companies = [c for companies in result.values() for c in companies]
        assert len(all_companies) == len(louvain)
        assert set(all_companies) == set(louvain.keys())

    def test_single_company_per_cluster(self, engine):
        louvain = {"A": 0, "B": 1, "C": 2}
        result = engine.identify_clusters_from_louvain(louvain)
        assert len(result) == 3
        for cluster_id, companies in result.items():
            assert len(companies) == 1

    def test_empty_returns_empty(self, engine):
        assert engine.identify_clusters_from_louvain({}) == {}

    def test_cluster_ids_match_community_ids(self, engine):
        louvain = {"C1": 5, "C2": 5, "C3": 10}
        result = engine.identify_clusters_from_louvain(louvain)
        assert 5 in result
        assert 10 in result


# ---------------------------------------------------------------------------
# calculate_network_density
# ---------------------------------------------------------------------------

class TestCalculateNetworkDensity:

    def test_fully_connected_cluster(self, engine):
        # Triangle: 3 nodes, 3 edges => density = 3 / (3*2/2) = 1.0
        graph = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}
        density = engine.calculate_network_density(["A", "B", "C"], graph)
        assert density == pytest.approx(1.0)

    def test_no_edges_cluster(self, engine):
        graph = {"A": [], "B": [], "C": []}
        density = engine.calculate_network_density(["A", "B", "C"], graph)
        assert density == pytest.approx(0.0)

    def test_single_node_returns_zero(self, engine):
        graph = {"A": []}
        density = engine.calculate_network_density(["A"], graph)
        assert density == pytest.approx(0.0)

    def test_empty_cluster_returns_zero(self, engine):
        density = engine.calculate_network_density([], {})
        assert density == pytest.approx(0.0)

    def test_partial_connectivity(self, engine):
        # 4 nodes, 2 edges => density = 2 / (4*3/2) = 2/6 ≈ 0.333
        graph = {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}
        density = engine.calculate_network_density(["A", "B", "C", "D"], graph)
        assert density == pytest.approx(2 / 6)

    def test_only_intra_cluster_edges_counted(self, engine):
        # C1 and C2 are in cluster; C3 is outside
        graph = {"C1": ["C2", "C3"], "C2": ["C1"], "C3": ["C1"]}
        density = engine.calculate_network_density(["C1", "C2"], graph)
        # Only C1-C2 edge counts; max_edges = 1
        assert density == pytest.approx(1.0)

    def test_density_formula(self, engine):
        # 5 nodes, 4 edges => density = 4 / (5*4/2) = 4/10 = 0.4
        graph = {
            "A": ["B", "C"],
            "B": ["A", "D"],
            "C": ["A"],
            "D": ["B", "E"],
            "E": ["D"],
        }
        density = engine.calculate_network_density(["A", "B", "C", "D", "E"], graph)
        assert density == pytest.approx(4 / 10)


# ---------------------------------------------------------------------------
# identify_density_gaps
# ---------------------------------------------------------------------------

class TestIdentifyDensityGaps:

    def test_significant_gap_detected(self, engine):
        # Large gap between cluster 0 (0.9) and cluster 1 (0.1)
        densities = {0: 0.9, 1: 0.1, 2: 0.5}
        gaps = engine.identify_density_gaps(densities)
        assert len(gaps) > 0
        # The 0 vs 1 gap (0.8) should be significant
        gap_01 = next((g for g in gaps if {g["cluster_a"], g["cluster_b"]} == {0, 1}), None)
        assert gap_01 is not None
        assert gap_01["significant"] is True

    def test_no_gap_when_all_equal(self, engine):
        densities = {0: 0.5, 1: 0.5, 2: 0.5}
        gaps = engine.identify_density_gaps(densities)
        # All gaps are 0, std_dev is 0, threshold is 0 => no significant gaps
        for g in gaps:
            assert g["gap"] == pytest.approx(0.0)

    def test_single_cluster_returns_empty(self, engine):
        gaps = engine.identify_density_gaps({0: 0.5})
        assert gaps == []

    def test_empty_returns_empty(self, engine):
        assert engine.identify_density_gaps({}) == []

    def test_gap_values_are_absolute_differences(self, engine):
        densities = {0: 0.8, 1: 0.3}
        gaps = engine.identify_density_gaps(densities)
        assert len(gaps) == 1
        assert gaps[0]["gap"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# flag_low_density_clusters
# ---------------------------------------------------------------------------

class TestFlagLowDensityClusters:

    def test_below_median_flagged(self, engine):
        densities = {0: 0.1, 1: 0.5, 2: 0.9}
        # Median = 0.5; cluster 0 (0.1) is below median
        low = engine.flag_low_density_clusters(densities)
        assert 0 in low
        assert 1 not in low
        assert 2 not in low

    def test_even_number_of_clusters(self, engine):
        densities = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8}
        # Median = (0.4 + 0.6) / 2 = 0.5; clusters 0 and 1 are below
        low = engine.flag_low_density_clusters(densities)
        assert 0 in low
        assert 1 in low
        assert 2 not in low
        assert 3 not in low

    def test_empty_returns_empty(self, engine):
        assert engine.flag_low_density_clusters({}) == []

    def test_single_cluster_returns_empty(self, engine):
        # Single cluster: median equals its own density, not strictly below
        low = engine.flag_low_density_clusters({0: 0.5})
        assert low == []

    def test_all_equal_density_returns_empty(self, engine):
        densities = {0: 0.5, 1: 0.5, 2: 0.5}
        # All equal median; none strictly below
        low = engine.flag_low_density_clusters(densities)
        assert low == []
