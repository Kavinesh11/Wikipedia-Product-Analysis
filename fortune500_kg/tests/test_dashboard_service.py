"""Unit tests for DashboardService (Tasks 18.1, 18.3, 18.5).

Covers:
- DashboardService.render_leaderboard (Requirements 5.1, 5.5, 5.6)
- DashboardService.render_network_graph (Requirements 5.2, 5.5)
- DashboardService.render_trend_chart (Requirements 5.3, 5.6)
- DashboardService.render_heatmap (Requirements 5.4, 13.4)
- DashboardService.configure_bloom_overlay (Requirements 6.1-6.5)
"""

from datetime import datetime
import pytest

from fortune500_kg.dashboard_service import DashboardService
from fortune500_kg.data_models import NetworkVisualization, BloomConfig


@pytest.fixture
def service():
    return DashboardService()


SCORES = {"C1": 9.0, "C2": 7.0, "C3": 5.0, "C4": 3.0}
RANKS = {"C1": 10, "C2": 50, "C3": 100, "C4": 200}
SECTORS = {"C1": "Technology", "C2": "Technology", "C3": "Finance", "C4": "Finance"}


# ---------------------------------------------------------------------------
# render_leaderboard
# ---------------------------------------------------------------------------

class TestRenderLeaderboard:

    def test_returns_visualization_with_bar_type(self, service):
        viz = service.render_leaderboard(SCORES, RANKS)
        assert viz.chart_type == "bar"

    def test_all_companies_included_without_filter(self, service):
        viz = service.render_leaderboard(SCORES, RANKS)
        assert len(viz.data) == 4

    def test_each_entry_has_score_and_rank(self, service):
        viz = service.render_leaderboard(SCORES, RANKS)
        for entry in viz.data:
            assert "innovation_score" in entry
            assert "fortune_500_rank" in entry
            assert "company_id" in entry

    def test_sector_filter_applied(self, service):
        viz = service.render_leaderboard(SCORES, RANKS, filters={"sector": "Technology"}, company_sectors=SECTORS)
        assert len(viz.data) == 2
        for entry in viz.data:
            assert entry["company_id"] in {"C1", "C2"}

    def test_sorted_by_score_descending(self, service):
        viz = service.render_leaderboard(SCORES, RANKS)
        scores = [e["innovation_score"] for e in viz.data]
        assert scores == sorted(scores, reverse=True)

    def test_filters_stored_in_visualization(self, service):
        filters = {"sector": "Technology"}
        viz = service.render_leaderboard(SCORES, RANKS, filters=filters, company_sectors=SECTORS)
        assert viz.filters_applied == filters

    def test_company_not_in_ranks_excluded(self, service):
        scores = {"C1": 9.0, "EXTRA": 5.0}
        ranks = {"C1": 10}
        viz = service.render_leaderboard(scores, ranks)
        assert len(viz.data) == 1
        assert viz.data[0]["company_id"] == "C1"


# ---------------------------------------------------------------------------
# render_network_graph
# ---------------------------------------------------------------------------

NODES = [
    {"id": "C1", "label": "Company 1", "metrics": {"innovation_score": 9.0}, "sector": "Technology"},
    {"id": "C2", "label": "Company 2", "metrics": {"innovation_score": 7.0}, "sector": "Technology"},
    {"id": "C3", "label": "Company 3", "metrics": {"innovation_score": 5.0}, "sector": "Finance"},
]
RELS = [
    {"source": "C1", "target": "C2", "type": "PARTNERS_WITH", "weight": 1.0},
    {"source": "C2", "target": "C3", "type": "ACQUIRED", "weight": 2.0},
]


class TestRenderNetworkGraph:

    def test_returns_network_visualization(self, service):
        viz = service.render_network_graph(NODES, RELS)
        assert isinstance(viz, NetworkVisualization)

    def test_all_nodes_included_without_filter(self, service):
        viz = service.render_network_graph(NODES, RELS)
        assert len(viz.nodes) == 3

    def test_each_node_has_metrics(self, service):
        viz = service.render_network_graph(NODES, RELS)
        for node in viz.nodes:
            assert isinstance(node.metrics, dict)

    def test_edges_included(self, service):
        viz = service.render_network_graph(NODES, RELS)
        assert len(viz.edges) == 2

    def test_sector_filter_removes_nodes_and_edges(self, service):
        viz = service.render_network_graph(NODES, RELS, filters={"sector": "Technology"})
        node_ids = {n.node_id for n in viz.nodes}
        assert node_ids == {"C1", "C2"}
        # C2-C3 edge should be excluded since C3 is filtered out
        for edge in viz.edges:
            assert edge.source in node_ids
            assert edge.target in node_ids

    def test_layout_is_force_directed(self, service):
        viz = service.render_network_graph(NODES, RELS)
        assert viz.layout == "force-directed"

    def test_edge_has_relationship_type(self, service):
        viz = service.render_network_graph(NODES, RELS)
        for edge in viz.edges:
            assert edge.relationship_type


# ---------------------------------------------------------------------------
# render_trend_chart
# ---------------------------------------------------------------------------

TS_DATA = [
    {"timestamp": datetime(2022, 1, 1), "value": 5.0},
    {"timestamp": datetime(2024, 1, 1), "value": 8.0},
    {"timestamp": datetime(2023, 1, 1), "value": 6.5},
]


class TestRenderTrendChart:

    def test_returns_line_chart(self, service):
        viz = service.render_trend_chart(TS_DATA, "C1", "innovation_score")
        assert viz.chart_type == "line"

    def test_data_ordered_chronologically(self, service):
        viz = service.render_trend_chart(TS_DATA, "C1", "innovation_score")
        timestamps = [e["timestamp"] for e in viz.data]
        assert timestamps == sorted(timestamps)

    def test_year_filter_applied(self, service):
        viz = service.render_trend_chart(TS_DATA, "C1", "innovation_score", filters={"year": 2023})
        assert len(viz.data) == 1
        assert viz.data[0]["timestamp"].year == 2023

    def test_all_data_without_filter(self, service):
        viz = service.render_trend_chart(TS_DATA, "C1", "innovation_score")
        assert len(viz.data) == 3

    def test_empty_data_returns_empty_visualization(self, service):
        viz = service.render_trend_chart([], "C1", "innovation_score")
        assert viz.data == []


# ---------------------------------------------------------------------------
# render_heatmap
# ---------------------------------------------------------------------------

class TestRenderHeatmap:

    def test_returns_heatmap_chart(self, service):
        centrality = {"Technology": 0.8, "Finance": 0.4}
        bins = {"Technology": [100, 200], "Finance": [50, 150]}
        viz = service.render_heatmap(centrality, bins)
        assert viz.chart_type == "heatmap"

    def test_matrix_has_correct_row_count(self, service):
        centrality = {"Technology": 0.8, "Finance": 0.4, "Healthcare": 0.6}
        bins = {"Technology": [100], "Finance": [100], "Healthcare": [100]}
        viz = service.render_heatmap(centrality, bins)
        assert len(viz.data) == 3

    def test_dimensions_in_config(self, service):
        centrality = {"Technology": 0.8, "Finance": 0.4}
        bins = {"Technology": [100, 200, 300], "Finance": [100, 200, 300]}
        viz = service.render_heatmap(centrality, bins)
        assert viz.config["dimensions"]["rows"] == 2
        assert viz.config["dimensions"]["cols"] == 3


# ---------------------------------------------------------------------------
# configure_bloom_overlay
# ---------------------------------------------------------------------------

class TestConfigureBloomOverlay:

    def test_returns_bloom_config(self, service):
        config = service.configure_bloom_overlay("innovation_score", "ecosystem_centrality")
        assert isinstance(config, BloomConfig)

    def test_node_size_metric_set(self, service):
        config = service.configure_bloom_overlay("innovation_score", "ecosystem_centrality")
        assert config.node_size_metric == "innovation_score"

    def test_node_color_metric_set(self, service):
        config = service.configure_bloom_overlay("innovation_score", "ecosystem_centrality")
        assert config.node_color_metric == "ecosystem_centrality"

    def test_relationship_display_enabled(self, service):
        config = service.configure_bloom_overlay("innovation_score", "ecosystem_centrality")
        assert config.relationship_display is True

    def test_edge_weight_display_enabled(self, service):
        config = service.configure_bloom_overlay("innovation_score", "ecosystem_centrality")
        assert config.edge_weight_display is True

    def test_filters_stored(self, service):
        filters = {"sector": "Technology", "revenue_range": [100, 500]}
        config = service.configure_bloom_overlay("innovation_score", "ecosystem_centrality", filters=filters)
        assert config.filters == filters
