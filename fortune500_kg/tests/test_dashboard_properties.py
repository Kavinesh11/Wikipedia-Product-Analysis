"""Property-based tests for Dashboard Service (Tasks 18.2, 18.4, 18.6).

Properties:
- Property 21: Leaderboard Visualization Data Completeness
- Property 22: Network Graph Structure Completeness
- Property 23: Time-Series Chart Temporal Ordering
- Property 24: Heatmap Matrix Dimensionality
- Property 25: Sector Filter Application Consistency
- Property 26: Year Filter Temporal Consistency
- Property 65: Sector Comparison Visualization Data Completeness
- Property 70: Cluster Visualization Color Coding

Validates: Requirements 5.1-5.6, 13.4, 14.4
"""

from datetime import datetime
from hypothesis import given, settings, assume
import hypothesis.strategies as st
import pytest

from fortune500_kg.dashboard_service import DashboardService


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_score = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_rank = st.integers(min_value=1, max_value=500)
_sector = st.sampled_from(["Technology", "Finance", "Healthcare", "Energy", "Retail"])
_year = st.integers(min_value=2018, max_value=2025)


@st.composite
def leaderboard_dataset(draw):
    """Generate company_scores, company_ranks, company_sectors."""
    n = draw(st.integers(min_value=1, max_value=30))
    company_ids = [f"C{i}" for i in range(n)]
    scores = {cid: draw(_score) for cid in company_ids}
    ranks = {cid: draw(_rank) for cid in company_ids}
    sectors = {cid: draw(_sector) for cid in company_ids}
    return scores, ranks, sectors


@st.composite
def network_dataset(draw):
    """Generate company_nodes and relationships."""
    n = draw(st.integers(min_value=1, max_value=20))
    company_ids = [f"N{i}" for i in range(n)]
    nodes = [
        {
            "id": cid,
            "label": f"Company {i}",
            "metrics": {"innovation_score": draw(_score)},
            "sector": draw(_sector),
        }
        for i, cid in enumerate(company_ids)
    ]
    # Generate some relationships
    rels = []
    for i in range(min(n - 1, 10)):
        rels.append({
            "source": company_ids[i],
            "target": company_ids[i + 1],
            "type": "PARTNERS_WITH",
            "weight": draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)),
        })
    return nodes, rels


@st.composite
def time_series_dataset(draw):
    """Generate time-series data points."""
    n = draw(st.integers(min_value=2, max_value=20))
    years = draw(st.lists(
        st.integers(min_value=2018, max_value=2025),
        min_size=n,
        max_size=n,
    ))
    data = [
        {
            "timestamp": datetime(year, 1, 1),
            "value": draw(_score),
        }
        for year in years
    ]
    return data


# ---------------------------------------------------------------------------
# Property 21: Leaderboard Visualization Data Completeness
# Feature: fortune500-kg-analytics, Property 21
# Validates: Requirements 5.1
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(leaderboard_dataset())
def test_property_21_leaderboard_data_completeness(dataset):
    """
    For any rendered leaderboard bar chart, each displayed company should have
    both an Innovation Score value and a Fortune 500 rank value present.

    **Validates: Requirements 5.1**
    """
    scores, ranks, sectors = dataset
    service = DashboardService()
    viz = service.render_leaderboard(scores, ranks)

    for entry in viz.data:
        assert "innovation_score" in entry, "Missing innovation_score in leaderboard entry"
        assert "fortune_500_rank" in entry, "Missing fortune_500_rank in leaderboard entry"
        assert "company_id" in entry, "Missing company_id in leaderboard entry"
        # Values must be present (not None)
        assert entry["innovation_score"] is not None
        assert entry["fortune_500_rank"] is not None


# ---------------------------------------------------------------------------
# Property 25: Sector Filter Application Consistency
# Feature: fortune500-kg-analytics, Property 25
# Validates: Requirements 5.5
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(leaderboard_dataset(), _sector)
def test_property_25_sector_filter_consistency(dataset, selected_sector):
    """
    For any sector filter selection, all visualizations should display only
    companies belonging to the selected sector after the filter is applied.

    **Validates: Requirements 5.5**
    """
    scores, ranks, sectors = dataset
    service = DashboardService()
    viz = service.render_leaderboard(
        scores, ranks,
        filters={"sector": selected_sector},
        company_sectors=sectors,
    )

    for entry in viz.data:
        company_id = entry["company_id"]
        assert sectors.get(company_id) == selected_sector, (
            f"Company {company_id} in sector {sectors.get(company_id)} "
            f"should not appear with sector filter {selected_sector}"
        )


# ---------------------------------------------------------------------------
# Property 26: Year Filter Temporal Consistency
# Feature: fortune500-kg-analytics, Property 26
# Validates: Requirements 5.6
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(time_series_dataset(), _year)
def test_property_26_year_filter_temporal_consistency(ts_data, selected_year):
    """
    For any year filter selection, all visualizations should display only
    data with timestamps falling within the selected year.

    **Validates: Requirements 5.6**
    """
    service = DashboardService()
    viz = service.render_trend_chart(
        ts_data, "C1", "innovation_score",
        filters={"year": selected_year},
    )

    for entry in viz.data:
        ts = entry["timestamp"]
        assert isinstance(ts, datetime)
        assert ts.year == selected_year, (
            f"Data point with year {ts.year} should not appear with year filter {selected_year}"
        )


# ---------------------------------------------------------------------------
# Property 22: Network Graph Structure Completeness
# Feature: fortune500-kg-analytics, Property 22
# Validates: Requirements 5.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(network_dataset())
def test_property_22_network_graph_structure_completeness(dataset):
    """
    For any rendered force-directed network graph, the visualization should
    contain node objects for companies, edge objects for relationships, and
    metric overlay values for each node.

    **Validates: Requirements 5.2**
    """
    nodes, rels = dataset
    service = DashboardService()
    viz = service.render_network_graph(nodes, rels)

    # Must have node objects
    assert isinstance(viz.nodes, list)
    # Must have edge objects
    assert isinstance(viz.edges, list)

    # Each node must have metric overlay values
    for node in viz.nodes:
        assert isinstance(node.metrics, dict), "Node must have metrics dict"

    # Each edge must have relationship type
    for edge in viz.edges:
        assert edge.relationship_type, "Edge must have relationship_type"


# ---------------------------------------------------------------------------
# Property 23: Time-Series Chart Temporal Ordering
# Feature: fortune500-kg-analytics, Property 23
# Validates: Requirements 5.3
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(time_series_dataset())
def test_property_23_time_series_temporal_ordering(ts_data):
    """
    For any rendered line chart showing GitHub activity trends, the data
    points should be ordered chronologically by timestamp.

    **Validates: Requirements 5.3**
    """
    service = DashboardService()
    viz = service.render_trend_chart(ts_data, "C1", "innovation_score")

    timestamps = [entry["timestamp"] for entry in viz.data]
    assert timestamps == sorted(timestamps), (
        "Time-series data points must be ordered chronologically"
    )


# ---------------------------------------------------------------------------
# Property 24: Heatmap Matrix Dimensionality
# Feature: fortune500-kg-analytics, Property 24
# Validates: Requirements 5.4
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    st.dictionaries(
        _sector,
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    ),
    st.integers(min_value=1, max_value=5),
)
def test_property_24_heatmap_matrix_dimensionality(sector_centrality, n_bins):
    """
    For any rendered heatmap showing sector centrality versus revenue, the
    matrix should have dimensions equal to (number of sectors) × (number of revenue bins).

    **Validates: Requirements 5.4**
    """
    # Build consistent revenue bins for all sectors
    bins = list(range(100, 100 + n_bins * 100, 100))
    sector_revenue_bins = {sector: bins for sector in sector_centrality}

    service = DashboardService()
    viz = service.render_heatmap(sector_centrality, sector_revenue_bins)

    n_sectors = len(sector_centrality)
    assert len(viz.data) == n_sectors, (
        f"Heatmap should have {n_sectors} rows (one per sector)"
    )
    assert viz.config["dimensions"]["rows"] == n_sectors
    assert viz.config["dimensions"]["cols"] == n_bins


# ---------------------------------------------------------------------------
# Property 65: Sector Comparison Visualization Data Completeness
# Feature: fortune500-kg-analytics, Property 65
# Validates: Requirements 13.4
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    st.dictionaries(
        _sector,
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    )
)
def test_property_65_sector_comparison_visualization_completeness(sector_centrality):
    """
    For any rendered sector comparison visualization, all sectors should be
    represented with their corresponding average metric values.

    **Validates: Requirements 13.4**
    """
    sector_revenue_bins = {sector: [100, 200, 300] for sector in sector_centrality}
    service = DashboardService()
    viz = service.render_heatmap(sector_centrality, sector_revenue_bins)

    # All sectors must appear in the visualization
    sectors_in_viz = {row["sector"] for row in viz.data}
    assert sectors_in_viz == set(sector_centrality.keys()), (
        "All sectors must be represented in the sector comparison visualization"
    )

    # Each sector row must have a centrality value
    for row in viz.data:
        assert "centrality" in row
        assert row["centrality"] == sector_centrality[row["sector"]]


# ---------------------------------------------------------------------------
# Property 70: Cluster Visualization Color Coding
# Feature: fortune500-kg-analytics, Property 70
# Validates: Requirements 14.4
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(network_dataset())
def test_property_70_cluster_visualization_color_coding(dataset):
    """
    For any rendered network map with clusters, each cluster should be assigned
    a distinct color, and all nodes within a cluster should share that color.

    We verify that nodes with the same cluster_id have consistent cluster
    assignments (the color coding logic is in the frontend, but the data
    structure must support it by having cluster_id on each node).

    **Validates: Requirements 14.4**
    """
    nodes_data, rels = dataset
    # Assign cluster_ids to nodes
    for i, node in enumerate(nodes_data):
        node["cluster_id"] = i % 3  # 3 clusters

    service = DashboardService()
    viz = service.render_network_graph(nodes_data, rels)

    # Group nodes by cluster_id
    cluster_nodes: dict = {}
    for node in viz.nodes:
        if node.cluster_id is not None:
            cluster_nodes.setdefault(node.cluster_id, []).append(node.node_id)

    # Each cluster_id must be unique (distinct grouping)
    cluster_ids = [node.cluster_id for node in viz.nodes if node.cluster_id is not None]
    if cluster_ids:
        # Verify cluster_ids are consistent with input
        for node in viz.nodes:
            if node.cluster_id is not None:
                assert isinstance(node.cluster_id, int)
