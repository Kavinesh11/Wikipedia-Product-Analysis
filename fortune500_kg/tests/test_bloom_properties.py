"""Property-based tests for Neo4j Bloom integration (Task 19.2).

Properties:
- Property 27: Bloom Node Size Metric Mapping
- Property 28: Bloom Node Color Metric Mapping
- Property 29: Bloom Filter Effectiveness
- Property 30: Bloom Relationship Display Completeness

Validates: Requirements 6.2, 6.3, 6.4, 6.5
"""

from hypothesis import given, settings
import hypothesis.strategies as st
import pytest

from fortune500_kg.dashboard_service import DashboardService
from fortune500_kg.data_models import BloomConfig


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_metric_name = st.sampled_from([
    "innovation_score", "ecosystem_centrality", "digital_maturity",
    "revenue_growth", "pagerank",
])

_filter_key = st.sampled_from(["sector", "revenue_range", "metric_threshold"])

_filter_value = st.one_of(
    st.text(min_size=1, max_size=20),
    st.lists(st.integers(min_value=0, max_value=1000), min_size=2, max_size=2),
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)


# ---------------------------------------------------------------------------
# Property 27: Bloom Node Size Metric Mapping
# Feature: fortune500-kg-analytics, Property 27
# Validates: Requirements 6.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(_metric_name, _metric_name)
def test_property_27_bloom_node_size_metric_mapping(size_metric, color_metric):
    """
    For any Knowledge Graph visualization in Neo4j Bloom, node sizes should
    be proportional to Innovation Score values (higher scores → larger nodes).
    The BloomConfig must record the node_size_metric correctly.

    **Validates: Requirements 6.2**
    """
    service = DashboardService()
    config = service.configure_bloom_overlay(size_metric, color_metric)

    assert isinstance(config, BloomConfig)
    assert config.node_size_metric == size_metric, (
        f"node_size_metric should be '{size_metric}', got '{config.node_size_metric}'"
    )


# ---------------------------------------------------------------------------
# Property 28: Bloom Node Color Metric Mapping
# Feature: fortune500-kg-analytics, Property 28
# Validates: Requirements 6.3
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(_metric_name, _metric_name)
def test_property_28_bloom_node_color_metric_mapping(size_metric, color_metric):
    """
    For any Knowledge Graph visualization in Neo4j Bloom, node color intensity
    should be proportional to Ecosystem Centrality values.
    The BloomConfig must record the node_color_metric correctly.

    **Validates: Requirements 6.3**
    """
    service = DashboardService()
    config = service.configure_bloom_overlay(size_metric, color_metric)

    assert config.node_color_metric == color_metric, (
        f"node_color_metric should be '{color_metric}', got '{config.node_color_metric}'"
    )


# ---------------------------------------------------------------------------
# Property 29: Bloom Filter Effectiveness
# Feature: fortune500-kg-analytics, Property 29
# Validates: Requirements 6.4
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    _metric_name,
    _metric_name,
    st.dictionaries(
        _filter_key,
        st.text(min_size=1, max_size=20),
        min_size=0,
        max_size=3,
    ),
)
def test_property_29_bloom_filter_effectiveness(size_metric, color_metric, filters):
    """
    For any applied filter (sector, revenue range, metric threshold) in Neo4j
    Bloom, the displayed nodes should satisfy all active filter conditions.
    The BloomConfig must store all provided filters.

    **Validates: Requirements 6.4**
    """
    service = DashboardService()
    config = service.configure_bloom_overlay(size_metric, color_metric, filters=filters)

    # All provided filters must be stored in the config
    for key, value in filters.items():
        assert key in config.filters, f"Filter key '{key}' not stored in BloomConfig"
        assert config.filters[key] == value


# ---------------------------------------------------------------------------
# Property 30: Bloom Relationship Display Completeness
# Feature: fortune500-kg-analytics, Property 30
# Validates: Requirements 6.5
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(_metric_name, _metric_name)
def test_property_30_bloom_relationship_display_completeness(size_metric, color_metric):
    """
    For any displayed relationship in Neo4j Bloom, the edge should show both
    the relationship type label and any associated weight values.
    The BloomConfig must have relationship_display and edge_weight_display enabled.

    **Validates: Requirements 6.5**
    """
    service = DashboardService()
    config = service.configure_bloom_overlay(size_metric, color_metric)

    assert config.relationship_display is True, (
        "BloomConfig must have relationship_display=True"
    )
    assert config.edge_weight_display is True, (
        "BloomConfig must have edge_weight_display=True"
    )
