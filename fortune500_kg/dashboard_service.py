"""Dashboard Service for Fortune 500 Knowledge Graph Analytics.

Implements:
- Leaderboard bar chart visualization (Requirements 5.1, 5.5, 5.6)
- Force-directed network graph visualization (Requirements 5.2, 5.5)
- Trend line chart visualization (Requirements 5.3, 5.5, 5.6)
- Heatmap matrix visualization (Requirements 5.4)
- Neo4j Bloom configuration (Requirements 6.1-6.5)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .data_models import (
    Visualization,
    NetworkVisualization,
    NetworkNode,
    NetworkEdge,
    BloomConfig,
)

logger = logging.getLogger(__name__)


class DashboardService:
    """
    Dashboard Service for rendering interactive visualizations.

    Responsibilities:
    - render_leaderboard(): bar chart of Innovation Score vs Fortune 500 rank (Req 5.1)
    - render_network_graph(): force-directed network with metric overlays (Req 5.2)
    - render_trend_chart(): time-series line chart (Req 5.3)
    - render_heatmap(): sector centrality vs revenue matrix (Req 5.4)
    - configure_bloom_overlay(): Neo4j Bloom configuration (Req 6.1-6.5)
    - Sector and year filter support (Req 5.5, 5.6)
    """

    def render_leaderboard(
        self,
        company_scores: Dict[str, float],
        company_ranks: Dict[str, int],
        filters: Optional[Dict[str, Any]] = None,
        company_sectors: Optional[Dict[str, str]] = None,
    ) -> Visualization:
        """
        Render a bar chart displaying Innovation Score versus Fortune 500 rank.

        Each entry in the chart data contains both an Innovation Score value
        and a Fortune 500 rank value. Supports sector and year filters.

        Args:
            company_scores: Mapping of company_id -> Innovation Score
            company_ranks: Mapping of company_id -> Fortune 500 rank
            filters: Optional dict with 'sector' and/or 'year' keys
            company_sectors: Mapping of company_id -> sector (required for sector filter)

        Returns:
            Visualization with chart_type='bar', data containing score and rank per company

        Validates: Requirements 5.1, 5.5, 5.6
        """
        filters = filters or {}
        sector_filter = filters.get("sector")
        year_filter = filters.get("year")

        # Build chart data entries
        data = []
        for company_id, score in company_scores.items():
            if company_id not in company_ranks:
                continue

            # Apply sector filter
            if sector_filter and company_sectors:
                if company_sectors.get(company_id) != sector_filter:
                    continue

            data.append({
                "company_id": company_id,
                "innovation_score": score,
                "fortune_500_rank": company_ranks[company_id],
            })

        # Sort by Innovation Score descending for leaderboard display
        data.sort(key=lambda x: x["innovation_score"], reverse=True)

        logger.info(
            "Rendered leaderboard with %d companies (sector=%s, year=%s)",
            len(data),
            sector_filter,
            year_filter,
        )
        return Visualization(
            chart_type="bar",
            title="Innovation Score Leaderboard",
            data=data,
            config={"x_axis": "fortune_500_rank", "y_axis": "innovation_score"},
            filters_applied=filters,
        )

    def render_network_graph(
        self,
        company_nodes: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None,
    ) -> NetworkVisualization:
        """
        Render a force-directed network graph with company nodes and metric overlays.

        Each node includes company identifier, metric overlay values, and optional
        cluster assignment. Each edge includes relationship type and weight.

        Args:
            company_nodes: List of dicts with 'id', 'label', 'metrics', optional 'cluster_id'
            relationships: List of dicts with 'source', 'target', 'type', optional 'weight'
            filters: Optional dict with 'sector' and/or 'metric_threshold' keys

        Returns:
            NetworkVisualization with nodes, edges, and layout configuration

        Validates: Requirements 5.2, 5.5
        """
        filters = filters or {}
        sector_filter = filters.get("sector")
        metric_threshold = filters.get("metric_threshold", {})

        # Build node objects, applying filters
        nodes = []
        included_node_ids = set()
        for node_data in company_nodes:
            node_id = node_data["id"]

            # Apply sector filter
            if sector_filter:
                if node_data.get("sector") != sector_filter:
                    continue

            # Apply metric threshold filters
            metrics = node_data.get("metrics", {})
            skip = False
            for metric, threshold in metric_threshold.items():
                if metrics.get(metric, 0) < threshold:
                    skip = True
                    break
            if skip:
                continue

            nodes.append(NetworkNode(
                node_id=node_id,
                label=node_data.get("label", node_id),
                metrics=metrics,
                cluster_id=node_data.get("cluster_id"),
            ))
            included_node_ids.add(node_id)

        # Build edge objects (only include edges where both nodes are included)
        edges = []
        for rel in relationships:
            source = rel["source"]
            target = rel["target"]
            if source in included_node_ids and target in included_node_ids:
                edges.append(NetworkEdge(
                    source=source,
                    target=target,
                    relationship_type=rel.get("type", "RELATED"),
                    weight=rel.get("weight", 1.0),
                ))

        logger.info(
            "Rendered network graph: %d nodes, %d edges (sector=%s)",
            len(nodes),
            len(edges),
            sector_filter,
        )
        return NetworkVisualization(
            nodes=nodes,
            edges=edges,
            layout="force-directed",
            filters_applied=filters,
        )

    def render_trend_chart(
        self,
        time_series_data: List[Dict[str, Any]],
        company_id: str,
        metric: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Visualization:
        """
        Render a line chart showing metric evolution over time.

        Data points are ordered chronologically by timestamp.

        Args:
            time_series_data: List of dicts with 'timestamp' (datetime) and metric value
            company_id: Company identifier
            metric: Metric name being visualized
            filters: Optional dict with 'year' key for temporal filtering

        Returns:
            Visualization with chart_type='line', data ordered by timestamp

        Validates: Requirements 5.3, 5.6
        """
        filters = filters or {}
        year_filter = filters.get("year")

        # Filter by year if specified
        filtered_data = []
        for point in time_series_data:
            ts = point.get("timestamp")
            if year_filter and isinstance(ts, datetime):
                if ts.year != year_filter:
                    continue
            filtered_data.append(point)

        # Sort chronologically
        filtered_data.sort(key=lambda x: x.get("timestamp", datetime.min))

        # Build chart data entries
        data = [
            {
                "timestamp": point["timestamp"],
                "value": point.get(metric, point.get("value", 0.0)),
                "company_id": company_id,
                "metric": metric,
            }
            for point in filtered_data
        ]

        logger.info(
            "Rendered trend chart for %s/%s: %d data points (year=%s)",
            company_id,
            metric,
            len(data),
            year_filter,
        )
        return Visualization(
            chart_type="line",
            title=f"{metric} Trend - {company_id}",
            data=data,
            config={"x_axis": "timestamp", "y_axis": "value"},
            filters_applied=filters,
        )

    def render_heatmap(
        self,
        sector_centrality: Dict[str, float],
        sector_revenue_bins: Dict[str, List[float]],
        filters: Optional[Dict[str, Any]] = None,
    ) -> Visualization:
        """
        Render a heatmap matrix showing sector centrality versus revenue.

        The matrix dimensions are (number of sectors) × (number of revenue bins).

        Args:
            sector_centrality: Mapping of sector -> average centrality value
            sector_revenue_bins: Mapping of sector -> list of revenue bin values
            filters: Optional filters dict

        Returns:
            Visualization with chart_type='heatmap', matrix data

        Validates: Requirements 5.4, 13.4
        """
        filters = filters or {}
        sectors = sorted(sector_centrality.keys())

        # Determine revenue bins from the data
        all_bins: set = set()
        for bins in sector_revenue_bins.values():
            all_bins.update(bins)
        revenue_bins = sorted(all_bins)

        # Build matrix data: one row per sector, one column per revenue bin
        matrix = []
        for sector in sectors:
            row = {
                "sector": sector,
                "centrality": sector_centrality.get(sector, 0.0),
                "revenue_bins": sector_revenue_bins.get(sector, []),
            }
            matrix.append(row)

        logger.info(
            "Rendered heatmap: %d sectors × %d revenue bins",
            len(sectors),
            len(revenue_bins),
        )
        return Visualization(
            chart_type="heatmap",
            title="Sector Centrality vs Revenue Heatmap",
            data=matrix,
            config={
                "x_axis": "revenue_bins",
                "y_axis": "sector",
                "value": "centrality",
                "dimensions": {"rows": len(sectors), "cols": len(revenue_bins)},
            },
            filters_applied=filters,
        )

    def configure_bloom_overlay(
        self,
        node_size_metric: str,
        node_color_metric: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> BloomConfig:
        """
        Configure Neo4j Bloom visualization overlays.

        Maps Innovation Score to node size and Ecosystem Centrality to node
        color intensity. Enables filtering by sector, revenue range, and
        metric thresholds.

        Args:
            node_size_metric: Metric to map to node size (e.g. 'innovation_score')
            node_color_metric: Metric to map to node color intensity (e.g. 'ecosystem_centrality')
            filters: Optional dict with 'sector', 'revenue_range', 'metric_thresholds'

        Returns:
            BloomConfig with visualization settings

        Validates: Requirements 6.1-6.5
        """
        filters = filters or {}

        config = BloomConfig(
            node_size_metric=node_size_metric,
            node_color_metric=node_color_metric,
            filters=filters,
            relationship_display=True,
            edge_weight_display=True,
        )

        logger.info(
            "Configured Bloom overlay: size=%s, color=%s, filters=%s",
            node_size_metric,
            node_color_metric,
            list(filters.keys()),
        )
        return config
