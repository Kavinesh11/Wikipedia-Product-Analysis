"""Fortune 500 Analytics Engine

Implements Innovation Score calculation, normalization, decile rankings,
correlation analysis, and in-memory metrics storage for Fortune 500 companies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class InnovationScoreRecord:
    """Stored result of an Innovation Score calculation."""
    company_id: str
    raw_score: float
    normalized_score: float   # 0-10 scale
    decile_rank: int           # 1-10
    github_stars: int
    github_forks: int
    employee_count: int
    timestamp: datetime


@dataclass
class CorrelationResult:
    """Result of a Pearson correlation calculation."""
    coefficient: float
    p_value: float
    confidence_interval: Tuple[float, float]   # (lower, upper) 95% CI
    sample_size: int


@dataclass
class CorrelationRecord:
    """Stored correlation analysis result."""
    metric1: str
    metric2: str
    coefficient: float
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    timestamp: datetime


@dataclass
class PageRankRecord:
    """Stored PageRank result for a single company."""
    company_id: str
    pagerank_score: float
    iterations_run: int
    timestamp: datetime


@dataclass
class LouvainRecord:
    """Stored Louvain community assignment for a single company."""
    company_id: str
    community_id: int
    timestamp: datetime


@dataclass
class EcosystemCentralityRecord:
    """Stored betweenness centrality / ecosystem centrality for a company."""
    company_id: str
    betweenness_centrality: float
    pagerank_score: float
    sector: str
    sector_avg_centrality: float
    timestamp: datetime


@dataclass
class DigitalMaturityRecord:
    """Stored Digital Maturity Index metric."""
    company_id: str
    dmi_value: float          # (stars + forks + contributors) / revenue_rank
    stars: int
    forks: int
    contributors: int
    revenue_rank: int
    sector: str
    sector_avg: float
    quartile: str             # 'top', 'upper_mid', 'lower_mid', 'bottom'
    timestamp: datetime


# ============================================================================
# METRICS REPOSITORY
# ============================================================================

class MetricsRepository:
    """In-memory storage for computed metrics."""

    def __init__(self) -> None:
        self.innovation_scores: List[InnovationScoreRecord] = []
        self.correlations: List[CorrelationRecord] = []
        self.pagerank_records: List[PageRankRecord] = []
        self.louvain_records: List[LouvainRecord] = []
        self.centrality_records: List[EcosystemCentralityRecord] = []
        self.digital_maturity_records: List[DigitalMaturityRecord] = []

    def store_innovation_score(self, record: InnovationScoreRecord) -> None:
        """Append an InnovationScoreRecord to the store."""
        self.innovation_scores.append(record)

    def get_innovation_scores(
        self, company_id: Optional[str] = None
    ) -> List[InnovationScoreRecord]:
        """Return all records, or only those matching *company_id*."""
        if company_id is None:
            return list(self.innovation_scores)
        return [r for r in self.innovation_scores if r.company_id == company_id]

    def store_correlation(self, record: CorrelationRecord) -> None:
        """Append a CorrelationRecord to the store."""
        self.correlations.append(record)

    def get_correlations(self) -> List[CorrelationRecord]:
        """Return all stored correlation records."""
        return list(self.correlations)

    def store_pagerank(self, record: PageRankRecord) -> None:
        """Append a PageRankRecord to the store."""
        self.pagerank_records.append(record)

    def get_pagerank_records(
        self, company_id: Optional[str] = None
    ) -> List[PageRankRecord]:
        """Return all PageRank records, or only those matching *company_id*."""
        if company_id is None:
            return list(self.pagerank_records)
        return [r for r in self.pagerank_records if r.company_id == company_id]

    def store_louvain(self, record: LouvainRecord) -> None:
        """Append a LouvainRecord to the store."""
        self.louvain_records.append(record)

    def get_louvain_records(
        self, company_id: Optional[str] = None
    ) -> List[LouvainRecord]:
        """Return all Louvain records, or only those matching *company_id*."""
        if company_id is None:
            return list(self.louvain_records)
        return [r for r in self.louvain_records if r.company_id == company_id]

    def store_centrality(self, record: EcosystemCentralityRecord) -> None:
        """Append an EcosystemCentralityRecord to the store."""
        self.centrality_records.append(record)

    def get_centrality_records(
        self, company_id: Optional[str] = None
    ) -> List[EcosystemCentralityRecord]:
        """Return all centrality records, or only those matching *company_id*."""
        if company_id is None:
            return list(self.centrality_records)
        return [r for r in self.centrality_records if r.company_id == company_id]

    def store_digital_maturity(self, record: DigitalMaturityRecord) -> None:
        """Append a DigitalMaturityRecord to the store."""
        self.digital_maturity_records.append(record)

    def get_digital_maturity_records(
        self, company_id: Optional[str] = None
    ) -> List[DigitalMaturityRecord]:
        """Return all records, or only those matching *company_id*."""
        if company_id is None:
            return list(self.digital_maturity_records)
        return [r for r in self.digital_maturity_records if r.company_id == company_id]

    # ------------------------------------------------------------------
    # Time-range query support (Req 12.1, 12.2)
    # ------------------------------------------------------------------

    def _all_timestamped_records(self) -> List[object]:
        """Return every stored record that carries a *timestamp* attribute."""
        return (
            self.innovation_scores
            + self.correlations
            + self.pagerank_records
            + self.louvain_records
            + self.centrality_records
            + self.digital_maturity_records
        )

    def get_records_in_range(
        self,
        start: datetime,
        end: datetime,
        record_type: Optional[type] = None,
    ) -> List[object]:
        """Return records whose timestamp falls within [start, end] (inclusive).

        Args:
            start: Lower bound (inclusive).
            end:   Upper bound (inclusive).
            record_type: If provided, filter to only this record type.

        Returns:
            List of matching records ordered by timestamp ascending.
        """
        pool = (
            [r for r in self._all_timestamped_records() if isinstance(r, record_type)]
            if record_type is not None
            else self._all_timestamped_records()
        )
        filtered = [r for r in pool if start <= r.timestamp <= end]  # type: ignore[attr-defined]
        return sorted(filtered, key=lambda r: r.timestamp)  # type: ignore[attr-defined]

    def get_innovation_scores_in_range(
        self,
        start: datetime,
        end: datetime,
        company_id: Optional[str] = None,
    ) -> List[InnovationScoreRecord]:
        """Return InnovationScoreRecords within [start, end], optionally filtered by company."""
        records = [
            r for r in self.innovation_scores
            if start <= r.timestamp <= end
        ]
        if company_id is not None:
            records = [r for r in records if r.company_id == company_id]
        return sorted(records, key=lambda r: r.timestamp)

    def get_digital_maturity_in_range(
        self,
        start: datetime,
        end: datetime,
        company_id: Optional[str] = None,
    ) -> List[DigitalMaturityRecord]:
        """Return DigitalMaturityRecords within [start, end], optionally filtered by company."""
        records = [
            r for r in self.digital_maturity_records
            if start <= r.timestamp <= end
        ]
        if company_id is not None:
            records = [r for r in records if r.company_id == company_id]
        return sorted(records, key=lambda r: r.timestamp)


# ============================================================================
# ANALYTICS ENGINE
# ============================================================================

class AnalyticsEngine:
    """Computes Innovation Scores, normalizations, rankings, and correlations."""

    def __init__(self, repository: Optional[MetricsRepository] = None) -> None:
        self.repository = repository or MetricsRepository()

    # ------------------------------------------------------------------
    # Innovation Score
    # ------------------------------------------------------------------

    def calculate_innovation_score(
        self,
        company_id: str,
        stars: int,
        forks: int,
        employee_count: int,
    ) -> float:
        """Return raw Innovation Score = (stars + forks) / employee_count.

        Raises:
            ValueError: if employee_count <= 0.
        """
        if employee_count <= 0:
            raise ValueError(
                f"employee_count must be > 0, got {employee_count}"
            )
        return (stars + forks) / employee_count

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalize *scores* to the [0, 10] range.

        If all scores are equal (zero range), every normalized score is 0.
        """
        if not scores:
            return {}

        values = np.array(list(scores.values()), dtype=float)
        min_val = values.min()
        max_val = values.max()

        if max_val == min_val:
            return {k: 0.0 for k in scores}

        return {
            k: float((v - min_val) / (max_val - min_val) * 10)
            for k, v in scores.items()
        }

    # ------------------------------------------------------------------
    # Decile Rankings
    # ------------------------------------------------------------------

    def calculate_decile_rankings(
        self, scores: Dict[str, float]
    ) -> Dict[str, int]:
        """Assign decile ranks 1-10 (1 = lowest, 10 = highest).

        Uses percentile-based assignment so that the bottom 10 % of scores
        receive decile 1 and the top 10 % receive decile 10.
        """
        if not scores:
            return {}

        company_ids = list(scores.keys())
        values = np.array([scores[k] for k in company_ids], dtype=float)

        # Compute percentile rank for each value (0-100)
        n = len(values)
        sorted_indices = np.argsort(values, kind="stable")
        ranks = np.empty(n, dtype=int)
        for rank_pos, idx in enumerate(sorted_indices):
            # percentile position: 0-based rank / (n-1) * 100
            if n == 1:
                percentile = 100.0
            else:
                percentile = rank_pos / (n - 1) * 100.0
            # Map percentile to decile 1-10
            decile = min(10, int(percentile / 10) + 1)
            ranks[idx] = decile

        return {company_ids[i]: int(ranks[i]) for i in range(n)}

    # ------------------------------------------------------------------
    # Store Innovation Score
    # ------------------------------------------------------------------

    def store_innovation_score(
        self,
        company_id: str,
        raw_score: float,
        normalized_score: float,
        decile: int,
        stars: int,
        forks: int,
        employee_count: int,
    ) -> InnovationScoreRecord:
        """Create and persist an InnovationScoreRecord with the current timestamp."""
        record = InnovationScoreRecord(
            company_id=company_id,
            raw_score=raw_score,
            normalized_score=normalized_score,
            decile_rank=decile,
            github_stars=stars,
            github_forks=forks,
            employee_count=employee_count,
            timestamp=datetime.utcnow(),
        )
        self.repository.store_innovation_score(record)
        return record

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def calculate_correlation(
        self,
        values1: List[float],
        values2: List[float],
    ) -> CorrelationResult:
        """Compute Pearson correlation with 95 % CI via Fisher z-transformation.

        Raises:
            ValueError: if fewer than 3 data points are provided.
        """
        n = len(values1)
        if n < 3 or len(values2) < 3:
            raise ValueError(
                "At least 3 data points are required for correlation."
            )
        if n != len(values2):
            raise ValueError(
                "Both value lists must have the same length."
            )

        r, p_value = stats.pearsonr(values1, values2)

        # Fisher z-transformation for 95 % CI
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        ci_lower = float(np.tanh(z - 1.96 * se))
        ci_upper = float(np.tanh(z + 1.96 * se))

        return CorrelationResult(
            coefficient=float(r),
            p_value=float(p_value),
            confidence_interval=(ci_lower, ci_upper),
            sample_size=n,
        )

    # ------------------------------------------------------------------
    # Store Correlation
    # ------------------------------------------------------------------

    def store_correlation(
        self,
        metric1: str,
        metric2: str,
        result: CorrelationResult,
    ) -> CorrelationRecord:
        """Create and persist a CorrelationRecord with the current timestamp."""
        record = CorrelationRecord(
            metric1=metric1,
            metric2=metric2,
            coefficient=result.coefficient,
            p_value=result.p_value,
            confidence_interval=result.confidence_interval,
            sample_size=result.sample_size,
            timestamp=datetime.utcnow(),
        )
        self.repository.store_correlation(record)
        return record

    # ------------------------------------------------------------------
    # PageRank
    # ------------------------------------------------------------------

    def execute_pagerank(
        self,
        graph: Dict[str, List[str]],
        max_iterations: int = 20,
        damping: float = 0.85,
        tol: float = 1.0e-6,
    ) -> Dict[str, float]:
        """Execute PageRank on an adjacency-list graph (no Neo4j required).

        Args:
            graph: Mapping of company_id -> list of neighbour company_ids.
            max_iterations: Hard cap on iteration count (≤ 20 per Req 3.1).
            damping: Damping factor (standard 0.85).
            tol: Convergence tolerance.

        Returns:
            Dictionary mapping company_id to PageRank score.

        Raises:
            ValueError: if max_iterations > 20.
        """
        if max_iterations > 20:
            raise ValueError(
                f"max_iterations must be ≤ 20 (Req 3.1), got {max_iterations}"
            )

        nodes = list(graph.keys())
        n = len(nodes)
        if n == 0:
            return {}

        idx = {node: i for i, node in enumerate(nodes)}
        scores = np.full(n, 1.0 / n)
        iterations_run = 0

        for iteration in range(max_iterations):
            new_scores = np.full(n, (1.0 - damping) / n)
            for node, neighbours in graph.items():
                if not neighbours:
                    # Dangling node: distribute evenly
                    new_scores += damping * scores[idx[node]] / n
                else:
                    share = damping * scores[idx[node]] / len(neighbours)
                    for nb in neighbours:
                        if nb in idx:
                            new_scores[idx[nb]] += share
            iterations_run = iteration + 1
            if np.linalg.norm(new_scores - scores, 1) < tol:
                scores = new_scores
                break
            scores = new_scores

        result: Dict[str, float] = {
            nodes[i]: float(scores[i]) for i in range(n)
        }

        # Persist to repository (Req 3.4)
        ts = datetime.utcnow()
        for company_id, score in result.items():
            self.repository.store_pagerank(
                PageRankRecord(
                    company_id=company_id,
                    pagerank_score=score,
                    iterations_run=iterations_run,
                    timestamp=ts,
                )
            )

        return result

    # ------------------------------------------------------------------
    # Louvain community detection
    # ------------------------------------------------------------------

    def execute_louvain(
        self,
        graph: Dict[str, List[str]],
    ) -> Dict[str, int]:
        """Assign every company node to exactly one community (Req 3.2).

        Uses a greedy modularity-based approach implemented with NetworkX so
        that no external Neo4j GDS dependency is required for unit/property
        testing.  In production the caller may substitute Neo4j GDS results
        directly via the repository.

        Args:
            graph: Mapping of company_id -> list of neighbour company_ids.

        Returns:
            Dictionary mapping company_id to integer community_id.
        """
        import networkx as nx
        from networkx.algorithms import community as nx_community

        nodes = list(graph.keys())
        if not nodes:
            return {}

        G = nx.Graph()
        G.add_nodes_from(nodes)
        for node, neighbours in graph.items():
            for nb in neighbours:
                if nb in graph:
                    G.add_edge(node, nb)

        # Louvain via NetworkX (greedy modularity as fallback for isolated nodes)
        try:
            communities = nx_community.louvain_communities(G, seed=42)
        except Exception:
            # Fallback: each node is its own community
            communities = [{node} for node in nodes]

        result: Dict[str, int] = {}
        for community_id, members in enumerate(communities):
            for member in members:
                result[member] = community_id

        # Ensure every node has an assignment (handles isolated nodes)
        for node in nodes:
            if node not in result:
                result[node] = len(communities)

        # Persist to repository (Req 3.4)
        ts = datetime.utcnow()
        for company_id, community_id in result.items():
            self.repository.store_louvain(
                LouvainRecord(
                    company_id=company_id,
                    community_id=community_id,
                    timestamp=ts,
                )
            )

        return result

    # ------------------------------------------------------------------
    # Betweenness Centrality / Ecosystem Centrality
    # ------------------------------------------------------------------

    def calculate_betweenness_centrality(
        self,
        graph: Dict[str, List[str]],
        company_sectors: Dict[str, str],
        top_n: int = 10,
    ) -> Dict[str, float]:
        """Calculate betweenness centrality for top-N nodes per company (Req 3.3).

        Args:
            graph: Mapping of company_id -> list of neighbour company_ids.
            company_sectors: Mapping of company_id -> sector name.
            top_n: Maximum number of top-connected nodes to analyse per company.

        Returns:
            Dictionary mapping company_id to betweenness centrality score.
        """
        import networkx as nx

        nodes = list(graph.keys())
        if not nodes:
            return {}

        G = nx.Graph()
        G.add_nodes_from(nodes)
        for node, neighbours in graph.items():
            for nb in neighbours:
                if nb in graph:
                    G.add_edge(node, nb)

        # Full betweenness centrality on the graph
        full_centrality: Dict[str, float] = nx.betweenness_centrality(G)

        # For each company, restrict analysis to its top-N neighbours by degree
        result: Dict[str, float] = {}
        for company_id in nodes:
            neighbours = list(G.neighbors(company_id))
            # Select top-N neighbours by their own centrality (proxy for connectivity)
            top_neighbours = sorted(
                neighbours,
                key=lambda nb: full_centrality.get(nb, 0.0),
                reverse=True,
            )[:top_n]
            # Ecosystem Centrality = average centrality of top-N neighbours + own
            relevant = [company_id] + top_neighbours
            result[company_id] = float(
                np.mean([full_centrality.get(n, 0.0) for n in relevant])
            )

        # Compute sector-level averages (Req 3.5)
        sector_scores: Dict[str, List[float]] = {}
        for company_id, score in result.items():
            sector = company_sectors.get(company_id, "Unknown")
            sector_scores.setdefault(sector, []).append(score)

        sector_avg: Dict[str, float] = {
            s: float(np.mean(v)) for s, v in sector_scores.items()
        }

        # Persist EcosystemCentralityRecord for each company (Req 3.4)
        ts = datetime.utcnow()
        for company_id, score in result.items():
            sector = company_sectors.get(company_id, "Unknown")
            self.repository.store_centrality(
                EcosystemCentralityRecord(
                    company_id=company_id,
                    betweenness_centrality=score,
                    pagerank_score=0.0,   # filled by caller if PageRank already run
                    sector=sector,
                    sector_avg_centrality=sector_avg.get(sector, 0.0),
                    timestamp=ts,
                )
            )

        return result

    # ------------------------------------------------------------------
    # Digital Maturity Index
    # ------------------------------------------------------------------

    def calculate_digital_maturity_index(
        self,
        company_id: str,
        stars: int,
        forks: int,
        contributors: int,
        revenue_rank: int,
    ) -> float:
        """Return Digital Maturity Index = (stars + forks + contributors) / revenue_rank.

        Raises:
            ValueError: if revenue_rank <= 0.
        """
        if revenue_rank <= 0:
            raise ValueError(
                f"revenue_rank must be > 0, got {revenue_rank}"
            )
        return (stars + forks + contributors) / revenue_rank

    def calculate_sector_digital_maturity(
        self,
        company_metrics: Dict[str, Dict],
    ) -> Dict[str, float]:
        """Calculate sector-level average Digital Maturity Index.

        Args:
            company_metrics: Mapping of company_id -> dict with keys:
                stars, forks, contributors, revenue_rank, sector

        Returns:
            Mapping of sector -> average DMI value.
        """
        sector_values: Dict[str, List[float]] = {}
        for company_id, metrics in company_metrics.items():
            dmi = self.calculate_digital_maturity_index(
                company_id=company_id,
                stars=metrics["stars"],
                forks=metrics["forks"],
                contributors=metrics["contributors"],
                revenue_rank=metrics["revenue_rank"],
            )
            sector = metrics["sector"]
            sector_values.setdefault(sector, []).append(dmi)

        return {
            sector: float(np.mean(values))
            for sector, values in sector_values.items()
        }

    def calculate_sector_gap_percentage(
        self,
        sector_a_avg: float,
        sector_b_avg: float,
    ) -> float:
        """Calculate percentage gap between two sector averages.

        Returns ((A - B) / B) * 100.

        Raises:
            ValueError: if sector_b_avg == 0.
        """
        if sector_b_avg == 0:
            raise ValueError("sector_b_avg must be non-zero for gap calculation.")
        return ((sector_a_avg - sector_b_avg) / sector_b_avg) * 100.0

    def identify_bottom_quartile(
        self,
        company_dmi_values: Dict[str, float],
    ) -> List[str]:
        """Identify companies in the bottom quartile by Digital Maturity Index.

        Returns the floor(N/4) companies with the lowest DMI values.

        Args:
            company_dmi_values: Mapping of company_id -> DMI value.

        Returns:
            List of company_ids in the bottom quartile (sorted ascending by DMI).
        """
        if not company_dmi_values:
            return []
        n = len(company_dmi_values)
        bottom_count = n // 4
        sorted_companies = sorted(company_dmi_values.items(), key=lambda x: x[1])
        return [cid for cid, _ in sorted_companies[:bottom_count]]

    def store_digital_maturity(
        self,
        company_id: str,
        dmi_value: float,
        stars: int,
        forks: int,
        contributors: int,
        revenue_rank: int,
        sector: str,
        sector_avg: float,
        quartile: str,
    ) -> "DigitalMaturityRecord":
        """Create and persist a DigitalMaturityRecord with the current timestamp."""
        record = DigitalMaturityRecord(
            company_id=company_id,
            dmi_value=dmi_value,
            stars=stars,
            forks=forks,
            contributors=contributors,
            revenue_rank=revenue_rank,
            sector=sector,
            sector_avg=sector_avg,
            quartile=quartile,
            timestamp=datetime.utcnow(),
        )
        self.repository.store_digital_maturity(record)
        return record

    # ------------------------------------------------------------------
    # Historical Trend Analysis (Req 12.3, 12.4, 12.5)
    # ------------------------------------------------------------------

    def calculate_yoy_growth_rate(
        self,
        v_current: float,
        v_previous: float,
    ) -> float:
        """Calculate year-over-year growth rate as a percentage.

        Formula: ((V_current - V_previous) / V_previous) * 100

        Args:
            v_current:  Metric value for the current period.
            v_previous: Metric value for the previous period.

        Returns:
            Growth rate as a percentage (can be negative).

        Raises:
            ValueError: if v_previous is zero.
        """
        if v_previous == 0:
            raise ValueError("v_previous must be non-zero for YoY growth calculation.")
        return ((v_current - v_previous) / v_previous) * 100.0

    def calculate_yoy_growth_series(
        self,
        time_series: List[Tuple[datetime, float]],
    ) -> List[Tuple[datetime, float]]:
        """Calculate YoY growth rates for a time-ordered series of (timestamp, value) pairs.

        Each output entry represents the growth rate from the previous data point
        to the current one.  The first data point has no predecessor and is omitted.

        Args:
            time_series: List of (timestamp, value) tuples, need not be pre-sorted.

        Returns:
            List of (timestamp, growth_rate_pct) tuples sorted by timestamp ascending,
            starting from the second data point.
        """
        if len(time_series) < 2:
            return []

        sorted_series = sorted(time_series, key=lambda x: x[0])
        result: List[Tuple[datetime, float]] = []
        for i in range(1, len(sorted_series)):
            ts, v_current = sorted_series[i]
            _, v_previous = sorted_series[i - 1]
            if v_previous == 0:
                continue  # skip undefined growth
            rate = self.calculate_yoy_growth_rate(v_current, v_previous)
            result.append((ts, rate))
        return result

    def identify_inflection_points(
        self,
        time_series: List[Tuple[datetime, float]],
    ) -> List[Tuple[datetime, str]]:
        """Identify timestamps where the trend changes direction.

        A direction change occurs when consecutive differences switch sign
        (positive → negative or negative → positive).

        Args:
            time_series: List of (timestamp, value) tuples.

        Returns:
            List of (timestamp, direction_change) tuples where direction_change is
            'increasing_to_decreasing' or 'decreasing_to_increasing'.
        """
        if len(time_series) < 3:
            return []

        sorted_series = sorted(time_series, key=lambda x: x[0])
        inflections: List[Tuple[datetime, str]] = []

        for i in range(1, len(sorted_series) - 1):
            _, v_prev = sorted_series[i - 1]
            ts, v_curr = sorted_series[i]
            _, v_next = sorted_series[i + 1]

            delta_before = v_curr - v_prev
            delta_after = v_next - v_curr

            if delta_before > 0 and delta_after < 0:
                inflections.append((ts, "increasing_to_decreasing"))
            elif delta_before < 0 and delta_after > 0:
                inflections.append((ts, "decreasing_to_increasing"))

        return inflections

    def get_time_series(
        self,
        company_id: str,
        metric: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Tuple[datetime, float]]:
        """Retrieve a time-ordered series of metric values for a company.

        Args:
            company_id: Company identifier.
            metric:     One of 'innovation_score' or 'digital_maturity'.
            start:      Optional lower bound (inclusive).
            end:        Optional upper bound (inclusive).

        Returns:
            List of (timestamp, value) tuples sorted by timestamp ascending.

        Raises:
            ValueError: if metric is not recognised.
        """
        if metric == "innovation_score":
            records = self.repository.get_innovation_scores(company_id)
            series = [(r.timestamp, r.normalized_score) for r in records]
        elif metric == "digital_maturity":
            records = self.repository.get_digital_maturity_records(company_id)
            series = [(r.timestamp, r.dmi_value) for r in records]
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use 'innovation_score' or 'digital_maturity'.")

        if start is not None:
            series = [(ts, v) for ts, v in series if ts >= start]
        if end is not None:
            series = [(ts, v) for ts, v in series if ts <= end]

        return sorted(series, key=lambda x: x[0])
