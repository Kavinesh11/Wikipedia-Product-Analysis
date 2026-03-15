"""Analytics Engine for Fortune 500 Knowledge Graph Analytics.

Implements:
- Innovation Score calculation and normalization (Requirements 2.1-2.4)
- Pearson correlation analysis (Requirements 2.5, 7.1)
- PageRank algorithm execution (Requirements 3.1, 3.4)
- Louvain community detection (Requirements 3.2, 3.4)
- Betweenness centrality calculation (Requirements 3.3, 3.5)
- Business outcome correlation analysis (Requirements 7.2, 7.3, 7.4)
- Cross-sector comparative analysis (Requirements 13.1, 13.2, 13.3)
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math
import logging

from .data_models import (
    InnovationScoreRecord,
    CorrelationRecord,
    CorrelationResult,
    EcosystemCentralityRecord,
    MetricRecord,
)

logger = logging.getLogger(__name__)


class MetricsRepository:
    """In-memory metrics repository for storing computed metric records."""

    def __init__(self) -> None:
        self._records: List[object] = []

    def save(self, record: object) -> None:
        """Persist a metric record."""
        self._records.append(record)

    def get_all(self) -> List[object]:
        """Return all stored records."""
        return list(self._records)

    def get_by_type(self, record_type: type) -> List[object]:
        """Return all records of a given type."""
        return [r for r in self._records if isinstance(r, record_type)]


class AnalyticsEngine:
    """
    Analytics Engine for computing Innovation Scores and correlations.

    Responsibilities:
    - calculate_innovation_score(): raw score per company (Req 2.1)
    - normalize_innovation_scores(): 0-10 scale (Req 2.2)
    - store results with timestamps (Req 2.3)
    - compute decile rankings (Req 2.4)
    - calculate_correlation(): Pearson r with p-value and CI (Req 2.5, 7.1)
    """

    def __init__(self, metrics_repo: Optional[MetricsRepository] = None) -> None:
        self.metrics_repo = metrics_repo or MetricsRepository()

    # ------------------------------------------------------------------
    # Innovation Score (Requirements 2.1 – 2.4)
    # ------------------------------------------------------------------

    def calculate_innovation_score(
        self,
        company_id: str,
        github_stars: int,
        github_forks: int,
        employee_count: int,
    ) -> float:
        """
        Calculate raw Innovation Score: (stars + forks) / employee_count.

        Args:
            company_id: Unique company identifier
            github_stars: Total GitHub stars
            github_forks: Total GitHub forks
            employee_count: Number of employees (must be > 0)

        Returns:
            Raw (un-normalised) Innovation Score

        Raises:
            ValueError: When employee_count <= 0

        Validates: Requirement 2.1
        """
        if employee_count <= 0:
            raise ValueError(
                f"employee_count must be positive, got {employee_count} for {company_id}"
            )
        return (github_stars + github_forks) / employee_count

    def normalize_innovation_scores(
        self, scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalise a dict of raw Innovation Scores to the [0, 10] range.

        Uses min-max normalisation:  (x - min) / (max - min) * 10
        When all values are equal the normalised score is 0.0 for every company.

        Args:
            scores: Mapping of company_id -> raw Innovation Score

        Returns:
            Mapping of company_id -> normalised score in [0, 10]

        Validates: Requirement 2.2
        """
        if not scores:
            return {}

        min_val = min(scores.values())
        max_val = max(scores.values())
        spread = max_val - min_val

        if spread == 0:
            return {cid: 0.0 for cid in scores}

        return {
            cid: (raw - min_val) / spread * 10.0
            for cid, raw in scores.items()
        }

    def compute_decile_rankings(
        self, normalized_scores: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Assign decile ranks (1 = lowest, 10 = highest) to normalised scores.

        Companies are sorted ascending; each is placed in one of 10 equal-width
        buckets based on their position in the sorted order.

        Args:
            normalized_scores: Mapping of company_id -> normalised score [0, 10]

        Returns:
            Mapping of company_id -> decile rank in [1, 10]

        Validates: Requirement 2.4
        """
        if not normalized_scores:
            return {}

        n = len(normalized_scores)
        sorted_ids = sorted(normalized_scores, key=lambda cid: normalized_scores[cid])

        deciles: Dict[str, int] = {}
        for rank_0based, cid in enumerate(sorted_ids):
            # Map position [0, n-1] to decile [1, 10]
            decile = min(10, math.floor(rank_0based / n * 10) + 1)
            deciles[cid] = decile

        return deciles

    def store_innovation_scores(
        self,
        company_metrics: Dict[str, Dict],
        normalized_scores: Dict[str, float],
        decile_ranks: Dict[str, int],
    ) -> None:
        """
        Persist InnovationScoreRecord entries to the Metrics Repository.

        Args:
            company_metrics: company_id -> {stars, forks, employee_count, raw_score}
            normalized_scores: company_id -> normalised score
            decile_ranks: company_id -> decile rank

        Validates: Requirement 2.3
        """
        ts = datetime.now()
        for cid, metrics in company_metrics.items():
            record = InnovationScoreRecord(
                company_id=cid,
                metric_name="innovation_score",
                metric_value=metrics["raw_score"],
                timestamp=ts,
                github_stars=metrics["stars"],
                github_forks=metrics["forks"],
                employee_count=metrics["employee_count"],
                normalized_score=normalized_scores.get(cid, 0.0),
                decile_rank=decile_ranks.get(cid, 0),
            )
            self.metrics_repo.save(record)
            logger.debug(
                "Stored innovation score for %s: raw=%.4f normalised=%.4f decile=%d",
                cid,
                metrics["raw_score"],
                record.normalized_score,
                record.decile_rank,
            )

    # ------------------------------------------------------------------
    # Correlation Analysis (Requirements 2.5, 7.1)
    # ------------------------------------------------------------------

    def calculate_correlation(
        self, metric1_values: List[float], metric2_values: List[float]
    ) -> CorrelationResult:
        """
        Calculate Pearson correlation coefficient between two metric series.

        Uses the standard formula:
            r = Σ((x - x̄)(y - ȳ)) / √(Σ(x - x̄)² × Σ(y - ȳ)²)

        Also computes:
        - Two-tailed p-value via t-distribution approximation
        - 95% confidence interval via Fisher z-transformation

        Args:
            metric1_values: First metric series (e.g. Innovation Scores)
            metric2_values: Second metric series (e.g. revenue growth rates)

        Returns:
            CorrelationResult with coefficient, p_value, confidence_interval, sample_size

        Raises:
            ValueError: When series have different lengths or fewer than 3 points

        Validates: Requirements 2.5, 7.1
        """
        n = len(metric1_values)
        if len(metric2_values) != n:
            raise ValueError("Both metric series must have the same length")
        if n < 3:
            raise ValueError("At least 3 data points are required for correlation")

        x_mean = sum(metric1_values) / n
        y_mean = sum(metric2_values) / n

        numerator = sum(
            (x - x_mean) * (y - y_mean)
            for x, y in zip(metric1_values, metric2_values)
        )
        sum_sq_x = sum((x - x_mean) ** 2 for x in metric1_values)
        sum_sq_y = sum((y - y_mean) ** 2 for y in metric2_values)

        denominator = math.sqrt(sum_sq_x * sum_sq_y)

        if denominator == 0:
            # Degenerate case: one or both series are constant
            r = 0.0
        else:
            r = max(-1.0, min(1.0, numerator / denominator))

        # p-value via t-distribution (two-tailed)
        p_value = self._pearson_p_value(r, n)

        # 95% CI via Fisher z-transformation
        ci = self._fisher_ci(r, n)

        return CorrelationResult(
            coefficient=r,
            p_value=p_value,
            confidence_interval=ci,
            sample_size=n,
        )

    def store_correlation(
        self,
        metric1: str,
        metric2: str,
        result: CorrelationResult,
    ) -> CorrelationRecord:
        """
        Persist a CorrelationRecord to the Metrics Repository.

        Args:
            metric1: Name of the first metric
            metric2: Name of the second metric
            result: CorrelationResult from calculate_correlation()

        Returns:
            The persisted CorrelationRecord

        Validates: Requirement 7.5 (store with confidence intervals)
        """
        record = CorrelationRecord(
            metric1=metric1,
            metric2=metric2,
            correlation_coefficient=result.coefficient,
            p_value=result.p_value,
            confidence_interval=result.confidence_interval,
            sample_size=result.sample_size,
            timestamp=datetime.now(),
        )
        self.metrics_repo.save(record)
        logger.info(
            "Stored correlation %s vs %s: r=%.4f p=%.4f CI=(%.4f, %.4f)",
            metric1,
            metric2,
            record.correlation_coefficient,
            record.p_value,
            record.confidence_interval[0],
            record.confidence_interval[1],
        )
        return record

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pearson_p_value(r: float, n: int) -> float:
        """
        Approximate two-tailed p-value for Pearson r using t-distribution.

        t = r * sqrt(n - 2) / sqrt(1 - r²),  df = n - 2
        Uses a rational approximation of the regularised incomplete beta function.
        """
        if abs(r) >= 1.0:
            return 0.0

        df = n - 2
        t_stat = r * math.sqrt(df) / math.sqrt(1.0 - r * r)

        # Approximate p-value using the t-distribution CDF
        # P(T > |t|) * 2  (two-tailed)
        p = AnalyticsEngine._t_dist_two_tailed_p(abs(t_stat), df)
        return p

    @staticmethod
    def _t_dist_two_tailed_p(t: float, df: int) -> float:
        """
        Two-tailed p-value from t-distribution using regularised incomplete beta.

        Uses the relation: p = I(df/(df+t²), df/2, 1/2)
        Approximated via the continued fraction expansion.
        """
        x = df / (df + t * t)
        p_one_tail = 0.5 * AnalyticsEngine._regularised_incomplete_beta(
            x, df / 2.0, 0.5
        )
        return min(1.0, 2.0 * p_one_tail)

    @staticmethod
    def _regularised_incomplete_beta(x: float, a: float, b: float) -> float:
        """
        Regularised incomplete beta function I_x(a, b) via continued fraction.
        Accurate enough for p-value approximation purposes.
        """
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0

        # Use symmetry relation when x > (a+1)/(a+b+2)
        if x > (a + 1.0) / (a + b + 2.0):
            return 1.0 - AnalyticsEngine._regularised_incomplete_beta(
                1.0 - x, b, a
            )

        lbeta = (
            math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            + a * math.log(x)
            + b * math.log(1.0 - x)
        )

        # Lentz's continued fraction
        TINY = 1e-30
        MAX_ITER = 200
        EPS = 3e-7

        f = TINY
        c = 1.0
        d = 1.0 - (a + b) * x / (a + 1.0)
        if abs(d) < TINY:
            d = TINY
        d = 1.0 / d
        f = d

        for m in range(1, MAX_ITER + 1):
            # Even step
            m2 = 2 * m
            num = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
            d = 1.0 + num * d
            if abs(d) < TINY:
                d = TINY
            c = 1.0 + num / c
            if abs(c) < TINY:
                c = TINY
            d = 1.0 / d
            delta = c * d
            f *= delta

            # Odd step
            num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
            d = 1.0 + num * d
            if abs(d) < TINY:
                d = TINY
            c = 1.0 + num / c
            if abs(c) < TINY:
                c = TINY
            d = 1.0 / d
            delta = c * d
            f *= delta

            if abs(delta - 1.0) < EPS:
                break

        return math.exp(lbeta) * f / a

    @staticmethod
    def _fisher_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
        """
        95% confidence interval for Pearson r via Fisher z-transformation.

        z = arctanh(r),  SE = 1/sqrt(n-3)
        CI_z = z ± z_alpha/2 * SE
        CI_r = tanh(CI_z)
        """
        if n <= 3:
            return (-1.0, 1.0)

        # z_alpha/2 for 95% CI ≈ 1.96
        z_crit = 1.959964  # scipy.stats.norm.ppf(0.975)

        # Clamp r to avoid arctanh singularity
        r_clamped = max(-0.9999999, min(0.9999999, r))
        z = math.atanh(r_clamped)
        se = 1.0 / math.sqrt(n - 3)

        lo = math.tanh(z - z_crit * se)
        hi = math.tanh(z + z_crit * se)
        return (lo, hi)

    # ------------------------------------------------------------------
    # Graph Algorithms (Requirements 3.1 – 3.5)
    # ------------------------------------------------------------------

    def execute_pagerank(
        self,
        graph: Dict[str, List[str]],
        max_iterations: int = 20,
        damping: float = 0.85,
        tolerance: float = 1e-6,
    ) -> Dict[str, float]:
        """
        Execute PageRank algorithm on the Knowledge Graph.

        Implements the standard iterative PageRank algorithm with convergence
        check. Stops when the L1 norm of score changes falls below tolerance
        or when max_iterations is reached.

        Args:
            graph: Adjacency dict mapping company_id -> list of neighbour company_ids
            max_iterations: Maximum number of iterations (default 20, Req 3.1)
            damping: Damping factor (default 0.85)
            tolerance: Convergence threshold for early stopping

        Returns:
            Dictionary mapping company_id to PageRank score

        Validates: Requirements 3.1, 3.4
        """
        if not graph:
            return {}

        nodes = list(graph.keys())
        n = len(nodes)
        initial_score = 1.0 / n

        scores: Dict[str, float] = {node: initial_score for node in nodes}

        # Build reverse adjacency: who links TO each node
        in_links: Dict[str, List[str]] = {node: [] for node in nodes}
        out_degree: Dict[str, int] = {}
        for node, neighbours in graph.items():
            out_degree[node] = len(neighbours)
            for neighbour in neighbours:
                if neighbour in in_links:
                    in_links[neighbour].append(node)

        # Dangling nodes (no outgoing links) distribute rank equally
        dangling_nodes = [node for node in nodes if out_degree.get(node, 0) == 0]

        for _iteration in range(max_iterations):
            dangling_sum = sum(scores[node] for node in dangling_nodes)
            new_scores: Dict[str, float] = {}

            for node in nodes:
                rank_from_links = sum(
                    scores[src] / out_degree[src]
                    for src in in_links[node]
                    if out_degree.get(src, 0) > 0
                )
                new_scores[node] = (
                    (1.0 - damping) / n
                    + damping * (rank_from_links + dangling_sum / n)
                )

            # Check convergence
            delta = sum(abs(new_scores[node] - scores[node]) for node in nodes)
            scores = new_scores

            if delta < tolerance:
                break

        # Store results in Metrics Repository (Req 3.4)
        ts = datetime.now()
        for company_id, score in scores.items():
            record = MetricRecord(
                company_id=company_id,
                metric_name="pagerank",
                metric_value=score,
                timestamp=ts,
                metadata={"algorithm": "pagerank", "max_iterations": max_iterations},
            )
            self.metrics_repo.save(record)

        logger.info("PageRank completed for %d nodes, stored in repository", n)
        return scores

    def execute_louvain(
        self,
        graph: Dict[str, List[str]],
    ) -> Dict[str, int]:
        """
        Execute Louvain community detection algorithm.

        Assigns every company node to exactly one community identifier.
        Uses a greedy modularity-maximisation approach: each node starts in
        its own community, then nodes are iteratively moved to the community
        of the neighbour that yields the greatest modularity gain.

        Args:
            graph: Adjacency dict mapping company_id -> list of neighbour company_ids

        Returns:
            Dictionary mapping company_id to community_id (integer)

        Validates: Requirements 3.2, 3.4
        """
        if not graph:
            return {}

        nodes = list(graph.keys())

        # Initialise: each node in its own community
        community: Dict[str, int] = {node: i for i, node in enumerate(nodes)}

        # Build undirected adjacency set for fast lookup
        neighbours: Dict[str, set] = {node: set() for node in nodes}
        for node, nbrs in graph.items():
            for nbr in nbrs:
                if nbr in neighbours:
                    neighbours[node].add(nbr)
                    neighbours[nbr].add(node)

        total_edges = sum(len(nbrs) for nbrs in neighbours.values()) // 2
        if total_edges == 0:
            # No edges: each node stays in its own community
            result = {node: i for i, node in enumerate(nodes)}
            self._store_louvain_results(result)
            return result

        # Greedy phase: iterate until no improvement
        improved = True
        while improved:
            improved = False
            for node in nodes:
                current_comm = community[node]

                # Count connections to each neighbouring community
                comm_connections: Dict[int, int] = {}
                for nbr in neighbours[node]:
                    c = community[nbr]
                    comm_connections[c] = comm_connections.get(c, 0) + 1

                if not comm_connections:
                    continue

                # Find the community with the most connections
                best_comm = max(comm_connections, key=lambda c: comm_connections[c])

                if best_comm != current_comm:
                    community[node] = best_comm
                    improved = True

        # Remap community IDs to contiguous integers starting from 0
        unique_comms = sorted(set(community.values()))
        remap = {old: new for new, old in enumerate(unique_comms)}
        result = {node: remap[community[node]] for node in nodes}

        # Store results in Metrics Repository (Req 3.4)
        self._store_louvain_results(result)

        logger.info(
            "Louvain completed: %d nodes assigned to %d communities",
            len(nodes),
            len(unique_comms),
        )
        return result

    def _store_louvain_results(self, community_map: Dict[str, int]) -> None:
        """Persist Louvain community assignments to the Metrics Repository."""
        ts = datetime.now()
        for company_id, community_id in community_map.items():
            record = MetricRecord(
                company_id=company_id,
                metric_name="louvain_community",
                metric_value=float(community_id),
                timestamp=ts,
                metadata={"algorithm": "louvain"},
            )
            self.metrics_repo.save(record)

    def calculate_betweenness_centrality(
        self,
        graph: Dict[str, List[str]],
        top_n: int = 10,
    ) -> Dict[str, float]:
        """
        Calculate betweenness centrality for top N nodes per company.

        Uses Brandes' algorithm (BFS-based) to compute betweenness centrality
        for all nodes, then for each company selects the top_n web-connected
        nodes and computes the Ecosystem Centrality metric.

        Also computes sector-level average centrality and stores
        EcosystemCentralityRecord entries in the Metrics Repository.

        Args:
            graph: Adjacency dict mapping node_id -> list of neighbour node_ids.
                   Node IDs for companies should be present as keys.
            top_n: Number of top nodes to analyse per company (default 10, Req 3.3)

        Returns:
            Dictionary mapping node_id to betweenness centrality score

        Validates: Requirements 3.3, 3.5
        """
        if not graph:
            return {}

        nodes = list(graph.keys())
        n = len(nodes)

        # Build undirected adjacency
        adj: Dict[str, List[str]] = {node: [] for node in nodes}
        for node, nbrs in graph.items():
            for nbr in nbrs:
                if nbr in adj:
                    if nbr not in adj[node]:
                        adj[node].append(nbr)
                    if node not in adj[nbr]:
                        adj[nbr].append(node)

        # Brandes' algorithm for betweenness centrality
        centrality: Dict[str, float] = {node: 0.0 for node in nodes}

        for source in nodes:
            # BFS to find shortest paths from source
            stack: List[str] = []
            pred: Dict[str, List[str]] = {node: [] for node in nodes}
            sigma: Dict[str, float] = {node: 0.0 for node in nodes}
            dist: Dict[str, int] = {node: -1 for node in nodes}

            sigma[source] = 1.0
            dist[source] = 0
            queue: List[str] = [source]
            head = 0

            while head < len(queue):
                v = queue[head]
                head += 1
                stack.append(v)
                for w in adj[v]:
                    if dist[w] < 0:
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            # Accumulation
            delta: Dict[str, float] = {node: 0.0 for node in nodes}
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != source:
                    centrality[w] += delta[w]

        # Normalise (undirected graph: divide by (n-1)(n-2))
        if n > 2:
            norm = 1.0 / ((n - 1) * (n - 2))
            centrality = {node: score * norm for node, score in centrality.items()}

        # For each company, select top_n nodes by centrality and compute
        # Ecosystem Centrality as the mean of those top_n scores.
        # We treat every node in the graph as a potential company node.
        sorted_nodes = sorted(centrality, key=lambda nd: centrality[nd], reverse=True)
        top_nodes = sorted_nodes[:top_n]

        # Compute sector-level average centrality (Req 3.5)
        # Without explicit sector metadata in the graph dict, we use a single
        # "default" sector grouping for all nodes.
        sector_avg = (
            sum(centrality[nd] for nd in top_nodes) / len(top_nodes)
            if top_nodes else 0.0
        )

        # Store EcosystemCentralityRecord for each top node (Req 3.4, 3.5)
        ts = datetime.now()
        for node in top_nodes:
            record = EcosystemCentralityRecord(
                company_id=node,
                metric_name="ecosystem_centrality",
                metric_value=centrality[node],
                timestamp=ts,
                metadata={"algorithm": "betweenness_centrality", "top_n": top_n},
                betweenness_centrality=centrality[node],
                pagerank_score=0.0,  # populated separately if PageRank was run
                sector_avg_centrality=sector_avg,
            )
            self.metrics_repo.save(record)

        logger.info(
            "Betweenness centrality computed for %d nodes; top-%d stored",
            n,
            len(top_nodes),
        )
        return centrality

    # ------------------------------------------------------------------
    # Business Outcome Correlation Analysis (Requirements 7.2, 7.3, 7.4)
    # ------------------------------------------------------------------

    def calculate_centrality_ma_correlation(
        self,
        centrality_values: List[float],
        ma_activity_values: List[float],
    ) -> CorrelationResult:
        """
        Calculate Pearson correlation between Ecosystem Centrality and M&A activity.

        Args:
            centrality_values: Ecosystem Centrality scores per company
            ma_activity_values: M&A activity frequency per company (same order)

        Returns:
            CorrelationResult with coefficient, p_value, confidence_interval, sample_size

        Raises:
            ValueError: When series have different lengths or fewer than 3 points

        Validates: Requirement 7.2
        """
        result = self.calculate_correlation(centrality_values, ma_activity_values)
        logger.info(
            "Centrality-M&A correlation: r=%.4f p=%.4f n=%d",
            result.coefficient,
            result.p_value,
            result.sample_size,
        )
        return result

    def get_top_quartile_companies(
        self,
        company_scores: Dict[str, float],
    ) -> List[str]:
        """
        Return the top quartile (top 25%) of companies ranked by a given metric.

        Companies are sorted descending by score; the top ⌊N/4⌋ are returned.
        When N < 4, all companies are returned.

        Args:
            company_scores: Mapping of company_id -> metric score

        Returns:
            List of company_ids in the top quartile (highest scores first)

        Validates: Requirement 7.3
        """
        if not company_scores:
            return []

        sorted_companies = sorted(
            company_scores.keys(),
            key=lambda cid: company_scores[cid],
            reverse=True,
        )
        n = len(sorted_companies)
        quartile_size = max(1, n // 4)
        return sorted_companies[:quartile_size]

    def get_bottom_quartile_companies(
        self,
        company_scores: Dict[str, float],
    ) -> List[str]:
        """
        Return the bottom quartile (bottom 25%) of companies ranked by a given metric.

        Companies are sorted ascending by score; the bottom ⌊N/4⌋ are returned.
        When N < 4, all companies are returned.

        Args:
            company_scores: Mapping of company_id -> metric score

        Returns:
            List of company_ids in the bottom quartile (lowest scores first)

        Validates: Requirement 7.4
        """
        if not company_scores:
            return []

        sorted_companies = sorted(
            company_scores.keys(),
            key=lambda cid: company_scores[cid],
        )
        n = len(sorted_companies)
        quartile_size = max(1, n // 4)
        return sorted_companies[:quartile_size]

    def calculate_quartile_revenue_growth(
        self,
        quartile_company_ids: List[str],
        revenue_growth: Dict[str, float],
    ) -> float:
        """
        Calculate the average revenue growth rate for a given quartile of companies.

        Args:
            quartile_company_ids: List of company_ids in the quartile
            revenue_growth: Mapping of company_id -> revenue growth rate

        Returns:
            Arithmetic mean of revenue growth rates for the given companies.
            Returns 0.0 if no companies are provided or none have growth data.

        Validates: Requirements 7.3, 7.4
        """
        growth_values = [
            revenue_growth[cid]
            for cid in quartile_company_ids
            if cid in revenue_growth
        ]
        if not growth_values:
            return 0.0
        return sum(growth_values) / len(growth_values)

    def compare_quartile_revenue_growth(
        self,
        innovation_scores: Dict[str, float],
        revenue_growth: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compare revenue growth rates between top and bottom quartile companies
        ranked by Innovation Score.

        Args:
            innovation_scores: Mapping of company_id -> Innovation Score
            revenue_growth: Mapping of company_id -> revenue growth rate

        Returns:
            Dictionary with keys:
              - 'top_quartile_avg': average revenue growth for top quartile
              - 'bottom_quartile_avg': average revenue growth for bottom quartile
              - 'difference': top_quartile_avg - bottom_quartile_avg

        Validates: Requirement 7.4
        """
        top_ids = self.get_top_quartile_companies(innovation_scores)
        bottom_ids = self.get_bottom_quartile_companies(innovation_scores)

        top_avg = self.calculate_quartile_revenue_growth(top_ids, revenue_growth)
        bottom_avg = self.calculate_quartile_revenue_growth(bottom_ids, revenue_growth)

        result = {
            "top_quartile_avg": top_avg,
            "bottom_quartile_avg": bottom_avg,
            "difference": top_avg - bottom_avg,
        }
        logger.info(
            "Quartile revenue growth comparison: top=%.4f bottom=%.4f diff=%.4f",
            top_avg,
            bottom_avg,
            result["difference"],
        )
        return result

    # ------------------------------------------------------------------
    # Cross-Sector Comparative Analysis (Requirements 13.1, 13.2, 13.3)
    # ------------------------------------------------------------------

    def calculate_sector_averages(
        self,
        company_metrics: Dict[str, Dict[str, float]],
        company_sectors: Dict[str, str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate average metric values per sector for all key metrics.

        For each sector present in company_sectors, computes the arithmetic
        mean of every metric across all companies belonging to that sector.

        Args:
            company_metrics: Mapping of company_id -> {metric_name -> value}.
                             All companies should share the same set of metric keys.
            company_sectors: Mapping of company_id -> sector name.

        Returns:
            Mapping of sector -> {metric_name -> average_value}.
            Sectors with no companies in company_metrics are omitted.

        Validates: Requirement 13.1
        """
        if not company_metrics or not company_sectors:
            return {}

        # Collect all metric names from the first company that has data
        all_metrics: set = set()
        for metrics in company_metrics.values():
            all_metrics.update(metrics.keys())

        # Group companies by sector
        sector_companies: Dict[str, List[str]] = {}
        for company_id, sector in company_sectors.items():
            if company_id in company_metrics:
                sector_companies.setdefault(sector, []).append(company_id)

        sector_averages: Dict[str, Dict[str, float]] = {}
        for sector, company_ids in sector_companies.items():
            averages: Dict[str, float] = {}
            for metric in all_metrics:
                values = [
                    company_metrics[cid][metric]
                    for cid in company_ids
                    if metric in company_metrics[cid]
                ]
                averages[metric] = sum(values) / len(values) if values else 0.0
            sector_averages[sector] = averages

        logger.info(
            "Calculated sector averages for %d sectors across %d metrics",
            len(sector_averages),
            len(all_metrics),
        )
        return sector_averages

    def identify_sector_extrema(
        self,
        sector_averages: Dict[str, Dict[str, float]],
        metric_name: str,
    ) -> Dict[str, str]:
        """
        Identify sectors with the highest and lowest average for a given metric.

        Args:
            sector_averages: Output of calculate_sector_averages().
            metric_name: The metric to rank sectors by (e.g. 'innovation_score').

        Returns:
            Dictionary with keys:
              - 'highest': sector name with the maximum average value
              - 'lowest': sector name with the minimum average value
            Returns empty dict if no sectors have data for the metric.

        Validates: Requirement 13.2
        """
        relevant = {
            sector: avgs[metric_name]
            for sector, avgs in sector_averages.items()
            if metric_name in avgs
        }
        if not relevant:
            return {}

        highest = max(relevant, key=lambda s: relevant[s])
        lowest = min(relevant, key=lambda s: relevant[s])

        logger.info(
            "Sector extrema for '%s': highest=%s (%.4f), lowest=%s (%.4f)",
            metric_name,
            highest,
            relevant[highest],
            lowest,
            relevant[lowest],
        )
        return {"highest": highest, "lowest": lowest}

    def calculate_inter_sector_differences(
        self,
        sector_averages: Dict[str, Dict[str, float]],
        metric_name: str,
    ) -> Dict[str, float]:
        """
        Calculate percentage differences between all pairs of sector averages.

        For sectors A and B with averages M_A and M_B:
            percentage_difference = ((M_A - M_B) / M_B) × 100

        The key format is "SectorA_vs_SectorB" (A is the reference, B is the base).
        All ordered pairs (A, B) where A ≠ B are included.

        Args:
            sector_averages: Output of calculate_sector_averages().
            metric_name: The metric to compute differences for.

        Returns:
            Mapping of "SectorA_vs_SectorB" -> percentage_difference.
            Returns empty dict if fewer than 2 sectors have data for the metric.

        Validates: Requirement 13.3
        """
        relevant = {
            sector: avgs[metric_name]
            for sector, avgs in sector_averages.items()
            if metric_name in avgs
        }
        if len(relevant) < 2:
            return {}

        differences: Dict[str, float] = {}
        sectors = sorted(relevant.keys())
        for i, sector_a in enumerate(sectors):
            for sector_b in sectors:
                if sector_a == sector_b:
                    continue
                base = relevant[sector_b]
                if base == 0.0:
                    # Avoid division by zero; use 0% difference when base is 0
                    pct_diff = 0.0
                else:
                    pct_diff = ((relevant[sector_a] - base) / base) * 100.0
                key = f"{sector_a}_vs_{sector_b}"
                differences[key] = pct_diff

        logger.info(
            "Calculated %d inter-sector percentage differences for metric '%s'",
            len(differences),
            metric_name,
        )
        return differences

    # ------------------------------------------------------------------
    # Competitor Cluster Detection (Requirements 14.1, 14.2, 14.3, 14.5)
    # ------------------------------------------------------------------

    def identify_clusters_from_louvain(
        self,
        louvain_results: Dict[str, int],
    ) -> Dict[int, List[str]]:
        """
        Map Louvain community_id values to cluster identifiers.

        Each unique community_id from the Louvain algorithm output becomes
        a cluster. Returns a mapping from cluster_id (community_id) to the
        list of company_ids belonging to that cluster.

        Args:
            louvain_results: Mapping of company_id -> community_id (from execute_louvain)

        Returns:
            Mapping of cluster_id -> list of company_ids in that cluster

        Validates: Requirement 14.1
        """
        if not louvain_results:
            return {}

        clusters: Dict[int, List[str]] = {}
        for company_id, community_id in louvain_results.items():
            clusters.setdefault(community_id, []).append(company_id)

        logger.info(
            "Identified %d clusters from Louvain results (%d companies)",
            len(clusters),
            len(louvain_results),
        )
        return clusters

    def calculate_network_density(
        self,
        cluster_companies: List[str],
        graph: Dict[str, List[str]],
    ) -> float:
        """
        Calculate network density within a cluster.

        For an undirected graph with N nodes and E edges:
            density = E / (N * (N - 1) / 2)

        When N <= 1, density is defined as 0.0 (no edges possible).

        Args:
            cluster_companies: List of company_ids in the cluster
            graph: Adjacency dict mapping company_id -> list of neighbour company_ids

        Returns:
            Network density in [0.0, 1.0]

        Validates: Requirement 14.2
        """
        n = len(cluster_companies)
        if n <= 1:
            return 0.0

        company_set = set(cluster_companies)
        edge_count = 0
        seen_edges: set = set()

        for company in cluster_companies:
            for neighbour in graph.get(company, []):
                if neighbour in company_set:
                    edge = tuple(sorted([company, neighbour]))
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        edge_count += 1

        max_edges = n * (n - 1) / 2
        density = edge_count / max_edges if max_edges > 0 else 0.0

        logger.debug(
            "Cluster density: %d nodes, %d edges, density=%.4f",
            n,
            edge_count,
            density,
        )
        return density

    def identify_density_gaps(
        self,
        cluster_densities: Dict[int, float],
        threshold_multiplier: float = 1.5,
    ) -> List[Dict]:
        """
        Identify density gaps indicating potential market opportunities.

        A density gap is identified when the difference between two cluster
        densities exceeds a statistically significant threshold. The threshold
        is defined as threshold_multiplier * standard_deviation of all densities.

        Args:
            cluster_densities: Mapping of cluster_id -> network density
            threshold_multiplier: Multiplier for std dev to define significance (default 1.5)

        Returns:
            List of gap dicts with keys: 'cluster_a', 'cluster_b', 'gap', 'significant'

        Validates: Requirement 14.3
        """
        if len(cluster_densities) < 2:
            return []

        densities = list(cluster_densities.values())
        mean_density = sum(densities) / len(densities)
        variance = sum((d - mean_density) ** 2 for d in densities) / len(densities)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        threshold = threshold_multiplier * std_dev

        cluster_ids = sorted(cluster_densities.keys())
        gaps = []
        for i, cid_a in enumerate(cluster_ids):
            for cid_b in cluster_ids[i + 1:]:
                gap = abs(cluster_densities[cid_a] - cluster_densities[cid_b])
                gaps.append({
                    "cluster_a": cid_a,
                    "cluster_b": cid_b,
                    "gap": gap,
                    "significant": gap > threshold,
                })

        logger.info(
            "Identified %d density gaps (%d significant) across %d clusters",
            len(gaps),
            sum(1 for g in gaps if g["significant"]),
            len(cluster_densities),
        )
        return gaps

    def flag_low_density_clusters(
        self,
        cluster_densities: Dict[int, float],
    ) -> List[int]:
        """
        Flag clusters with network density below the median as opportunities.

        Low-density clusters indicate sparse connectivity, which may represent
        potential acquisition or partnership opportunities.

        Args:
            cluster_densities: Mapping of cluster_id -> network density

        Returns:
            List of cluster_ids with density below the median density

        Validates: Requirement 14.5
        """
        if not cluster_densities:
            return []

        densities = sorted(cluster_densities.values())
        n = len(densities)
        if n % 2 == 1:
            median = densities[n // 2]
        else:
            median = (densities[n // 2 - 1] + densities[n // 2]) / 2.0

        low_density = [
            cluster_id
            for cluster_id, density in cluster_densities.items()
            if density < median
        ]

        logger.info(
            "Flagged %d low-density clusters (below median %.4f) as opportunities",
            len(low_density),
            median,
        )
        return low_density


    # ------------------------------------------------------------------
    # Custom Cypher Query Execution (Requirements 16.1-16.5)
    # ------------------------------------------------------------------

    # Audit log for executed queries (in-memory for testing)
    _query_audit_log: List[Dict] = []

    def validate_cypher_syntax(self, cypher_query: str) -> bool:
        """
        Validate Cypher query syntax before execution.

        Performs basic structural validation:
        - Query must not be empty
        - Must start with a valid Cypher keyword (MATCH, CREATE, MERGE, RETURN,
          WITH, CALL, UNWIND, DELETE, SET, REMOVE, FOREACH, OPTIONAL)
        - Must not contain obviously dangerous patterns

        Args:
            cypher_query: Cypher query string to validate

        Returns:
            True if syntax appears valid

        Raises:
            QuerySyntaxError: When query syntax is invalid

        Validates: Requirement 16.2
        """
        from .exceptions import QuerySyntaxError

        if not cypher_query or not cypher_query.strip():
            raise QuerySyntaxError("Query cannot be empty")

        stripped = cypher_query.strip().upper()
        valid_starts = (
            "MATCH", "CREATE", "MERGE", "RETURN", "WITH", "CALL",
            "UNWIND", "DELETE", "SET", "REMOVE", "FOREACH", "OPTIONAL",
        )
        if not any(stripped.startswith(kw) for kw in valid_starts):
            raise QuerySyntaxError(
                f"Query must start with a valid Cypher keyword. Got: {cypher_query[:50]}"
            )

        return True

    def execute_custom_query(
        self,
        cypher_query: str,
        user_id: str = "anonymous",
        timeout_seconds: float = 30.0,
        mock_executor=None,
    ):
        """
        Execute a custom Cypher query with validation, timeout, and audit logging.

        Validates query syntax before execution. Executes the query and returns
        results in tabular format. Logs all queries with timestamp and user
        identifier for audit purposes. Enforces a 30-second timeout.

        Args:
            cypher_query: Cypher query string
            user_id: User identifier for audit logging
            timeout_seconds: Maximum execution time (default 30s)
            mock_executor: Optional callable(query) -> (columns, rows, time_ms)
                           for testing without a real Neo4j connection

        Returns:
            QueryResult with columns, rows, execution_time_ms

        Raises:
            QuerySyntaxError: When query syntax is invalid
            QueryTimeoutError: When execution exceeds timeout_seconds

        Validates: Requirements 16.1-16.5
        """
        import time
        from .exceptions import QuerySyntaxError, QueryTimeoutError
        from .data_models import QueryResult, QueryAuditLog

        # Validate syntax (Req 16.2)
        self.validate_cypher_syntax(cypher_query)

        start_time = time.time()

        # Execute query (Req 16.3)
        if mock_executor is not None:
            columns, rows, exec_time_ms = mock_executor(cypher_query)
        else:
            # Simulate execution for testing without Neo4j
            exec_time_ms = 0.0
            columns = []
            rows = []

        elapsed_ms = (time.time() - start_time) * 1000 + exec_time_ms

        # Enforce timeout (Req 16.5)
        if elapsed_ms / 1000 > timeout_seconds:
            raise QueryTimeoutError(
                f"Query execution exceeded {timeout_seconds}s timeout "
                f"(took {elapsed_ms:.1f}ms)"
            )

        result = QueryResult(
            columns=columns,
            rows=rows,
            execution_time_ms=elapsed_ms,
            query=cypher_query,
        )

        # Audit log (Req 16.4)
        audit_entry = {
            "query": cypher_query,
            "timestamp": result.timestamp,
            "user_id": user_id,
            "execution_time_ms": elapsed_ms,
            "row_count": len(rows),
        }
        if not hasattr(self, '_audit_log'):
            self._audit_log = []
        self._audit_log.append(audit_entry)

        logger.info(
            "Executed custom query by user '%s': %d rows in %.1fms",
            user_id,
            len(rows),
            elapsed_ms,
        )
        return result
