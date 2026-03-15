"""Data models for Fortune 500 Knowledge Graph Analytics."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


@dataclass
class Company:
    """Company entity."""
    id: str
    name: str
    sector: str
    revenue_rank: int
    employee_count: int
    github_org: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Relationship:
    """Relationship between entities."""
    from_id: str
    to_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrawlData:
    """Structured data from Crawl4AI containing company info."""
    companies: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionResult:
    """Result of data ingestion operation."""
    node_count: int
    edge_count: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GitHubMetrics:
    """GitHub metrics for a company's organization."""
    stars: int
    forks: int
    contributors: int
    organization: str
    retrieved_at: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationError:
    """Single validation error for a company field."""
    company_id: str
    field_name: str
    error_type: str  # 'missing', 'invalid', 'out_of_range'
    error_message: str


@dataclass
class DataQualityReport:
    """Data quality validation results."""
    report_date: datetime
    total_companies: int
    companies_with_complete_data: int
    completeness_percentage: float
    missing_github_org: List[str] = field(default_factory=list)
    missing_employee_count: List[str] = field(default_factory=list)
    missing_revenue_rank: List[str] = field(default_factory=list)
    crawl4ai_records: int = 0
    github_api_records: int = 0
    github_api_failures: int = 0
    validation_errors: List[Any] = field(default_factory=list)  # List[ValidationError]


@dataclass
class MetricRecord:
    """Base class for all metric records."""
    company_id: str
    metric_name: str
    metric_value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InnovationScoreRecord(MetricRecord):
    """Innovation Score metric."""
    github_stars: int = 0
    github_forks: int = 0
    employee_count: int = 0
    normalized_score: float = 0.0  # 0-10 scale
    decile_rank: int = 0


@dataclass
class CorrelationRecord:
    """Correlation analysis result."""
    metric1: str
    metric2: str
    correlation_coefficient: float
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    timestamp: datetime


@dataclass
class EcosystemCentralityRecord(MetricRecord):
    """Ecosystem Centrality metric."""
    betweenness_centrality: float = 0.0
    pagerank_score: float = 0.0
    sector_avg_centrality: float = 0.0


@dataclass
class CorrelationResult:
    """Result of a correlation calculation."""
    coefficient: float
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int


@dataclass
class PredictionRecord:
    """ML prediction result."""
    company_id: str
    prediction_type: str  # 'revenue_growth', 'market_position'
    predicted_value: float
    confidence_score: float
    prediction_date: datetime
    target_date: datetime
    actual_value: Optional[float] = None  # Filled when outcome known


@dataclass
class DigitalMaturityRecord(MetricRecord):
    """Digital Maturity Index metric."""
    stars: int = 0
    forks: int = 0
    contributors: int = 0
    revenue_rank: int = 1
    sector: str = ""
    sector_avg: float = 0.0
    quartile: str = ""  # 'top', 'upper_mid', 'lower_mid', 'bottom'


@dataclass
class Recommendation:
    """Business recommendation."""
    priority: int  # 1 (highest) to 5 (lowest)
    category: str  # 'investment', 'acquisition', 'partnership'
    title: str
    description: str
    target_companies: List[str]
    expected_outcome: str
    confidence_level: float  # 0.0 to 1.0
    supporting_metrics: Dict[str, float]


@dataclass
class AcquisitionTarget:
    """A company identified as a potential acquisition target.

    Criteria: high Ecosystem Centrality (above sector median) and
    low market valuation (revenue_rank above sector median, i.e. lower revenue).
    """

    company_id: str
    company_name: str
    rationale: str
    metrics: Dict[str, float]


@dataclass
class ROIMetrics:
    """ROI calculation results for the analytics system.

    Attributes:
        time_savings: Monetary value of time saved vs traditional methods.
        revenue_impact: Difference between top and bottom quartile average revenue.
        decision_speed_improvement: Percentage improvement in decision-making speed.
        knowledge_loss_avoidance: Estimated value of knowledge loss avoided.
        total_benefits: Sum of all quantified benefit categories.
        system_costs: Total cost of the analytics system.
        roi_ratio: total_benefits / system_costs.
    """

    time_savings: float
    revenue_impact: float
    decision_speed_improvement: float  # Percentage
    knowledge_loss_avoidance: float
    total_benefits: float
    system_costs: float
    roi_ratio: float  # benefits / costs


# ---------------------------------------------------------------------------
# Executive Report data models (Requirement 11)
# ---------------------------------------------------------------------------

@dataclass
class MetricsSummary:
    """Aggregated metrics overview for the executive report."""
    total_companies: int
    avg_innovation_score: float
    avg_digital_maturity: float
    top_sector: str
    highest_growth_company: str
    highest_growth_rate: float


@dataclass
class LeaderboardEntry:
    """Single leaderboard entry ranked by Innovation Score."""
    rank: int
    company_name: str
    sector: str
    innovation_score: float
    digital_maturity: float
    ecosystem_centrality: float
    yoy_change: float  # Year-over-year change


@dataclass
class InflectionPoint:
    """A point where a metric trend changes direction."""
    timestamp: datetime
    metric_name: str
    value: float
    direction_change: str  # 'increasing_to_decreasing' or 'decreasing_to_increasing'


@dataclass
class TrendsAnalysis:
    """Historical trends analysis section of the executive report."""
    innovation_score_trend: List[Tuple[datetime, float]]
    digital_maturity_trend: List[Tuple[datetime, float]]
    sector_trends: Dict[str, List[Tuple[datetime, float]]]
    inflection_points: List[InflectionPoint]


@dataclass
class ROIAnalysis:
    """ROI analysis section of the executive report."""
    time_savings_hours: float
    time_savings_value: float
    revenue_impact_top_quartile: float
    revenue_impact_bottom_quartile: float
    decision_speed_improvement: float
    knowledge_loss_avoidance: float
    total_benefits: float
    system_costs: float
    roi_ratio: float


@dataclass
class SectorAnalysis:
    """Results of cross-sector comparative analysis.

    Attributes:
        sector_averages: Mapping of sector -> {metric_name -> average_value}
        highest_sector: Sector with the highest average for the primary metric
        lowest_sector: Sector with the lowest average for the primary metric
        inter_sector_differences: Mapping of (sector_a, sector_b) -> percentage_difference
    """

    sector_averages: Dict[str, Dict[str, float]]
    highest_sector: str
    lowest_sector: str
    inter_sector_differences: Dict[str, float]  # key: "SectorA_vs_SectorB"


@dataclass
class BestPractice:
    """A best practice identified from a high-performing sector.

    Attributes:
        sector: The high-performing sector this practice comes from
        metric_name: The metric on which this sector excels
        sector_avg: The sector's average value for the metric
        overall_median: The median across all sectors for context
        description: Human-readable description of the practice
        target_sectors: Sectors that should adopt this practice (below-median performers)
    """

    sector: str
    metric_name: str
    sector_avg: float
    overall_median: float
    description: str
    target_sectors: List[str]


@dataclass
class ExecutiveReport:
    """Comprehensive executive report structure."""
    report_id: str
    generation_date: datetime
    time_period: str  # e.g., 'Q4 2024'
    metrics_summary: MetricsSummary
    leaderboard: List[LeaderboardEntry]
    trends: TrendsAnalysis
    recommendations: List[Recommendation]
    roi_analysis: ROIAnalysis
    pdf_path: Optional[str]
    html_path: Optional[str]
