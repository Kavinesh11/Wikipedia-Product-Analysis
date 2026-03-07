"""Data Transfer Objects (DTOs)

Dataclasses for transferring data between system components.
These are lightweight objects that don't have database dependencies.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd


# ============================================================================
# DATA COLLECTION DTOs
# ============================================================================

@dataclass
class PageviewRecord:
    """Pageview data record from Wikimedia API
    
    Represents traffic statistics for a single article at a point in time.
    """
    article: str
    timestamp: datetime
    device_type: str  # "desktop", "mobile-web", "mobile-app"
    views_human: int
    views_bot: int
    views_total: int
    
    def __post_init__(self):
        """Validate data after initialization"""
        if self.views_total != self.views_human + self.views_bot:
            raise ValueError("views_total must equal views_human + views_bot")
        if self.device_type not in ["desktop", "mobile-web", "mobile-app"]:
            raise ValueError(f"Invalid device_type: {self.device_type}")


@dataclass
class RevisionRecord:
    """Edit history revision record
    
    Represents a single edit/revision to a Wikipedia article.
    """
    article: str
    revision_id: int
    timestamp: datetime
    editor_type: str  # "anonymous" or "registered"
    editor_id: str
    is_reverted: bool
    bytes_changed: int
    edit_summary: str
    
    def __post_init__(self):
        """Validate data after initialization"""
        if self.editor_type not in ["anonymous", "registered"]:
            raise ValueError(f"Invalid editor_type: {self.editor_type}")


@dataclass
class ArticleContent:
    """Crawled article content and metadata
    
    Represents structured data extracted from a Wikipedia article.
    """
    title: str
    url: str
    summary: str
    infobox: Dict[str, Any]
    tables: List[pd.DataFrame]
    categories: List[str]
    internal_links: List[str]
    crawl_timestamp: datetime
    
    def __post_init__(self):
        """Validate data after initialization"""
        if not self.url.startswith("http"):
            raise ValueError(f"Invalid URL: {self.url}")


@dataclass
class TopArticleRecord:
    """Top article record from Wikimedia API
    
    Represents an article in the top articles list.
    """
    article: str
    rank: int
    views: int
    date: datetime


@dataclass
class AggregateStats:
    """Aggregate Wikipedia traffic statistics
    
    Represents total traffic across all of Wikipedia.
    """
    start_date: datetime
    end_date: datetime
    total_views: int
    total_articles: int
    avg_views_per_article: float


# ============================================================================
# ANALYTICS DTOs
# ============================================================================

@dataclass
class SeasonalityPattern:
    """Detected seasonality pattern in time series
    
    Represents periodic patterns in pageview data.
    """
    period: str  # "weekly", "monthly", "yearly"
    strength: float  # 0-1, strength of seasonal component
    peak_day: Optional[int] = None  # Day of week (0-6) or month (1-12)


@dataclass
class ForecastResult:
    """Time series forecast result
    
    Contains predictions with confidence intervals.
    """
    article: str
    predictions: pd.DataFrame  # columns: date, yhat, yhat_lower, yhat_upper
    seasonality: SeasonalityPattern
    growth_rate: float
    confidence: float
    
    def __post_init__(self):
        """Validate predictions dataframe"""
        required_cols = ["date", "yhat", "yhat_lower", "yhat_upper"]
        if not all(col in self.predictions.columns for col in required_cols):
            raise ValueError(f"predictions must have columns: {required_cols}")


@dataclass
class ReputationScore:
    """Brand reputation risk assessment
    
    Combines multiple signals into a reputation risk score.
    """
    article: str
    risk_score: float  # 0-1, higher means more risk
    edit_velocity: float  # edits per hour
    vandalism_rate: float  # percentage of reverted edits
    anonymous_edit_pct: float  # percentage of anonymous edits
    alert_level: str  # "low", "medium", "high"
    timestamp: datetime
    
    def __post_init__(self):
        """Validate score ranges"""
        if not 0 <= self.risk_score <= 1:
            raise ValueError("risk_score must be between 0 and 1")
        if not 0 <= self.vandalism_rate <= 100:
            raise ValueError("vandalism_rate must be between 0 and 100")
        if not 0 <= self.anonymous_edit_pct <= 100:
            raise ValueError("anonymous_edit_pct must be between 0 and 100")
        if self.alert_level not in ["low", "medium", "high"]:
            raise ValueError(f"Invalid alert_level: {self.alert_level}")


@dataclass
class HypeMetrics:
    """Hype detection metrics
    
    Measures rapid attention growth and trending status.
    """
    article: str
    hype_score: float  # 0-1, composite hype metric
    view_velocity: float  # rate of pageview growth
    edit_growth: float  # rate of edit activity growth
    content_expansion: float  # rate of content size growth
    attention_density: float  # sustained attention metric
    is_trending: bool
    spike_events: List['SpikeEvent'] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate score ranges"""
        if not 0 <= self.hype_score <= 1:
            raise ValueError("hype_score must be between 0 and 1")


@dataclass
class SpikeEvent:
    """Attention spike event
    
    Represents a sudden increase in pageviews.
    """
    timestamp: datetime
    magnitude: float  # standard deviations above mean
    duration_days: int
    spike_type: str  # "sustained" or "temporary"
    
    def __post_init__(self):
        """Validate spike type"""
        if self.spike_type not in ["sustained", "temporary"]:
            raise ValueError(f"Invalid spike_type: {self.spike_type}")


@dataclass
class EditMetrics:
    """Edit pattern metrics
    
    Aggregated metrics about edit activity.
    """
    article: str
    edit_velocity: float  # edits per hour
    vandalism_rate: float  # percentage
    anonymous_edit_pct: float  # percentage
    total_edits: int
    reverted_edits: int
    time_window_hours: int


@dataclass
class ClusteringResult:
    """Article clustering result
    
    Results from topic clustering algorithm.
    """
    cluster_assignments: Dict[str, int]  # article -> cluster_id
    cluster_labels: Dict[int, str]  # cluster_id -> label
    confidence_scores: Dict[str, float]  # article -> confidence
    n_clusters: int


@dataclass
class GrowthMetrics:
    """Growth metrics for a topic cluster
    
    Measures cluster-level growth trends.
    """
    cluster_id: int
    cluster_name: str
    growth_rate: float  # percentage
    cagr: float  # compound annual growth rate
    total_views: int
    article_count: int
    is_emerging: bool


@dataclass
class ComparisonResult:
    """Industry comparison result
    
    Comparative metrics across multiple clusters.
    """
    clusters: List[GrowthMetrics]
    baseline_normalized: bool
    time_period_days: int


@dataclass
class KnowledgeGraph:
    """Knowledge graph structure
    
    Represents article relationships as a graph.
    """
    nodes: List[str]  # article titles
    edges: List[tuple]  # (source, target) pairs
    centrality_scores: Dict[str, float]  # article -> centrality
    communities: Dict[str, int]  # article -> community_id


@dataclass
class Community:
    """Detected community in knowledge graph
    
    Represents a cluster of related articles.
    """
    community_id: int
    articles: List[str]
    size: int
    density: float  # edge density within community


# ============================================================================
# ETL DTOs
# ============================================================================

@dataclass
class ValidationResult:
    """Data validation result
    
    Results from validating a dataset.
    """
    is_valid: bool
    total_records: int
    valid_records: int
    invalid_records: int
    errors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineResult:
    """ETL pipeline execution result
    
    Summary of pipeline execution.
    """
    pipeline_name: str
    status: str  # "success", "partial", "failed"
    start_time: datetime
    end_time: datetime
    records_processed: int
    records_loaded: int
    records_quarantined: int
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        """Calculate pipeline duration"""
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class Alert:
    """System alert/notification
    
    Represents an alert to be sent to users.
    """
    alert_id: str
    alert_type: str  # "reputation_risk", "hype_detected", "pipeline_failure"
    priority: str  # "low", "medium", "high", "critical"
    article: Optional[str]
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate alert fields"""
        if self.priority not in ["low", "medium", "high", "critical"]:
            raise ValueError(f"Invalid priority: {self.priority}")


@dataclass
class VandalismMetrics:
    """Vandalism detection metrics
    
    Metrics related to potential vandalism.
    """
    article: str
    total_edits: int
    reverted_edits: int
    vandalism_percentage: float
    revert_patterns: List[Dict[str, Any]] = field(default_factory=list)

