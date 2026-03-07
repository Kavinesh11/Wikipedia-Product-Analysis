"""SQLAlchemy Data Models

Defines database schema for the Wikipedia Intelligence System.
Includes dimension tables, fact tables, and aggregated metrics tables.

Note: Uses JSON and Text types for cross-database compatibility (PostgreSQL and SQLite).
"""
from datetime import datetime
from typing import List
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Date, Text,
    ForeignKey, Index, CheckConstraint, UniqueConstraint, ARRAY, BigInteger, JSON
)
from sqlalchemy.orm import relationship
from src.storage.database import Base


# ============================================================================
# DIMENSION TABLES
# ============================================================================

class DimArticle(Base):
    """Articles dimension table
    
    Stores metadata about Wikipedia articles being tracked.
    """
    __tablename__ = "dim_articles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False, unique=True, index=True)
    url = Column(Text, nullable=False)
    namespace = Column(String(50))
    first_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    pageviews = relationship("FactPageview", back_populates="article")
    edits = relationship("FactEdit", back_populates="article")
    crawl_results = relationship("FactCrawlResult", back_populates="article")
    daily_metrics = relationship("AggArticleMetricsDaily", back_populates="article")
    clusters = relationship("MapArticleCluster", back_populates="article")
    
    def __repr__(self):
        return f"<DimArticle(id={self.id}, title='{self.title}')>"


class DimDate(Base):
    """Dates dimension table
    
    Provides date hierarchy for time-based analysis.
    """
    __tablename__ = "dim_dates"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    year = Column(Integer, nullable=False)
    quarter = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    week = Column(Integer, nullable=False)
    day_of_week = Column(Integer, nullable=False)
    is_weekend = Column(Boolean, nullable=False)
    
    # Relationships
    pageviews = relationship("FactPageview", back_populates="date_dim")
    
    def __repr__(self):
        return f"<DimDate(id={self.id}, date={self.date})>"


class DimCluster(Base):
    """Clusters dimension table
    
    Stores topic clusters for industry grouping.
    """
    __tablename__ = "dim_clusters"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_name = Column(String(200), nullable=False)
    industry = Column(String(100))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    articles = relationship("MapArticleCluster", back_populates="cluster")
    metrics = relationship("AggClusterMetrics", back_populates="cluster")
    
    def __repr__(self):
        return f"<DimCluster(id={self.id}, name='{self.cluster_name}')>"


class MapArticleCluster(Base):
    """Article-Cluster mapping table
    
    Many-to-many relationship between articles and clusters.
    """
    __tablename__ = "map_article_clusters"
    
    article_id = Column(Integer, ForeignKey("dim_articles.id"), primary_key=True)
    cluster_id = Column(Integer, ForeignKey("dim_clusters.id"), primary_key=True)
    confidence_score = Column(Float, CheckConstraint("confidence_score >= 0 AND confidence_score <= 1"))
    
    # Relationships
    article = relationship("DimArticle", back_populates="clusters")
    cluster = relationship("DimCluster", back_populates="articles")
    
    def __repr__(self):
        return f"<MapArticleCluster(article_id={self.article_id}, cluster_id={self.cluster_id})>"



# ============================================================================
# FACT TABLES
# ============================================================================

class FactPageview(Base):
    """Pageviews fact table
    
    Stores article traffic statistics with device segmentation.
    Partitioned by date for query performance.
    """
    __tablename__ = "fact_pageviews"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey("dim_articles.id"), nullable=False, index=True)
    date_id = Column(Integer, ForeignKey("dim_dates.id"), nullable=False, index=True)
    hour = Column(Integer, CheckConstraint("hour >= 0 AND hour < 24"))
    device_type = Column(String(20), nullable=False)
    views_human = Column(Integer, nullable=False, default=0)
    views_bot = Column(Integer, nullable=False, default=0)
    views_total = Column(Integer, nullable=False, default=0)
    
    # Relationships
    article = relationship("DimArticle", back_populates="pageviews")
    date_dim = relationship("DimDate", back_populates="pageviews")
    
    # Composite unique constraint
    __table_args__ = (
        UniqueConstraint("article_id", "date_id", "hour", "device_type", 
                        name="uq_pageviews_article_date_hour_device"),
        Index("idx_pageviews_article_date", "article_id", "date_id"),
        Index("idx_pageviews_date", "date_id"),
    )
    
    def __repr__(self):
        return f"<FactPageview(id={self.id}, article_id={self.article_id}, views={self.views_total})>"


class FactEdit(Base):
    """Edits fact table
    
    Stores Wikipedia revision history and edit metadata.
    """
    __tablename__ = "fact_edits"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey("dim_articles.id"), nullable=False, index=True)
    revision_id = Column(BigInteger, nullable=False, unique=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    editor_type = Column(String(20), nullable=False)  # "anonymous" or "registered"
    is_reverted = Column(Boolean, default=False)
    bytes_changed = Column(Integer)
    edit_summary = Column(Text)
    
    # Relationships
    article = relationship("DimArticle", back_populates="edits")
    
    # Indexes
    __table_args__ = (
        Index("idx_edits_article_timestamp", "article_id", "timestamp"),
        Index("idx_edits_timestamp", "timestamp"),
    )
    
    def __repr__(self):
        return f"<FactEdit(id={self.id}, article_id={self.article_id}, revision_id={self.revision_id})>"


class FactCrawlResult(Base):
    """Crawl results fact table
    
    Stores extracted content and metadata from Wikipedia articles.
    """
    __tablename__ = "fact_crawl_results"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey("dim_articles.id"), nullable=False, index=True)
    crawl_timestamp = Column(DateTime, nullable=False)
    content_length = Column(Integer)
    infobox_data = Column(JSON)  # Use JSON for cross-database compatibility
    categories = Column(Text)  # Store as JSON string for SQLite compatibility
    internal_links = Column(Text)  # Store as JSON string for SQLite compatibility
    tables_count = Column(Integer, default=0)
    
    # Relationships
    article = relationship("DimArticle", back_populates="crawl_results")
    
    # Indexes
    __table_args__ = (
        Index("idx_crawl_article", "article_id"),
    )
    
    def __repr__(self):
        return f"<FactCrawlResult(id={self.id}, article_id={self.article_id})>"



# ============================================================================
# AGGREGATED METRICS TABLES
# ============================================================================

class AggArticleMetricsDaily(Base):
    """Daily article metrics aggregation table
    
    Pre-computed daily metrics for fast dashboard queries.
    """
    __tablename__ = "agg_article_metrics_daily"
    
    article_id = Column(Integer, ForeignKey("dim_articles.id"), primary_key=True)
    date = Column(Date, primary_key=True)
    total_views = Column(Integer, nullable=False)
    view_growth_rate = Column(Float)
    edit_count = Column(Integer, default=0)
    edit_velocity = Column(Float)
    hype_score = Column(Float)
    reputation_risk = Column(Float)
    
    # Relationships
    article = relationship("DimArticle", back_populates="daily_metrics")
    
    def __repr__(self):
        return f"<AggArticleMetricsDaily(article_id={self.article_id}, date={self.date})>"


class AggClusterMetrics(Base):
    """Cluster metrics aggregation table
    
    Pre-computed cluster-level metrics for industry analysis.
    """
    __tablename__ = "agg_cluster_metrics"
    
    cluster_id = Column(Integer, ForeignKey("dim_clusters.id"), primary_key=True)
    date = Column(Date, primary_key=True)
    total_views = Column(Integer, nullable=False)
    article_count = Column(Integer, nullable=False)
    avg_growth_rate = Column(Float)
    topic_cagr = Column(Float)
    
    # Relationships
    cluster = relationship("DimCluster", back_populates="metrics")
    
    def __repr__(self):
        return f"<AggClusterMetrics(cluster_id={self.cluster_id}, date={self.date})>"

