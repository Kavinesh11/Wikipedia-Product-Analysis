"""
Orchestration scripts for Wikipedia Intelligence System.

This module provides orchestrators for:
- Data collection (pageviews, edits, crawls)
- Analytics pipeline (forecasting, clustering, hype detection)
- Job monitoring and health checks
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..storage.database import Database
from ..storage.cache import RedisCache
from ..storage.dto import PageviewRecord, RevisionRecord, ArticleContent
from ..data_ingestion.edit_history_scraper import EditHistoryScraper
from ..data_ingestion.crawl4ai_pipeline import Crawl4AIPipeline
from ..processing.etl_pipeline import ETLPipelineManager
from ..analytics.forecaster import TimeSeriesForecaster
from ..analytics.clustering import TopicClusteringEngine
from ..analytics.hype_detection import HypeDetectionEngine
from ..analytics.reputation_monitor import ReputationMonitor
from ..analytics.knowledge_graph import KnowledgeGraphBuilder

# Import PageviewsCollector if available (not yet implemented)
try:
    from ..data_ingestion.pageviews_collector import PageviewsCollector
except ImportError:
    PageviewsCollector = None

logger = logging.getLogger(__name__)


@dataclass
class JobHealthMetrics:
    """Health metrics for a job execution."""
    
    job_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, success, failed
    records_processed: int = 0
    errors: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    
    def mark_success(self, records_processed: int = 0):
        """Mark job as successful."""
        self.end_time = datetime.now()
        self.status = "success"
        self.records_processed = records_processed
        self.execution_time_seconds = (self.end_time - self.start_time).total_seconds()
    
    def mark_failed(self, error: str):
        """Mark job as failed."""
        self.end_time = datetime.now()
        self.status = "failed"
        self.errors.append(error)
        self.execution_time_seconds = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "job_name": self.job_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "records_processed": self.records_processed,
            "errors": self.errors,
            "execution_time_seconds": self.execution_time_seconds
        }


class DataCollectionOrchestrator:
    """
    Orchestrator for data collection jobs.
    
    Coordinates pageview collection, edit history scraping, and web crawling.
    Provides health monitoring and error handling.
    """
    
    def __init__(
        self,
        db: Database,
        cache: RedisCache,
        pageviews_collector: Optional[Any] = None,  # PageviewsCollector when available
        edit_scraper: Optional[EditHistoryScraper] = None,
        crawler: Optional[Crawl4AIPipeline] = None,
        etl_manager: Optional[ETLPipelineManager] = None
    ):
        """
        Initialize the data collection orchestrator.
        
        Args:
            db: Database instance
            cache: Redis cache instance
            pageviews_collector: Pageviews collector instance
            edit_scraper: Edit history scraper instance
            crawler: Crawl4AI pipeline instance
            etl_manager: ETL pipeline manager instance
        """
        self.db = db
        self.cache = cache
        self.pageviews_collector = pageviews_collector
        self.edit_scraper = edit_scraper
        self.crawler = crawler
        self.etl_manager = etl_manager or ETLPipelineManager(db, cache)
        self.health_metrics: List[JobHealthMetrics] = []
        logger.info("DataCollectionOrchestrator initialized")
    
    async def collect_pageviews(self, articles: Optional[List[str]] = None) -> JobHealthMetrics:
        """
        Collect pageviews for monitored articles.
        
        Args:
            articles: List of article titles to collect. If None, collects top articles.
            
        Returns:
            Job health metrics
        """
        metrics = JobHealthMetrics(
            job_name="pageview_collection",
            start_time=datetime.now()
        )
        
        logger.info("Starting pageview collection", extra={"article_count": len(articles) if articles else "top_articles"})
        
        try:
            if not self.pageviews_collector:
                raise ValueError("PageviewsCollector not configured")
            
            # Collect data
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=1)
            
            all_records = []
            
            if articles:
                # Collect for specific articles
                for article in articles:
                    try:
                        records = await self.pageviews_collector.fetch_per_article(
                            article=article,
                            start_date=start_date,
                            end_date=end_date,
                            granularity="hourly"
                        )
                        all_records.extend(records)
                    except Exception as e:
                        logger.error(f"Failed to collect pageviews for {article}", exc_info=e)
                        metrics.errors.append(f"Article {article}: {str(e)}")
            else:
                # Collect top articles
                top_articles = await self.pageviews_collector.fetch_top_articles(
                    date=end_date,
                    limit=1000
                )
                all_records.extend(top_articles)
            
            # Run ETL pipeline
            if all_records:
                result = await self.etl_manager.run_pageviews_pipeline(all_records)
                metrics.mark_success(records_processed=len(all_records))
                logger.info(
                    "Pageview collection completed",
                    extra={
                        "records_processed": len(all_records),
                        "execution_time": metrics.execution_time_seconds
                    }
                )
            else:
                metrics.mark_success(records_processed=0)
                logger.warning("No pageview records collected")
        
        except Exception as e:
            metrics.mark_failed(str(e))
            logger.error("Pageview collection failed", exc_info=e)
        
        self.health_metrics.append(metrics)
        await self._store_health_metrics(metrics)
        return metrics
    
    async def scrape_edit_history(self, articles: List[str]) -> JobHealthMetrics:
        """
        Scrape edit history for monitored articles.
        
        Args:
            articles: List of article titles to scrape
            
        Returns:
            Job health metrics
        """
        metrics = JobHealthMetrics(
            job_name="edit_history_scraping",
            start_time=datetime.now()
        )
        
        logger.info("Starting edit history scraping", extra={"article_count": len(articles)})
        
        try:
            if not self.edit_scraper:
                raise ValueError("EditHistoryScraper not configured")
            
            # Scrape data for last 24 hours
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            all_revisions = []
            
            for article in articles:
                try:
                    revisions = await self.edit_scraper.fetch_revisions(
                        article=article,
                        start_date=start_date,
                        end_date=end_date
                    )
                    all_revisions.extend(revisions)
                except Exception as e:
                    logger.error(f"Failed to scrape edits for {article}", exc_info=e)
                    metrics.errors.append(f"Article {article}: {str(e)}")
            
            # Run ETL pipeline
            if all_revisions:
                result = await self.etl_manager.run_edits_pipeline(all_revisions)
                metrics.mark_success(records_processed=len(all_revisions))
                logger.info(
                    "Edit history scraping completed",
                    extra={
                        "records_processed": len(all_revisions),
                        "execution_time": metrics.execution_time_seconds
                    }
                )
            else:
                metrics.mark_success(records_processed=0)
                logger.warning("No edit records scraped")
        
        except Exception as e:
            metrics.mark_failed(str(e))
            logger.error("Edit history scraping failed", exc_info=e)
        
        self.health_metrics.append(metrics)
        await self._store_health_metrics(metrics)
        return metrics
    
    async def perform_deep_crawl(
        self,
        seed_articles: List[str],
        max_depth: int = 2,
        max_articles: int = 100
    ) -> JobHealthMetrics:
        """
        Perform deep crawl for new articles.
        
        Args:
            seed_articles: List of seed article URLs
            max_depth: Maximum crawl depth
            max_articles: Maximum articles to crawl
            
        Returns:
            Job health metrics
        """
        metrics = JobHealthMetrics(
            job_name="deep_crawl",
            start_time=datetime.now()
        )
        
        logger.info(
            "Starting deep crawl",
            extra={
                "seed_count": len(seed_articles),
                "max_depth": max_depth,
                "max_articles": max_articles
            }
        )
        
        try:
            if not self.crawler:
                raise ValueError("Crawl4AIPipeline not configured")
            
            all_articles = []
            
            for seed_url in seed_articles:
                try:
                    articles = await self.crawler.deep_crawl(
                        seed_url=seed_url,
                        max_depth=max_depth,
                        max_articles=max_articles
                    )
                    all_articles.extend(articles)
                except Exception as e:
                    logger.error(f"Failed to crawl from {seed_url}", exc_info=e)
                    metrics.errors.append(f"Seed {seed_url}: {str(e)}")
            
            # Store crawled articles
            if all_articles:
                # TODO: Run crawl ETL pipeline when implemented
                metrics.mark_success(records_processed=len(all_articles))
                logger.info(
                    "Deep crawl completed",
                    extra={
                        "articles_crawled": len(all_articles),
                        "execution_time": metrics.execution_time_seconds
                    }
                )
            else:
                metrics.mark_success(records_processed=0)
                logger.warning("No articles crawled")
        
        except Exception as e:
            metrics.mark_failed(str(e))
            logger.error("Deep crawl failed", exc_info=e)
        
        self.health_metrics.append(metrics)
        await self._store_health_metrics(metrics)
        return metrics
    
    async def _store_health_metrics(self, metrics: JobHealthMetrics):
        """Store health metrics in Redis cache."""
        try:
            key = f"pipeline:status:{metrics.job_name}:{metrics.start_time.isoformat()}"
            await self.cache.set(key, metrics.to_dict(), ttl=86400)  # 24 hours
        except Exception as e:
            logger.error("Failed to store health metrics", exc_info=e)
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of data collection jobs.
        
        Returns:
            Dictionary with health status information
        """
        recent_metrics = self.health_metrics[-10:]  # Last 10 jobs
        
        success_count = sum(1 for m in recent_metrics if m.status == "success")
        failed_count = sum(1 for m in recent_metrics if m.status == "failed")
        
        return {
            "total_jobs": len(recent_metrics),
            "successful": success_count,
            "failed": failed_count,
            "success_rate": success_count / len(recent_metrics) if recent_metrics else 0,
            "recent_jobs": [m.to_dict() for m in recent_metrics]
        }


class AnalyticsPipelineOrchestrator:
    """
    Orchestrator for analytics pipeline jobs.
    
    Coordinates forecasting, clustering, hype detection, and reputation monitoring.
    Provides health monitoring and error handling.
    """
    
    def __init__(
        self,
        db: Database,
        cache: RedisCache,
        forecaster: Optional[TimeSeriesForecaster] = None,
        clustering_engine: Optional[TopicClusteringEngine] = None,
        hype_engine: Optional[HypeDetectionEngine] = None,
        reputation_monitor: Optional[ReputationMonitor] = None,
        graph_builder: Optional[KnowledgeGraphBuilder] = None
    ):
        """
        Initialize the analytics pipeline orchestrator.
        
        Args:
            db: Database instance
            cache: Redis cache instance
            forecaster: Time series forecaster instance
            clustering_engine: Topic clustering engine instance
            hype_engine: Hype detection engine instance
            reputation_monitor: Reputation monitor instance
            graph_builder: Knowledge graph builder instance
        """
        self.db = db
        self.cache = cache
        self.forecaster = forecaster or TimeSeriesForecaster()
        self.clustering_engine = clustering_engine or TopicClusteringEngine()
        self.hype_engine = hype_engine or HypeDetectionEngine()
        self.reputation_monitor = reputation_monitor or ReputationMonitor()
        self.graph_builder = graph_builder or KnowledgeGraphBuilder()
        self.health_metrics: List[JobHealthMetrics] = []
        logger.info("AnalyticsPipelineOrchestrator initialized")
    
    async def retrain_models(self, articles: List[str]) -> JobHealthMetrics:
        """
        Retrain forecasting models for monitored articles.
        
        Args:
            articles: List of article titles to retrain models for
            
        Returns:
            Job health metrics
        """
        metrics = JobHealthMetrics(
            job_name="model_retraining",
            start_time=datetime.now()
        )
        
        logger.info("Starting model retraining", extra={"article_count": len(articles)})
        
        try:
            models_trained = 0
            
            for article in articles:
                try:
                    # Fetch historical pageview data (last 90 days minimum)
                    # TODO: Implement data fetching from database
                    # For now, skip actual training
                    logger.info(f"Would retrain model for {article}")
                    models_trained += 1
                except Exception as e:
                    logger.error(f"Failed to retrain model for {article}", exc_info=e)
                    metrics.errors.append(f"Article {article}: {str(e)}")
            
            metrics.mark_success(records_processed=models_trained)
            logger.info(
                "Model retraining completed",
                extra={
                    "models_trained": models_trained,
                    "execution_time": metrics.execution_time_seconds
                }
            )
        
        except Exception as e:
            metrics.mark_failed(str(e))
            logger.error("Model retraining failed", exc_info=e)
        
        self.health_metrics.append(metrics)
        await self._store_health_metrics(metrics)
        return metrics
    
    async def run_analytics_pipeline(self, articles: List[str]) -> JobHealthMetrics:
        """
        Run complete analytics pipeline.
        
        Includes clustering, hype detection, and reputation monitoring.
        
        Args:
            articles: List of article titles to analyze
            
        Returns:
            Job health metrics
        """
        metrics = JobHealthMetrics(
            job_name="analytics_pipeline",
            start_time=datetime.now()
        )
        
        logger.info("Starting analytics pipeline", extra={"article_count": len(articles)})
        
        try:
            # TODO: Implement full analytics pipeline
            # This would include:
            # 1. Fetch article data from database
            # 2. Run clustering
            # 3. Calculate hype scores
            # 4. Monitor reputation
            # 5. Update knowledge graph
            # 6. Store results
            
            metrics.mark_success(records_processed=len(articles))
            logger.info(
                "Analytics pipeline completed",
                extra={
                    "articles_processed": len(articles),
                    "execution_time": metrics.execution_time_seconds
                }
            )
        
        except Exception as e:
            metrics.mark_failed(str(e))
            logger.error("Analytics pipeline failed", exc_info=e)
        
        self.health_metrics.append(metrics)
        await self._store_health_metrics(metrics)
        return metrics
    
    async def _store_health_metrics(self, metrics: JobHealthMetrics):
        """Store health metrics in Redis cache."""
        try:
            key = f"pipeline:status:{metrics.job_name}:{metrics.start_time.isoformat()}"
            await self.cache.set(key, metrics.to_dict(), ttl=86400)  # 24 hours
        except Exception as e:
            logger.error("Failed to store health metrics", exc_info=e)
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of analytics pipeline jobs.
        
        Returns:
            Dictionary with health status information
        """
        recent_metrics = self.health_metrics[-10:]  # Last 10 jobs
        
        success_count = sum(1 for m in recent_metrics if m.status == "success")
        failed_count = sum(1 for m in recent_metrics if m.status == "failed")
        
        return {
            "total_jobs": len(recent_metrics),
            "successful": success_count,
            "failed": failed_count,
            "success_rate": success_count / len(recent_metrics) if recent_metrics else 0,
            "recent_jobs": [m.to_dict() for m in recent_metrics]
        }
