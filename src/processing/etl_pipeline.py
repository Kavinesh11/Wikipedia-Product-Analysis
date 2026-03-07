"""ETL Pipeline Manager

Orchestrates data transformation from raw sources to analytics-ready formats.
Implements validation, deduplication, data lineage tracking, and health metrics.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import insert, select

from src.storage.database import Database
from src.storage.cache import RedisCache
from src.storage.dto import (
    PageviewRecord, RevisionRecord, ArticleContent,
    ValidationResult, PipelineResult
)
from src.storage.models import (
    DimArticle, DimDate, FactPageview, FactEdit, FactCrawlResult
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DataLineage:
    """Data lineage tracking information"""
    source: str  # "pageviews_api", "edit_history", "crawl4ai"
    source_timestamp: datetime
    pipeline_name: str
    transformation_steps: List[str] = field(default_factory=list)
    loaded_timestamp: Optional[datetime] = None
    record_count: int = 0


class ETLPipelineManager:
    """Manages ETL pipelines for data ingestion and transformation
    
    Provides validation, deduplication, lineage tracking, and health metrics
    for all data pipelines.
    """
    
    def __init__(self, db: Database, cache: RedisCache):
        """Initialize ETL Pipeline Manager
        
        Args:
            db: Database connection manager
            cache: Redis cache manager
        """
        self.db = db
        self.cache = cache
        self.logger = get_logger(__name__)
    
    async def run_pageviews_pipeline(
        self,
        raw_data: List[PageviewRecord]
    ) -> PipelineResult:
        """Transform pageviews data and load to warehouse
        
        Args:
            raw_data: List of pageview records from API
            
        Returns:
            Pipeline execution result
        """
        pipeline_name = "pageviews_pipeline"
        start_time = datetime.utcnow()
        
        self.logger.info(f"Starting {pipeline_name}", extra={
            "pipeline": pipeline_name,
            "record_count": len(raw_data)
        })
        
        try:
            # Step 1: Validate data
            validation_result = self.validate_data(raw_data)
            
            if not validation_result.is_valid:
                self.logger.warning(
                    f"Validation found {validation_result.invalid_records} invalid records",
                    extra={"errors": validation_result.errors}
                )
            
            # Step 2: Deduplicate records
            valid_data = [
                raw_data[i] for i in range(len(raw_data))
                if i < validation_result.valid_records or 
                   i >= len(raw_data) - (len(raw_data) - validation_result.valid_records)
            ]
            deduplicated_data = self.deduplicate(valid_data)
            
            # Step 3: Load to database
            records_loaded = 0
            records_quarantined = validation_result.invalid_records
            
            with self.db.get_session() as session:
                for record in deduplicated_data:
                    try:
                        # Get or create article
                        article = self._get_or_create_article(
                            session, record.article, f"https://en.wikipedia.org/wiki/{record.article}"
                        )
                        
                        # Get or create date dimension
                        date_dim = self._get_or_create_date(session, record.timestamp.date())
                        
                        # Insert or update pageview fact
                        self._upsert_pageview(session, record, article.id, date_dim.id)
                        
                        records_loaded += 1
                        
                    except Exception as e:
                        self.logger.error(
                            f"Failed to load pageview record: {e}",
                            extra={"record": str(record)},
                            exc_info=True
                        )
                        records_quarantined += 1
            
            # Step 4: Track lineage
            lineage = DataLineage(
                source="pageviews_api",
                source_timestamp=start_time,
                pipeline_name=pipeline_name,
                transformation_steps=["validation", "deduplication", "dimension_lookup"],
                loaded_timestamp=datetime.utcnow(),
                record_count=records_loaded
            )
            self._track_lineage(lineage)
            
            # Step 5: Record metrics
            end_time = datetime.utcnow()
            result = PipelineResult(
                pipeline_name=pipeline_name,
                status="success" if records_quarantined == 0 else "partial",
                start_time=start_time,
                end_time=end_time,
                records_processed=len(raw_data),
                records_loaded=records_loaded,
                records_quarantined=records_quarantined,
                errors=[str(e) for e in validation_result.errors]
            )
            
            self._record_pipeline_metrics(result)
            
            self.logger.info(
                f"Completed {pipeline_name}",
                extra={
                    "status": result.status,
                    "duration_seconds": result.duration_seconds,
                    "records_loaded": records_loaded
                }
            )
            
            return result
            
        except Exception as e:
            end_time = datetime.utcnow()
            self.logger.error(
                f"Pipeline {pipeline_name} failed: {e}",
                exc_info=True
            )
            
            return PipelineResult(
                pipeline_name=pipeline_name,
                status="failed",
                start_time=start_time,
                end_time=end_time,
                records_processed=len(raw_data),
                records_loaded=0,
                records_quarantined=len(raw_data),
                errors=[str(e)]
            )

    async def run_edits_pipeline(
        self,
        raw_data: List[RevisionRecord]
    ) -> PipelineResult:
        """Transform edit history data and load to warehouse
        
        Args:
            raw_data: List of revision records from Wikipedia
            
        Returns:
            Pipeline execution result
        """
        pipeline_name = "edits_pipeline"
        start_time = datetime.utcnow()
        
        self.logger.info(f"Starting {pipeline_name}", extra={
            "pipeline": pipeline_name,
            "record_count": len(raw_data)
        })
        
        try:
            # Step 1: Validate data
            validation_result = self.validate_data(raw_data)
            
            if not validation_result.is_valid:
                self.logger.warning(
                    f"Validation found {validation_result.invalid_records} invalid records",
                    extra={"errors": validation_result.errors}
                )
            
            # Step 2: Deduplicate records
            valid_data = [
                raw_data[i] for i in range(len(raw_data))
                if i < validation_result.valid_records or 
                   i >= len(raw_data) - (len(raw_data) - validation_result.valid_records)
            ]
            deduplicated_data = self.deduplicate(valid_data)
            
            # Step 3: Load to database
            records_loaded = 0
            records_quarantined = validation_result.invalid_records
            
            with self.db.get_session() as session:
                for record in deduplicated_data:
                    try:
                        # Get or create article
                        article = self._get_or_create_article(
                            session, record.article, f"https://en.wikipedia.org/wiki/{record.article}"
                        )
                        
                        # Insert or update edit fact
                        self._upsert_edit(session, record, article.id)
                        
                        records_loaded += 1
                        
                    except Exception as e:
                        self.logger.error(
                            f"Failed to load edit record: {e}",
                            extra={"record": str(record)},
                            exc_info=True
                        )
                        records_quarantined += 1
            
            # Step 4: Track lineage
            lineage = DataLineage(
                source="edit_history",
                source_timestamp=start_time,
                pipeline_name=pipeline_name,
                transformation_steps=["validation", "deduplication", "dimension_lookup"],
                loaded_timestamp=datetime.utcnow(),
                record_count=records_loaded
            )
            self._track_lineage(lineage)
            
            # Step 5: Record metrics
            end_time = datetime.utcnow()
            result = PipelineResult(
                pipeline_name=pipeline_name,
                status="success" if records_quarantined == 0 else "partial",
                start_time=start_time,
                end_time=end_time,
                records_processed=len(raw_data),
                records_loaded=records_loaded,
                records_quarantined=records_quarantined,
                errors=[str(e) for e in validation_result.errors]
            )
            
            self._record_pipeline_metrics(result)
            
            self.logger.info(
                f"Completed {pipeline_name}",
                extra={
                    "status": result.status,
                    "duration_seconds": result.duration_seconds,
                    "records_loaded": records_loaded
                }
            )
            
            return result
            
        except Exception as e:
            end_time = datetime.utcnow()
            self.logger.error(
                f"Pipeline {pipeline_name} failed: {e}",
                exc_info=True
            )
            
            return PipelineResult(
                pipeline_name=pipeline_name,
                status="failed",
                start_time=start_time,
                end_time=end_time,
                records_processed=len(raw_data),
                records_loaded=0,
                records_quarantined=len(raw_data),
                errors=[str(e)]
            )
    
    async def run_crawl_pipeline(
        self,
        raw_data: List[ArticleContent]
    ) -> PipelineResult:
        """Transform crawled content and load to warehouse
        
        Args:
            raw_data: List of article content from Crawl4AI
            
        Returns:
            Pipeline execution result
        """
        pipeline_name = "crawl_pipeline"
        start_time = datetime.utcnow()
        
        self.logger.info(f"Starting {pipeline_name}", extra={
            "pipeline": pipeline_name,
            "record_count": len(raw_data)
        })
        
        try:
            # Step 1: Validate data
            validation_result = self.validate_data(raw_data)
            
            if not validation_result.is_valid:
                self.logger.warning(
                    f"Validation found {validation_result.invalid_records} invalid records",
                    extra={"errors": validation_result.errors}
                )
            
            # Step 2: Deduplicate records
            valid_data = [
                raw_data[i] for i in range(len(raw_data))
                if i < validation_result.valid_records or 
                   i >= len(raw_data) - (len(raw_data) - validation_result.valid_records)
            ]
            deduplicated_data = self.deduplicate(valid_data)
            
            # Step 3: Load to database
            records_loaded = 0
            records_quarantined = validation_result.invalid_records
            
            with self.db.get_session() as session:
                for record in deduplicated_data:
                    try:
                        # Get or create article
                        article = self._get_or_create_article(
                            session, record.title, record.url
                        )
                        
                        # Insert crawl result
                        self._insert_crawl_result(session, record, article.id)
                        
                        records_loaded += 1
                        
                    except Exception as e:
                        self.logger.error(
                            f"Failed to load crawl record: {e}",
                            extra={"record": str(record)},
                            exc_info=True
                        )
                        records_quarantined += 1
            
            # Step 4: Track lineage
            lineage = DataLineage(
                source="crawl4ai",
                source_timestamp=start_time,
                pipeline_name=pipeline_name,
                transformation_steps=["validation", "deduplication", "dimension_lookup"],
                loaded_timestamp=datetime.utcnow(),
                record_count=records_loaded
            )
            self._track_lineage(lineage)
            
            # Step 5: Record metrics
            end_time = datetime.utcnow()
            result = PipelineResult(
                pipeline_name=pipeline_name,
                status="success" if records_quarantined == 0 else "partial",
                start_time=start_time,
                end_time=end_time,
                records_processed=len(raw_data),
                records_loaded=records_loaded,
                records_quarantined=records_quarantined,
                errors=[str(e) for e in validation_result.errors]
            )
            
            self._record_pipeline_metrics(result)
            
            self.logger.info(
                f"Completed {pipeline_name}",
                extra={
                    "status": result.status,
                    "duration_seconds": result.duration_seconds,
                    "records_loaded": records_loaded
                }
            )
            
            return result
            
        except Exception as e:
            end_time = datetime.utcnow()
            self.logger.error(
                f"Pipeline {pipeline_name} failed: {e}",
                exc_info=True
            )
            
            return PipelineResult(
                pipeline_name=pipeline_name,
                status="failed",
                start_time=start_time,
                end_time=end_time,
                records_processed=len(raw_data),
                records_loaded=0,
                records_quarantined=len(raw_data),
                errors=[str(e)]
            )

    def validate_data(
        self,
        data: List[Any]
    ) -> ValidationResult:
        """Validate data against schema and business rules
        
        Args:
            data: List of data records to validate
            
        Returns:
            Validation result with error details
        """
        valid_records = 0
        invalid_records = 0
        errors = []
        
        for i, record in enumerate(data):
            try:
                # Type-specific validation
                if isinstance(record, PageviewRecord):
                    self._validate_pageview_record(record)
                elif isinstance(record, RevisionRecord):
                    self._validate_revision_record(record)
                elif isinstance(record, ArticleContent):
                    self._validate_article_content(record)
                else:
                    raise ValueError(f"Unknown record type: {type(record)}")
                
                valid_records += 1
                
            except Exception as e:
                invalid_records += 1
                errors.append({
                    "record_index": i,
                    "record": str(record),
                    "error": str(e)
                })
                
                # Quarantine invalid record
                self._quarantine_record(record, str(e))
        
        is_valid = invalid_records == 0
        
        return ValidationResult(
            is_valid=is_valid,
            total_records=len(data),
            valid_records=valid_records,
            invalid_records=invalid_records,
            errors=errors
        )
    
    def _validate_pageview_record(self, record: PageviewRecord) -> None:
        """Validate pageview record"""
        if not record.article:
            raise ValueError("Article name is required")
        
        if record.views_total < 0:
            raise ValueError("Views cannot be negative")
        
        if record.views_human < 0 or record.views_bot < 0:
            raise ValueError("View counts cannot be negative")
        
        if record.views_total != record.views_human + record.views_bot:
            raise ValueError("Total views must equal human + bot views")
        
        if record.device_type not in ["desktop", "mobile-web", "mobile-app"]:
            raise ValueError(f"Invalid device type: {record.device_type}")
    
    def _validate_revision_record(self, record: RevisionRecord) -> None:
        """Validate revision record"""
        if not record.article:
            raise ValueError("Article name is required")
        
        if record.revision_id <= 0:
            raise ValueError("Revision ID must be positive")
        
        if record.editor_type not in ["anonymous", "registered"]:
            raise ValueError(f"Invalid editor type: {record.editor_type}")
    
    def _validate_article_content(self, record: ArticleContent) -> None:
        """Validate article content record"""
        if not record.title:
            raise ValueError("Article title is required")
        
        if not record.url or not record.url.startswith("http"):
            raise ValueError("Valid URL is required")
    
    def deduplicate(
        self,
        data: List[Any]
    ) -> List[Any]:
        """Remove duplicate records based on composite keys
        
        Args:
            data: List of data records
            
        Returns:
            Deduplicated list of records
        """
        if not data:
            return []
        
        # Type-specific deduplication
        if isinstance(data[0], PageviewRecord):
            return self._deduplicate_pageviews(data)
        elif isinstance(data[0], RevisionRecord):
            return self._deduplicate_revisions(data)
        elif isinstance(data[0], ArticleContent):
            return self._deduplicate_crawl_results(data)
        else:
            return data
    
    def _deduplicate_pageviews(self, data: List[PageviewRecord]) -> List[PageviewRecord]:
        """Deduplicate pageview records by (article, timestamp, device_type)"""
        seen = set()
        deduplicated = []
        
        for record in data:
            key = (record.article, record.timestamp, record.device_type)
            if key not in seen:
                seen.add(key)
                deduplicated.append(record)
        
        if len(deduplicated) < len(data):
            self.logger.info(
                f"Deduplicated {len(data) - len(deduplicated)} pageview records"
            )
        
        return deduplicated
    
    def _deduplicate_revisions(self, data: List[RevisionRecord]) -> List[RevisionRecord]:
        """Deduplicate revision records by revision_id"""
        seen = set()
        deduplicated = []
        
        for record in data:
            if record.revision_id not in seen:
                seen.add(record.revision_id)
                deduplicated.append(record)
        
        if len(deduplicated) < len(data):
            self.logger.info(
                f"Deduplicated {len(data) - len(deduplicated)} revision records"
            )
        
        return deduplicated
    
    def _deduplicate_crawl_results(self, data: List[ArticleContent]) -> List[ArticleContent]:
        """Deduplicate crawl results by (title, crawl_timestamp)"""
        seen = set()
        deduplicated = []
        
        for record in data:
            key = (record.title, record.crawl_timestamp)
            if key not in seen:
                seen.add(key)
                deduplicated.append(record)
        
        if len(deduplicated) < len(data):
            self.logger.info(
                f"Deduplicated {len(data) - len(deduplicated)} crawl records"
            )
        
        return deduplicated

    def _quarantine_record(self, record: Any, error: str) -> None:
        """Quarantine invalid record for manual review
        
        Args:
            record: Invalid data record
            error: Validation error message
        """
        quarantine_key = f"quarantine:{datetime.utcnow().strftime('%Y%m%d')}"
        
        quarantine_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "record_type": type(record).__name__,
            "record": str(record),
            "error": error
        }
        
        try:
            # Store in Redis list
            self.cache.set(
                f"{quarantine_key}:{id(record)}",
                quarantine_entry,
                ttl=86400 * 7  # Keep for 7 days
            )
            
            self.logger.warning(
                "Record quarantined",
                extra=quarantine_entry
            )
        except Exception as e:
            self.logger.error(
                f"Failed to quarantine record: {e}",
                exc_info=True
            )
    
    def _track_lineage(self, lineage: DataLineage) -> None:
        """Track data lineage information
        
        Args:
            lineage: Data lineage information
        """
        lineage_key = f"lineage:{lineage.pipeline_name}:{lineage.source_timestamp.isoformat()}"
        
        lineage_data = {
            "source": lineage.source,
            "source_timestamp": lineage.source_timestamp.isoformat(),
            "pipeline_name": lineage.pipeline_name,
            "transformation_steps": lineage.transformation_steps,
            "loaded_timestamp": lineage.loaded_timestamp.isoformat() if lineage.loaded_timestamp else None,
            "record_count": lineage.record_count
        }
        
        try:
            self.cache.set(
                lineage_key,
                lineage_data,
                ttl=86400 * 30  # Keep for 30 days
            )
            
            self.logger.debug(
                "Data lineage tracked",
                extra=lineage_data
            )
        except Exception as e:
            self.logger.error(
                f"Failed to track lineage: {e}",
                exc_info=True
            )
    
    def _record_pipeline_metrics(self, result: PipelineResult) -> None:
        """Record pipeline health metrics
        
        Args:
            result: Pipeline execution result
        """
        metrics_key = f"metrics:pipeline:{result.pipeline_name}"
        
        metrics = {
            "pipeline_name": result.pipeline_name,
            "status": result.status,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "duration_seconds": result.duration_seconds,
            "records_processed": result.records_processed,
            "records_loaded": result.records_loaded,
            "records_quarantined": result.records_quarantined,
            "success_rate": result.records_loaded / result.records_processed if result.records_processed > 0 else 0,
            "error_count": len(result.errors)
        }
        
        try:
            # Store latest metrics
            self.cache.set(
                f"{metrics_key}:latest",
                metrics,
                ttl=86400  # Keep for 24 hours
            )
            
            # Store historical metrics
            self.cache.set(
                f"{metrics_key}:{result.start_time.isoformat()}",
                metrics,
                ttl=86400 * 7  # Keep for 7 days
            )
            
            self.logger.info(
                "Pipeline metrics recorded",
                extra=metrics
            )
        except Exception as e:
            self.logger.error(
                f"Failed to record metrics: {e}",
                exc_info=True
            )
    
    # ========================================================================
    # Database Helper Methods
    # ========================================================================
    
    def _get_or_create_article(
        self,
        session,
        title: str,
        url: str
    ) -> DimArticle:
        """Get existing article or create new one
        
        Args:
            session: Database session
            title: Article title
            url: Article URL
            
        Returns:
            Article dimension record
        """
        # Try to find existing article
        article = session.query(DimArticle).filter_by(title=title).first()
        
        if article is None:
            # Create new article
            article = DimArticle(
                title=title,
                url=url,
                namespace="Main"
            )
            session.add(article)
            session.flush()  # Get the ID
        else:
            # Update last_updated timestamp
            article.last_updated = datetime.utcnow()
        
        return article
    
    def _get_or_create_date(self, session, date) -> DimDate:
        """Get existing date dimension or create new one
        
        Args:
            session: Database session
            date: Date object
            
        Returns:
            Date dimension record
        """
        # Try to find existing date
        date_dim = session.query(DimDate).filter_by(date=date).first()
        
        if date_dim is None:
            # Create new date dimension
            date_dim = DimDate(
                date=date,
                year=date.year,
                quarter=(date.month - 1) // 3 + 1,
                month=date.month,
                week=date.isocalendar()[1],
                day_of_week=date.weekday(),
                is_weekend=date.weekday() >= 5
            )
            session.add(date_dim)
            session.flush()  # Get the ID
        
        return date_dim
    
    def _upsert_pageview(
        self,
        session,
        record: PageviewRecord,
        article_id: int,
        date_id: int
    ) -> None:
        """Insert or update pageview fact record
        
        Args:
            session: Database session
            record: Pageview record
            article_id: Article dimension ID
            date_id: Date dimension ID
        """
        # Use PostgreSQL upsert (ON CONFLICT DO UPDATE)
        # For SQLite, we'll do a simple insert (duplicates handled by unique constraint)
        
        stmt = insert(FactPageview).values(
            article_id=article_id,
            date_id=date_id,
            hour=record.timestamp.hour,
            device_type=record.device_type,
            views_human=record.views_human,
            views_bot=record.views_bot,
            views_total=record.views_total
        )
        
        # Try PostgreSQL-specific upsert
        try:
            stmt = stmt.on_conflict_do_update(
                index_elements=['article_id', 'date_id', 'hour', 'device_type'],
                set_={
                    'views_human': record.views_human,
                    'views_bot': record.views_bot,
                    'views_total': record.views_total
                }
            )
        except AttributeError:
            # SQLite doesn't support on_conflict_do_update in the same way
            # Just do a regular insert and let unique constraint handle it
            pass
        
        session.execute(stmt)
    
    def _upsert_edit(
        self,
        session,
        record: RevisionRecord,
        article_id: int
    ) -> None:
        """Insert or update edit fact record
        
        Args:
            session: Database session
            record: Revision record
            article_id: Article dimension ID
        """
        stmt = insert(FactEdit).values(
            article_id=article_id,
            revision_id=record.revision_id,
            timestamp=record.timestamp,
            editor_type=record.editor_type,
            is_reverted=record.is_reverted,
            bytes_changed=record.bytes_changed,
            edit_summary=record.edit_summary
        )
        
        # Try PostgreSQL-specific upsert
        try:
            stmt = stmt.on_conflict_do_update(
                index_elements=['revision_id'],
                set_={
                    'is_reverted': record.is_reverted,
                    'bytes_changed': record.bytes_changed,
                    'edit_summary': record.edit_summary
                }
            )
        except AttributeError:
            # SQLite fallback
            pass
        
        session.execute(stmt)
    
    def _insert_crawl_result(
        self,
        session,
        record: ArticleContent,
        article_id: int
    ) -> None:
        """Insert crawl result fact record
        
        Args:
            session: Database session
            record: Article content record
            article_id: Article dimension ID
        """
        crawl_result = FactCrawlResult(
            article_id=article_id,
            crawl_timestamp=record.crawl_timestamp,
            content_length=len(record.summary) if record.summary else 0,
            infobox_data=record.infobox,
            categories=json.dumps(record.categories),  # Store as JSON string
            internal_links=json.dumps(record.internal_links),  # Store as JSON string
            tables_count=len(record.tables)
        )
        
        session.add(crawl_result)
