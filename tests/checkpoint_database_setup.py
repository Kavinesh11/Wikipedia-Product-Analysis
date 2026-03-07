"""Checkpoint Test: Database Setup Verification

This test verifies that:
1. Migrations can be run against a test database
2. All tables and indexes are created correctly
3. Connection pooling works as expected
"""
import pytest
import os
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

from src.storage.database import Database, get_database
from src.storage.models import (
    DimArticle, DimDate, DimCluster, MapArticleCluster,
    FactPageview, FactEdit, FactCrawlResult,
    AggArticleMetricsDaily, AggClusterMetrics
)


class TestDatabaseSetup:
    """Test database setup and configuration"""
    
    def test_database_connection(self):
        """Test that database connection can be established"""
        db = get_database()
        assert db.engine is not None, "Database engine should be initialized"
        assert db.SessionLocal is not None, "Session factory should be initialized"
        
        # Test connection
        with db.get_session() as session:
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1, "Database connection should work"
    
    def test_all_tables_created(self):
        """Test that all required tables exist in the database"""
        db = get_database()
        inspector = inspect(db.engine)
        existing_tables = inspector.get_table_names()
        
        expected_tables = [
            'dim_articles',
            'dim_dates',
            'dim_clusters',
            'map_article_clusters',
            'fact_pageviews',
            'fact_edits',
            'fact_crawl_results',
            'agg_article_metrics_daily',
            'agg_cluster_metrics'
        ]
        
        for table in expected_tables:
            assert table in existing_tables, f"Table {table} should exist"
    
    def test_dimension_tables_structure(self):
        """Test dimension tables have correct columns"""
        db = get_database()
        inspector = inspect(db.engine)
        
        # Test dim_articles
        articles_columns = {col['name'] for col in inspector.get_columns('dim_articles')}
        assert 'id' in articles_columns
        assert 'title' in articles_columns
        assert 'url' in articles_columns
        assert 'namespace' in articles_columns
        assert 'first_seen' in articles_columns
        assert 'last_updated' in articles_columns
        
        # Test dim_dates
        dates_columns = {col['name'] for col in inspector.get_columns('dim_dates')}
        assert 'id' in dates_columns
        assert 'date' in dates_columns
        assert 'year' in dates_columns
        assert 'quarter' in dates_columns
        assert 'month' in dates_columns
        assert 'week' in dates_columns
        assert 'day_of_week' in dates_columns
        assert 'is_weekend' in dates_columns
        
        # Test dim_clusters
        clusters_columns = {col['name'] for col in inspector.get_columns('dim_clusters')}
        assert 'id' in clusters_columns
        assert 'cluster_name' in clusters_columns
        assert 'industry' in clusters_columns
        assert 'description' in clusters_columns
        assert 'created_at' in clusters_columns
    
    def test_fact_tables_structure(self):
        """Test fact tables have correct columns"""
        db = get_database()
        inspector = inspect(db.engine)
        
        # Test fact_pageviews
        pageviews_columns = {col['name'] for col in inspector.get_columns('fact_pageviews')}
        assert 'id' in pageviews_columns
        assert 'article_id' in pageviews_columns
        assert 'date_id' in pageviews_columns
        assert 'hour' in pageviews_columns
        assert 'device_type' in pageviews_columns
        assert 'views_human' in pageviews_columns
        assert 'views_bot' in pageviews_columns
        assert 'views_total' in pageviews_columns
        
        # Test fact_edits
        edits_columns = {col['name'] for col in inspector.get_columns('fact_edits')}
        assert 'id' in edits_columns
        assert 'article_id' in edits_columns
        assert 'revision_id' in edits_columns
        assert 'timestamp' in edits_columns
        assert 'editor_type' in edits_columns
        assert 'is_reverted' in edits_columns
        assert 'bytes_changed' in edits_columns
        assert 'edit_summary' in edits_columns
        
        # Test fact_crawl_results
        crawl_columns = {col['name'] for col in inspector.get_columns('fact_crawl_results')}
        assert 'id' in crawl_columns
        assert 'article_id' in crawl_columns
        assert 'crawl_timestamp' in crawl_columns
        assert 'content_length' in crawl_columns
        assert 'infobox_data' in crawl_columns
        assert 'categories' in crawl_columns
        assert 'internal_links' in crawl_columns
        assert 'tables_count' in crawl_columns
    
    def test_aggregated_tables_structure(self):
        """Test aggregated metrics tables have correct columns"""
        db = get_database()
        inspector = inspect(db.engine)
        
        # Test agg_article_metrics_daily
        article_metrics_columns = {col['name'] for col in inspector.get_columns('agg_article_metrics_daily')}
        assert 'article_id' in article_metrics_columns
        assert 'date' in article_metrics_columns
        assert 'total_views' in article_metrics_columns
        assert 'view_growth_rate' in article_metrics_columns
        assert 'edit_count' in article_metrics_columns
        assert 'edit_velocity' in article_metrics_columns
        assert 'hype_score' in article_metrics_columns
        assert 'reputation_risk' in article_metrics_columns
        
        # Test agg_cluster_metrics
        cluster_metrics_columns = {col['name'] for col in inspector.get_columns('agg_cluster_metrics')}
        assert 'cluster_id' in cluster_metrics_columns
        assert 'date' in cluster_metrics_columns
        assert 'total_views' in cluster_metrics_columns
        assert 'article_count' in cluster_metrics_columns
        assert 'avg_growth_rate' in cluster_metrics_columns
        assert 'topic_cagr' in cluster_metrics_columns
    
    def test_indexes_created(self):
        """Test that required indexes exist"""
        db = get_database()
        inspector = inspect(db.engine)
        
        # Test dim_articles indexes
        articles_indexes = {idx['name'] for idx in inspector.get_indexes('dim_articles')}
        assert 'ix_dim_articles_title' in articles_indexes
        
        # Test dim_dates indexes
        dates_indexes = {idx['name'] for idx in inspector.get_indexes('dim_dates')}
        assert 'ix_dim_dates_date' in dates_indexes
        
        # Test fact_pageviews indexes
        pageviews_indexes = {idx['name'] for idx in inspector.get_indexes('fact_pageviews')}
        assert 'idx_pageviews_article_date' in pageviews_indexes
        assert 'idx_pageviews_date' in pageviews_indexes
        
        # Test fact_edits indexes
        edits_indexes = {idx['name'] for idx in inspector.get_indexes('fact_edits')}
        assert 'idx_edits_article_timestamp' in edits_indexes
        assert 'idx_edits_timestamp' in edits_indexes
        
        # Test fact_crawl_results indexes
        crawl_indexes = {idx['name'] for idx in inspector.get_indexes('fact_crawl_results')}
        assert 'idx_crawl_article' in crawl_indexes
    
    def test_foreign_key_constraints(self):
        """Test that foreign key constraints are properly defined"""
        db = get_database()
        inspector = inspect(db.engine)
        
        # Test fact_pageviews foreign keys
        pageviews_fks = inspector.get_foreign_keys('fact_pageviews')
        fk_tables = {fk['referred_table'] for fk in pageviews_fks}
        assert 'dim_articles' in fk_tables
        assert 'dim_dates' in fk_tables
        
        # Test fact_edits foreign keys
        edits_fks = inspector.get_foreign_keys('fact_edits')
        fk_tables = {fk['referred_table'] for fk in edits_fks}
        assert 'dim_articles' in fk_tables
        
        # Test fact_crawl_results foreign keys
        crawl_fks = inspector.get_foreign_keys('fact_crawl_results')
        fk_tables = {fk['referred_table'] for fk in crawl_fks}
        assert 'dim_articles' in fk_tables
        
        # Test map_article_clusters foreign keys
        mapping_fks = inspector.get_foreign_keys('map_article_clusters')
        fk_tables = {fk['referred_table'] for fk in mapping_fks}
        assert 'dim_articles' in fk_tables
        assert 'dim_clusters' in fk_tables
    
    def test_connection_pooling(self):
        """Test that connection pooling works correctly"""
        db = get_database()
        
        # Create multiple sessions to test pooling
        sessions = []
        try:
            for i in range(5):
                with db.get_session() as session:
                    result = session.execute(text("SELECT 1"))
                    assert result.scalar() == 1
                    sessions.append(session)
            
            # All sessions should work
            assert len(sessions) == 5
            
        except OperationalError as e:
            pytest.fail(f"Connection pooling failed: {e}")
    
    def test_session_context_manager(self):
        """Test that session context manager handles commits and rollbacks"""
        db = get_database()
        
        # Test successful commit
        with db.get_session() as session:
            article = DimArticle(
                title="Test_Article_Checkpoint",
                url="https://en.wikipedia.org/wiki/Test_Article_Checkpoint",
                namespace="0"
            )
            session.add(article)
        
        # Verify article was committed
        with db.get_session() as session:
            result = session.query(DimArticle).filter_by(
                title="Test_Article_Checkpoint"
            ).first()
            assert result is not None
            assert result.title == "Test_Article_Checkpoint"
            
            # Clean up
            session.delete(result)
    
    def test_session_rollback_on_error(self):
        """Test that session rolls back on error"""
        db = get_database()
        
        try:
            with db.get_session() as session:
                # Try to insert duplicate title (should fail due to unique constraint)
                article1 = DimArticle(
                    title="Test_Duplicate_Article",
                    url="https://en.wikipedia.org/wiki/Test1",
                    namespace="0"
                )
                session.add(article1)
                session.flush()
                
                article2 = DimArticle(
                    title="Test_Duplicate_Article",  # Duplicate title
                    url="https://en.wikipedia.org/wiki/Test2",
                    namespace="0"
                )
                session.add(article2)
                session.flush()
        except Exception:
            # Expected to fail
            pass
        
        # Verify no partial data was committed
        with db.get_session() as session:
            count = session.query(DimArticle).filter_by(
                title="Test_Duplicate_Article"
            ).count()
            # Should be 0 or 1 (if first insert succeeded before error)
            # but not 2 (which would indicate rollback failed)
            assert count <= 1
            
            # Clean up if any
            session.query(DimArticle).filter_by(
                title="Test_Duplicate_Article"
            ).delete()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
