"""
Checkpoint Test: Verify Database Setup

This test verifies that:
1. Migrations can be run against a test database
2. All tables and indexes are created correctly
3. Connection pooling works as expected
"""
import pytest
from sqlalchemy import inspect, text
from src.storage.database import Database, Base
from src.utils.config import get_config


class TestDatabaseCheckpoint:
    """Test database setup and migrations"""
    
    def test_database_connection(self):
        """Test that database connection can be established"""
        db = Database()
        assert db.engine is not None
        assert db.SessionLocal is not None
        
        # Test connection
        with db.get_session() as session:
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1
    
    def test_all_tables_created(self):
        """Test that all required tables exist"""
        db = Database()
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        # Expected tables from migration
        expected_tables = [
            'dim_articles',
            'dim_dates',
            'dim_clusters',
            'fact_pageviews',
            'fact_edits',
            'fact_crawl_results',
            'map_article_clusters',
            'agg_article_metrics_daily',
            'agg_cluster_metrics',
            'alembic_version'  # Alembic tracking table
        ]
        
        for table in expected_tables:
            assert table in tables, f"Table {table} not found in database"
    
    def test_dimension_tables_structure(self):
        """Test dimension tables have correct structure"""
        db = Database()
        inspector = inspect(db.engine)
        
        # Test dim_articles
        columns = {col['name']: col for col in inspector.get_columns('dim_articles')}
        assert 'id' in columns
        assert 'title' in columns
        assert 'url' in columns
        assert 'namespace' in columns
        assert 'first_seen' in columns
        assert 'last_updated' in columns
        
        # Test dim_dates
        columns = {col['name']: col for col in inspector.get_columns('dim_dates')}
        assert 'id' in columns
        assert 'date' in columns
        assert 'year' in columns
        assert 'quarter' in columns
        assert 'month' in columns
        assert 'week' in columns
        assert 'day_of_week' in columns
        assert 'is_weekend' in columns
        
        # Test dim_clusters
        columns = {col['name']: col for col in inspector.get_columns('dim_clusters')}
        assert 'id' in columns
        assert 'cluster_name' in columns
        assert 'industry' in columns
        assert 'description' in columns
        assert 'created_at' in columns
    
    def test_fact_tables_structure(self):
        """Test fact tables have correct structure"""
        db = Database()
        inspector = inspect(db.engine)
        
        # Test fact_pageviews
        columns = {col['name']: col for col in inspector.get_columns('fact_pageviews')}
        assert 'id' in columns
        assert 'article_id' in columns
        assert 'date_id' in columns
        assert 'hour' in columns
        assert 'device_type' in columns
        assert 'views_human' in columns
        assert 'views_bot' in columns
        assert 'views_total' in columns
        
        # Test fact_edits
        columns = {col['name']: col for col in inspector.get_columns('fact_edits')}
        assert 'id' in columns
        assert 'article_id' in columns
        assert 'revision_id' in columns
        assert 'timestamp' in columns
        assert 'editor_type' in columns
        assert 'is_reverted' in columns
        assert 'bytes_changed' in columns
        assert 'edit_summary' in columns
        
        # Test fact_crawl_results
        columns = {col['name']: col for col in inspector.get_columns('fact_crawl_results')}
        assert 'id' in columns
        assert 'article_id' in columns
        assert 'crawl_timestamp' in columns
        assert 'content_length' in columns
        assert 'infobox_data' in columns
        assert 'categories' in columns
        assert 'internal_links' in columns
        assert 'tables_count' in columns
    
    def test_indexes_created(self):
        """Test that all required indexes exist"""
        db = Database()
        inspector = inspect(db.engine)
        
        # Test dim_articles indexes
        indexes = {idx['name']: idx for idx in inspector.get_indexes('dim_articles')}
        assert 'ix_dim_articles_title' in indexes
        
        # Test dim_dates indexes
        indexes = {idx['name']: idx for idx in inspector.get_indexes('dim_dates')}
        assert 'ix_dim_dates_date' in indexes
        
        # Test fact_pageviews indexes
        indexes = {idx['name']: idx for idx in inspector.get_indexes('fact_pageviews')}
        assert 'idx_pageviews_article_date' in indexes
        assert 'idx_pageviews_date' in indexes
        
        # Test fact_edits indexes
        indexes = {idx['name']: idx for idx in inspector.get_indexes('fact_edits')}
        assert 'idx_edits_article_timestamp' in indexes
        assert 'idx_edits_timestamp' in indexes
        
        # Test fact_crawl_results indexes
        indexes = {idx['name']: idx for idx in inspector.get_indexes('fact_crawl_results')}
        assert 'idx_crawl_article' in indexes
    
    def test_foreign_keys_created(self):
        """Test that foreign key constraints exist"""
        db = Database()
        inspector = inspect(db.engine)
        
        # Test fact_pageviews foreign keys
        fks = inspector.get_foreign_keys('fact_pageviews')
        fk_tables = [fk['referred_table'] for fk in fks]
        assert 'dim_articles' in fk_tables
        assert 'dim_dates' in fk_tables
        
        # Test fact_edits foreign keys
        fks = inspector.get_foreign_keys('fact_edits')
        fk_tables = [fk['referred_table'] for fk in fks]
        assert 'dim_articles' in fk_tables
        
        # Test fact_crawl_results foreign keys
        fks = inspector.get_foreign_keys('fact_crawl_results')
        fk_tables = [fk['referred_table'] for fk in fks]
        assert 'dim_articles' in fk_tables
        
        # Test map_article_clusters foreign keys
        fks = inspector.get_foreign_keys('map_article_clusters')
        fk_tables = [fk['referred_table'] for fk in fks]
        assert 'dim_articles' in fk_tables
        assert 'dim_clusters' in fk_tables
    
    def test_connection_pooling(self):
        """Test that connection pooling works correctly"""
        db = Database(pool_size=5)
        
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session_context = db.get_session()
            session = session_context.__enter__()
            sessions.append((session, session_context))
            
            # Verify session works
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        # Close all sessions
        for session, context in sessions:
            context.__exit__(None, None, None)
        
        # Verify pool is still functional
        with db.get_session() as session:
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1
    
    def test_unique_constraints(self):
        """Test that unique constraints are enforced"""
        db = Database()
        inspector = inspect(db.engine)
        
        # Test dim_articles unique constraint on title
        constraints = inspector.get_unique_constraints('dim_articles')
        constraint_columns = [set(c['column_names']) for c in constraints]
        assert {'title'} in constraint_columns
        
        # Test dim_dates unique constraint on date
        constraints = inspector.get_unique_constraints('dim_dates')
        constraint_columns = [set(c['column_names']) for c in constraints]
        assert {'date'} in constraint_columns
        
        # Test fact_edits unique constraint on revision_id
        constraints = inspector.get_unique_constraints('fact_edits')
        constraint_columns = [set(c['column_names']) for c in constraints]
        assert {'revision_id'} in constraint_columns
    
    def test_aggregated_tables_structure(self):
        """Test aggregated metrics tables have correct structure"""
        db = Database()
        inspector = inspect(db.engine)
        
        # Test agg_article_metrics_daily
        columns = {col['name']: col for col in inspector.get_columns('agg_article_metrics_daily')}
        assert 'article_id' in columns
        assert 'date' in columns
        assert 'total_views' in columns
        assert 'view_growth_rate' in columns
        assert 'edit_count' in columns
        assert 'edit_velocity' in columns
        assert 'hype_score' in columns
        assert 'reputation_risk' in columns
        
        # Test agg_cluster_metrics
        columns = {col['name']: col for col in inspector.get_columns('agg_cluster_metrics')}
        assert 'cluster_id' in columns
        assert 'date' in columns
        assert 'total_views' in columns
        assert 'article_count' in columns
        assert 'avg_growth_rate' in columns
        assert 'topic_cagr' in columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
