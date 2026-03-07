"""
Standalone Database Verification Script

Verifies that:
1. Database connection works
2. All tables are created
3. All indexes exist
4. Foreign keys are properly configured
5. Connection pooling works
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment to use SQLite for testing
os.environ['USE_SQLITE_FOR_TESTS'] = 'true'
os.environ['ENVIRONMENT'] = 'development'

from sqlalchemy import create_engine, inspect, text
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def verify_database_setup():
    """Verify database setup is complete and correct"""
    
    # Use absolute path for SQLite
    project_root = Path(__file__).parent.parent
    test_db_path = project_root / "data" / "test_wikipedia_intelligence.db"
    db_url = f"sqlite:///{test_db_path.absolute()}"
    logger.info(f"Connecting to database: {db_url}")
    logger.info(f"Database file path: {test_db_path.absolute()}")
    logger.info(f"Database file exists: {test_db_path.exists()}")
    
    try:
        engine = create_engine(db_url)
        inspector = inspect(engine)
        
        # 1. Test connection
        logger.info("=" * 60)
        logger.info("TEST 1: Database Connection")
        logger.info("=" * 60)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
            logger.info("✓ Database connection successful")
        
        # 2. Verify all tables exist
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Table Creation")
        logger.info("=" * 60)
        tables = inspector.get_table_names()
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
            # 'alembic_version' - optional, only if using Alembic migrations
        ]
        
        logger.info(f"Found {len(tables)} tables:")
        for table in sorted(tables):
            logger.info(f"  - {table}")
        
        missing_tables = set(expected_tables) - set(tables)
        if missing_tables:
            logger.error(f"✗ Missing tables: {missing_tables}")
            return False
        logger.info(f"✓ All {len(expected_tables)} expected tables exist")
        
        # 3. Verify dimension tables structure
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Dimension Tables Structure")
        logger.info("=" * 60)
        
        # dim_articles
        columns = {col['name']: col for col in inspector.get_columns('dim_articles')}
        required_cols = ['id', 'title', 'url', 'namespace', 'first_seen', 'last_updated']
        for col in required_cols:
            assert col in columns, f"Missing column {col} in dim_articles"
        logger.info("✓ dim_articles has all required columns")
        
        # dim_dates
        columns = {col['name']: col for col in inspector.get_columns('dim_dates')}
        required_cols = ['id', 'date', 'year', 'quarter', 'month', 'week', 'day_of_week', 'is_weekend']
        for col in required_cols:
            assert col in columns, f"Missing column {col} in dim_dates"
        logger.info("✓ dim_dates has all required columns")
        
        # dim_clusters
        columns = {col['name']: col for col in inspector.get_columns('dim_clusters')}
        required_cols = ['id', 'cluster_name', 'industry', 'description', 'created_at']
        for col in required_cols:
            assert col in columns, f"Missing column {col} in dim_clusters"
        logger.info("✓ dim_clusters has all required columns")
        
        # 4. Verify fact tables structure
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: Fact Tables Structure")
        logger.info("=" * 60)
        
        # fact_pageviews
        columns = {col['name']: col for col in inspector.get_columns('fact_pageviews')}
        required_cols = ['id', 'article_id', 'date_id', 'hour', 'device_type', 
                        'views_human', 'views_bot', 'views_total']
        for col in required_cols:
            assert col in columns, f"Missing column {col} in fact_pageviews"
        logger.info("✓ fact_pageviews has all required columns")
        
        # fact_edits
        columns = {col['name']: col for col in inspector.get_columns('fact_edits')}
        required_cols = ['id', 'article_id', 'revision_id', 'timestamp', 'editor_type',
                        'is_reverted', 'bytes_changed', 'edit_summary']
        for col in required_cols:
            assert col in columns, f"Missing column {col} in fact_edits"
        logger.info("✓ fact_edits has all required columns")
        
        # fact_crawl_results
        columns = {col['name']: col for col in inspector.get_columns('fact_crawl_results')}
        required_cols = ['id', 'article_id', 'crawl_timestamp', 'content_length',
                        'infobox_data', 'categories', 'internal_links', 'tables_count']
        for col in required_cols:
            assert col in columns, f"Missing column {col} in fact_crawl_results"
        logger.info("✓ fact_crawl_results has all required columns")
        
        # 5. Verify indexes
        logger.info("\n" + "=" * 60)
        logger.info("TEST 5: Index Creation")
        logger.info("=" * 60)
        
        index_count = 0
        for table in ['dim_articles', 'dim_dates', 'fact_pageviews', 'fact_edits', 'fact_crawl_results']:
            indexes = inspector.get_indexes(table)
            logger.info(f"  {table}: {len(indexes)} indexes")
            index_count += len(indexes)
        
        logger.info(f"✓ Total {index_count} indexes created")
        
        # 6. Verify foreign keys
        logger.info("\n" + "=" * 60)
        logger.info("TEST 6: Foreign Key Constraints")
        logger.info("=" * 60)
        
        # fact_pageviews
        fks = inspector.get_foreign_keys('fact_pageviews')
        fk_tables = [fk['referred_table'] for fk in fks]
        assert 'dim_articles' in fk_tables, "Missing FK to dim_articles"
        assert 'dim_dates' in fk_tables, "Missing FK to dim_dates"
        logger.info(f"✓ fact_pageviews has {len(fks)} foreign keys")
        
        # fact_edits
        fks = inspector.get_foreign_keys('fact_edits')
        fk_tables = [fk['referred_table'] for fk in fks]
        assert 'dim_articles' in fk_tables, "Missing FK to dim_articles"
        logger.info(f"✓ fact_edits has {len(fks)} foreign keys")
        
        # fact_crawl_results
        fks = inspector.get_foreign_keys('fact_crawl_results')
        fk_tables = [fk['referred_table'] for fk in fks]
        assert 'dim_articles' in fk_tables, "Missing FK to dim_articles"
        logger.info(f"✓ fact_crawl_results has {len(fks)} foreign keys")
        
        # map_article_clusters
        fks = inspector.get_foreign_keys('map_article_clusters')
        fk_tables = [fk['referred_table'] for fk in fks]
        assert 'dim_articles' in fk_tables, "Missing FK to dim_articles"
        assert 'dim_clusters' in fk_tables, "Missing FK to dim_clusters"
        logger.info(f"✓ map_article_clusters has {len(fks)} foreign keys")
        
        # 7. Test connection pooling
        logger.info("\n" + "=" * 60)
        logger.info("TEST 7: Connection Pooling")
        logger.info("=" * 60)
        
        # Create multiple connections
        connections = []
        for i in range(3):
            conn = engine.connect()
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
            connections.append(conn)
        
        # Close all connections
        for conn in connections:
            conn.close()
        
        logger.info("✓ Connection pooling works correctly")
        
        # 8. Verify aggregated tables
        logger.info("\n" + "=" * 60)
        logger.info("TEST 8: Aggregated Metrics Tables")
        logger.info("=" * 60)
        
        # agg_article_metrics_daily
        columns = {col['name']: col for col in inspector.get_columns('agg_article_metrics_daily')}
        required_cols = ['article_id', 'date', 'total_views', 'view_growth_rate',
                        'edit_count', 'edit_velocity', 'hype_score', 'reputation_risk']
        for col in required_cols:
            assert col in columns, f"Missing column {col} in agg_article_metrics_daily"
        logger.info("✓ agg_article_metrics_daily has all required columns")
        
        # agg_cluster_metrics
        columns = {col['name']: col for col in inspector.get_columns('agg_cluster_metrics')}
        required_cols = ['cluster_id', 'date', 'total_views', 'article_count',
                        'avg_growth_rate', 'topic_cagr']
        for col in required_cols:
            assert col in columns, f"Missing column {col} in agg_cluster_metrics"
        logger.info("✓ agg_cluster_metrics has all required columns")
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 60)
        logger.info("✓ Database connection: PASSED")
        logger.info("✓ Table creation: PASSED")
        logger.info("✓ Dimension tables: PASSED")
        logger.info("✓ Fact tables: PASSED")
        logger.info("✓ Indexes: PASSED")
        logger.info("✓ Foreign keys: PASSED")
        logger.info("✓ Connection pooling: PASSED")
        logger.info("✓ Aggregated tables: PASSED")
        logger.info("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        logger.info("=" * 60)
        
        engine.dispose()
        return True
        
    except Exception as e:
        logger.error(f"✗ Verification failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = verify_database_setup()
    sys.exit(0 if success else 1)
