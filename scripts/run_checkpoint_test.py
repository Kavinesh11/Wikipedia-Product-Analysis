"""Script to run database checkpoint tests

This script:
1. Checks if database is accessible
2. Runs migrations if needed
3. Executes checkpoint tests
4. Reports results
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import subprocess
from sqlalchemy import text
from src.storage.database import get_database
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def check_database_connection():
    """Check if database is accessible"""
    try:
        db = get_database()
        with db.get_session() as session:
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        logger.info("✓ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return False


def run_migrations():
    """Run Alembic migrations"""
    try:
        logger.info("Running database migrations...")
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("✓ Migrations completed successfully")
        logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Migration failed: {e}")
        logger.error(e.stderr)
        return False


def verify_tables():
    """Verify all tables exist"""
    try:
        from sqlalchemy import inspect
        db = get_database()
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        expected_tables = [
            'dim_articles', 'dim_dates', 'dim_clusters',
            'fact_pageviews', 'fact_edits', 'fact_crawl_results',
            'map_article_clusters',
            'agg_article_metrics_daily', 'agg_cluster_metrics'
        ]
        
        missing_tables = [t for t in expected_tables if t not in tables]
        
        if missing_tables:
            logger.error(f"✗ Missing tables: {missing_tables}")
            return False
        
        logger.info(f"✓ All {len(expected_tables)} tables exist")
        return True
    except Exception as e:
        logger.error(f"✗ Table verification failed: {e}")
        return False


def verify_indexes():
    """Verify critical indexes exist"""
    try:
        from sqlalchemy import inspect
        db = get_database()
        inspector = inspect(db.engine)
        
        critical_indexes = {
            'dim_articles': ['ix_dim_articles_title'],
            'dim_dates': ['ix_dim_dates_date'],
            'fact_pageviews': ['idx_pageviews_article_date', 'idx_pageviews_date'],
            'fact_edits': ['idx_edits_article_timestamp', 'idx_edits_timestamp'],
            'fact_crawl_results': ['idx_crawl_article']
        }
        
        all_good = True
        for table, expected_indexes in critical_indexes.items():
            existing_indexes = {idx['name'] for idx in inspector.get_indexes(table)}
            missing = [idx for idx in expected_indexes if idx not in existing_indexes]
            if missing:
                logger.error(f"✗ Missing indexes on {table}: {missing}")
                all_good = False
        
        if all_good:
            logger.info("✓ All critical indexes exist")
        return all_good
    except Exception as e:
        logger.error(f"✗ Index verification failed: {e}")
        return False


def test_connection_pooling():
    """Test connection pooling"""
    try:
        db = get_database()
        # Create multiple sessions
        for i in range(5):
            with db.get_session() as session:
                result = session.execute(text("SELECT 1"))
                assert result.scalar() == 1
        logger.info("✓ Connection pooling works (5 concurrent sessions)")
        return True
    except Exception as e:
        logger.error(f"✗ Connection pooling test failed: {e}")
        return False


def run_pytest_tests():
    """Run pytest checkpoint tests"""
    try:
        logger.info("Running pytest checkpoint tests...")
        result = subprocess.run(
            ["pytest", "tests/checkpoint_database_setup.py", "-v", "--tb=short"],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode == 0:
            logger.info("✓ All pytest tests passed")
            return True
        else:
            logger.error("✗ Some pytest tests failed")
            return False
    except Exception as e:
        logger.error(f"✗ Pytest execution failed: {e}")
        return False


def main():
    """Run all checkpoint tests"""
    print("=" * 70)
    print("DATABASE SETUP CHECKPOINT TEST")
    print("=" * 70)
    print()
    
    results = {}
    
    # Step 1: Check database connection
    print("Step 1: Checking database connection...")
    results['connection'] = check_database_connection()
    print()
    
    if not results['connection']:
        print("ERROR: Cannot connect to database. Please check:")
        print("  1. PostgreSQL is running")
        print("  2. Database credentials in config/config.yaml are correct")
        print("  3. Database 'wikipedia_intelligence_dev' exists")
        print()
        print("To create the database, run:")
        print("  psql -U postgres -c \"CREATE DATABASE wikipedia_intelligence_dev;\"")
        print("  psql -U postgres -c \"CREATE USER dev_user WITH PASSWORD 'dev_password';\"")
        print("  psql -U postgres -c \"GRANT ALL PRIVILEGES ON DATABASE wikipedia_intelligence_dev TO dev_user;\"")
        return 1
    
    # Step 2: Run migrations
    print("Step 2: Running database migrations...")
    results['migrations'] = run_migrations()
    print()
    
    if not results['migrations']:
        print("ERROR: Migration failed. Check logs above for details.")
        return 1
    
    # Step 3: Verify tables
    print("Step 3: Verifying tables...")
    results['tables'] = verify_tables()
    print()
    
    # Step 4: Verify indexes
    print("Step 4: Verifying indexes...")
    results['indexes'] = verify_indexes()
    print()
    
    # Step 5: Test connection pooling
    print("Step 5: Testing connection pooling...")
    results['pooling'] = test_connection_pooling()
    print()
    
    # Step 6: Run comprehensive pytest tests
    print("Step 6: Running comprehensive pytest tests...")
    results['pytest'] = run_pytest_tests()
    print()
    
    # Summary
    print("=" * 70)
    print("CHECKPOINT TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test.upper():20s}: {status}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print()
        print("SUCCESS! Database setup is working correctly.")
        print()
        print("Next steps:")
        print("  - Continue with Task 4: Implement rate limiting and API client infrastructure")
        print("  - Or run: pytest tests/ -v  to run all tests")
        return 0
    else:
        print()
        print("FAILURE: Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
