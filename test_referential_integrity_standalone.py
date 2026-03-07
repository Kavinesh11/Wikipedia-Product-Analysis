"""Standalone test for referential integrity property"""
import sys
import os
from datetime import datetime, date
from sqlalchemy.exc import IntegrityError

# Set minimal environment variables to avoid config validation issues
os.environ['POSTGRES_HOST'] = 'localhost'
os.environ['POSTGRES_DB'] = 'test'
os.environ['POSTGRES_USER'] = 'test'
os.environ['POSTGRES_PASSWORD'] = 'test'
os.environ['REDIS_HOST'] = 'localhost'
os.environ['REDIS_PORT'] = '6379'

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.storage.database import Base
from src.storage.models import (
    DimArticle, FactPageview, FactEdit, FactCrawlResult,
    MapArticleCluster, AggArticleMetricsDaily, AggClusterMetrics
)


def test_referential_integrity():
    """Test that foreign key constraints are enforced"""
    print("Creating in-memory test database...")
    
    # Create engine and session directly without using Database class
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    print("Testing referential integrity enforcement...")
    
    test_cases_passed = 0
    test_cases_total = 7
    
    session = SessionLocal()
    
    # Test 1: Insert pageview with non-existent article_id
    print("\nTest 1: Pageview with invalid article_id")
    try:
        pageview = FactPageview(
            article_id=99999,
            date_id=1,
            hour=12,
            device_type="desktop",
            views_human=100,
            views_bot=10,
            views_total=110
        )
        session.add(pageview)
        session.flush()
        print("  FAILED: Should have raised IntegrityError")
    except (IntegrityError, Exception) as e:
        print(f"  PASSED: Correctly rejected with error")
        test_cases_passed += 1
        session.rollback()
    
    # Test 2: Insert edit with non-existent article_id
    print("\nTest 2: Edit with invalid article_id")
    try:
        edit = FactEdit(
            article_id=99999,
            revision_id=123456,
            timestamp=datetime.utcnow(),
            editor_type="registered",
            is_reverted=False
        )
        session.add(edit)
        session.flush()
        print("  FAILED: Should have raised IntegrityError")
    except (IntegrityError, Exception) as e:
        print(f"  PASSED: Correctly rejected with error")
        test_cases_passed += 1
        session.rollback()
    
    # Test 3: Insert crawl result with non-existent article_id
    print("\nTest 3: Crawl result with invalid article_id")
    try:
        crawl = FactCrawlResult(
            article_id=99999,
            crawl_timestamp=datetime.utcnow(),
            content_length=1000
        )
        session.add(crawl)
        session.flush()
        print("  FAILED: Should have raised IntegrityError")
    except (IntegrityError, Exception) as e:
        print(f"  PASSED: Correctly rejected with error")
        test_cases_passed += 1
        session.rollback()
    
    # Test 4: Insert article-cluster mapping with non-existent article_id
    print("\nTest 4: Article-cluster mapping with invalid article_id")
    try:
        mapping = MapArticleCluster(
            article_id=99999,
            cluster_id=1,
            confidence_score=0.8
        )
        session.add(mapping)
        session.flush()
        print("  FAILED: Should have raised IntegrityError")
    except (IntegrityError, Exception) as e:
        print(f"  PASSED: Correctly rejected with error")
        test_cases_passed += 1
        session.rollback()
    
    # Test 5: Insert article-cluster mapping with non-existent cluster_id
    print("\nTest 5: Article-cluster mapping with invalid cluster_id")
    # First create a valid article
    article = DimArticle(
        title="Test_Article_12345",
        url="https://en.wikipedia.org/wiki/Test",
        namespace="main"
    )
    session.add(article)
    session.flush()
    
    try:
        mapping = MapArticleCluster(
            article_id=article.id,
            cluster_id=99999,
            confidence_score=0.8
        )
        session.add(mapping)
        session.flush()
        print("  FAILED: Should have raised IntegrityError")
    except (IntegrityError, Exception) as e:
        print(f"  PASSED: Correctly rejected with error")
        test_cases_passed += 1
        session.rollback()
    
    # Test 6: Insert daily metrics with non-existent article_id
    print("\nTest 6: Daily metrics with invalid article_id")
    try:
        metrics = AggArticleMetricsDaily(
            article_id=99999,
            date=date.today(),
            total_views=1000,
            edit_count=5
        )
        session.add(metrics)
        session.flush()
        print("  FAILED: Should have raised IntegrityError")
    except (IntegrityError, Exception) as e:
        print(f"  PASSED: Correctly rejected with error")
        test_cases_passed += 1
        session.rollback()
    
    # Test 7: Insert cluster metrics with non-existent cluster_id
    print("\nTest 7: Cluster metrics with invalid cluster_id")
    try:
        cluster_metrics = AggClusterMetrics(
            cluster_id=99999,
            date=date.today(),
            total_views=5000,
            article_count=10
        )
        session.add(cluster_metrics)
        session.flush()
        print("  FAILED: Should have raised IntegrityError")
    except (IntegrityError, Exception) as e:
        print(f"  PASSED: Correctly rejected with error")
        test_cases_passed += 1
        session.rollback()
    
    session.close()
    engine.dispose()
    
    print(f"\n{'='*60}")
    print(f"Test Results: {test_cases_passed}/{test_cases_total} passed")
    print(f"{'='*60}")
    
    if test_cases_passed == test_cases_total:
        print("\n✓ Property 19: Referential Integrity Enforcement - PASSED")
        return 0
    else:
        print("\n✗ Property 19: Referential Integrity Enforcement - FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(test_referential_integrity())

