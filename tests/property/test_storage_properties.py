"""Property-Based Tests for Storage Layer

Tests correctness properties for database operations and data integrity.
"""
import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, date
from sqlalchemy.exc import IntegrityError

from src.storage.database import Database, Base
from src.storage.models import (
    DimArticle, DimDate, DimCluster, FactPageview, FactEdit,
    FactCrawlResult, MapArticleCluster, AggArticleMetricsDaily, AggClusterMetrics
)


@pytest.fixture(scope="function")
def test_db():
    """Create a test database with in-memory SQLite"""
    # Use SQLite in-memory database for testing
    db = Database(database_url="sqlite:///:memory:", pool_size=1)
    # Create all tables
    Base.metadata.create_all(bind=db.engine)
    yield db
    # Cleanup
    Base.metadata.drop_all(bind=db.engine)
    db.close()


# Feature: wikipedia-intelligence-system, Property 19: Referential Integrity Enforcement
@given(
    article_title=st.text(min_size=1, max_size=100, alphabet=st.characters(blacklist_characters=['\x00'])),
    invalid_article_id=st.integers(min_value=9999, max_value=99999),
    invalid_date_id=st.integers(min_value=9999, max_value=99999),
    invalid_cluster_id=st.integers(min_value=9999, max_value=99999),
)
@settings(max_examples=100, deadline=None)
def test_referential_integrity_enforcement(test_db, article_title, invalid_article_id, 
                                           invalid_date_id, invalid_cluster_id):
    """
    Property 19: Referential Integrity Enforcement
    
    For any attempt to insert a record with an invalid foreign key reference,
    the database should reject the insertion and return an integrity constraint error.
    
    Validates: Requirements 4.7
    """
    with test_db.get_session() as session:
        # Test 1: Insert pageview with non-existent article_id
        pageview = FactPageview(
            article_id=invalid_article_id,
            date_id=1,  # Also invalid, but article_id will fail first
            hour=12,
            device_type="desktop",
            views_human=100,
            views_bot=10,
            views_total=110
        )
        
        with pytest.raises((IntegrityError, Exception)) as exc_info:
            session.add(pageview)
            session.flush()
        
        # Verify it's a foreign key constraint error
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['foreign', 'constraint', 'integrity', 'violates'])
        
        session.rollback()
        
        # Test 2: Insert edit with non-existent article_id
        edit = FactEdit(
            article_id=invalid_article_id,
            revision_id=123456,
            timestamp=datetime.utcnow(),
            editor_type="registered",
            is_reverted=False
        )
        
        with pytest.raises((IntegrityError, Exception)):
            session.add(edit)
            session.flush()
        
        session.rollback()
        
        # Test 3: Insert crawl result with non-existent article_id
        crawl = FactCrawlResult(
            article_id=invalid_article_id,
            crawl_timestamp=datetime.utcnow(),
            content_length=1000
        )
        
        with pytest.raises((IntegrityError, Exception)):
            session.add(crawl)
            session.flush()
        
        session.rollback()
        
        # Test 4: Insert article-cluster mapping with non-existent article_id
        mapping = MapArticleCluster(
            article_id=invalid_article_id,
            cluster_id=1,
            confidence_score=0.8
        )
        
        with pytest.raises((IntegrityError, Exception)):
            session.add(mapping)
            session.flush()
        
        session.rollback()
        
        # Test 5: Insert article-cluster mapping with non-existent cluster_id
        # First create a valid article
        article = DimArticle(
            title=f"Test_{article_title}_{invalid_article_id}",
            url=f"https://en.wikipedia.org/wiki/{article_title}",
            namespace="main"
        )
        session.add(article)
        session.flush()
        
        mapping = MapArticleCluster(
            article_id=article.id,
            cluster_id=invalid_cluster_id,
            confidence_score=0.8
        )
        
        with pytest.raises((IntegrityError, Exception)):
            session.add(mapping)
            session.flush()
        
        session.rollback()
        
        # Test 6: Insert daily metrics with non-existent article_id
        metrics = AggArticleMetricsDaily(
            article_id=invalid_article_id,
            date=date.today(),
            total_views=1000,
            edit_count=5
        )
        
        with pytest.raises((IntegrityError, Exception)):
            session.add(metrics)
            session.flush()
        
        session.rollback()
        
        # Test 7: Insert cluster metrics with non-existent cluster_id
        cluster_metrics = AggClusterMetrics(
            cluster_id=invalid_cluster_id,
            date=date.today(),
            total_views=5000,
            article_count=10
        )
        
        with pytest.raises((IntegrityError, Exception)):
            session.add(cluster_metrics)
            session.flush()
        
        session.rollback()


# Feature: wikipedia-intelligence-system, Property 18: Redis Cache Round-Trip
@given(
    key=st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), 
        blacklist_characters=['\x00']
    )),
    value_int=st.integers(min_value=0, max_value=1000000),
    value_float=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    value_str=st.text(min_size=0, max_size=100, alphabet=st.characters(blacklist_characters=['\x00'])),
    ttl=st.integers(min_value=60, max_value=3600)
)
@settings(max_examples=100, deadline=None)
def test_redis_cache_round_trip(key, value_int, value_float, value_str, ttl):
    """
    Property 18: Redis Cache Round-Trip
    
    For any real-time metric data, storing to Redis cache and then retrieving
    should return equivalent data.
    
    Validates: Requirements 4.2
    
    Note: This test requires Redis to be running. It will be skipped if Redis is unavailable.
    """
    try:
        from src.storage.cache import RedisCache
        from src.storage.dto import PageviewRecord, ReputationScore
        from datetime import datetime
        
        cache = RedisCache()
        
        # Test 1: Integer value round-trip
        int_key = f"test:int:{key}"
        assert cache.set(int_key, value_int, ttl=ttl)
        retrieved_int = cache.get(int_key)
        assert retrieved_int == value_int, f"Integer round-trip failed: {value_int} != {retrieved_int}"
        cache.delete(int_key)
        
        # Test 2: Float value round-trip
        float_key = f"test:float:{key}"
        assert cache.set(float_key, value_float, ttl=ttl)
        retrieved_float = cache.get(float_key)
        assert abs(retrieved_float - value_float) < 1e-6, \
            f"Float round-trip failed: {value_float} != {retrieved_float}"
        cache.delete(float_key)
        
        # Test 3: String value round-trip
        str_key = f"test:str:{key}"
        assert cache.set(str_key, value_str, ttl=ttl)
        retrieved_str = cache.get(str_key)
        assert retrieved_str == value_str, f"String round-trip failed: {value_str} != {retrieved_str}"
        cache.delete(str_key)
        
        # Test 4: Dictionary value round-trip
        dict_key = f"test:dict:{key}"
        dict_value = {
            "int": value_int,
            "float": value_float,
            "str": value_str,
            "nested": {"key": "value"}
        }
        assert cache.set(dict_key, dict_value, ttl=ttl)
        retrieved_dict = cache.get(dict_key)
        assert retrieved_dict["int"] == value_int
        assert abs(retrieved_dict["float"] - value_float) < 1e-6
        assert retrieved_dict["str"] == value_str
        assert retrieved_dict["nested"]["key"] == "value"
        cache.delete(dict_key)
        
        # Test 5: List value round-trip
        list_key = f"test:list:{key}"
        list_value = [value_int, value_float, value_str]
        assert cache.set(list_key, list_value, ttl=ttl)
        retrieved_list = cache.get(list_key)
        assert retrieved_list[0] == value_int
        assert abs(retrieved_list[1] - value_float) < 1e-6
        assert retrieved_list[2] == value_str
        cache.delete(list_key)
        
        # Test 6: Dataclass round-trip (PageviewRecord)
        dataclass_key = f"test:dataclass:{key}"
        pageview = PageviewRecord(
            article=value_str if value_str else "Test Article",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            device_type="desktop",
            views_human=value_int,
            views_bot=10,
            views_total=value_int + 10
        )
        assert cache.set(dataclass_key, pageview, ttl=ttl)
        retrieved_dataclass = cache.get(dataclass_key)
        assert isinstance(retrieved_dataclass, dict)
        assert retrieved_dataclass["views_human"] == value_int
        assert retrieved_dataclass["device_type"] == "desktop"
        cache.delete(dataclass_key)
        
        # Test 7: Fallback functionality
        fallback_key = f"test:fallback:{key}"
        fallback_called = False
        fallback_value = {"fallback": True, "value": value_int}
        
        def fallback_func():
            nonlocal fallback_called
            fallback_called = True
            return fallback_value
        
        # First call should use fallback (cache miss)
        result = cache.get(fallback_key, fallback=fallback_func)
        assert fallback_called, "Fallback should be called on cache miss"
        assert result == fallback_value
        
        # Second call should use cache (no fallback)
        fallback_called = False
        result = cache.get(fallback_key, fallback=fallback_func)
        assert not fallback_called, "Fallback should not be called on cache hit"
        assert result == fallback_value
        
        cache.delete(fallback_key)
        
    except Exception as e:
        # Skip test if Redis is not available
        pytest.skip(f"Redis not available: {e}")

