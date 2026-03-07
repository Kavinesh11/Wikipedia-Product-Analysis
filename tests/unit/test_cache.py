"""Unit tests for Redis cache utilities"""
import pytest
from unittest.mock import Mock, patch
from src.storage.cache import RedisCache


def test_cache_initialization():
    """Test cache can be initialized with URL"""
    with patch('src.storage.cache.ConnectionPool') as mock_pool:
        with patch('src.storage.cache.redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            
            assert cache.redis_url == "redis://localhost:6379/0"
            mock_client.ping.assert_called_once()


def test_cache_get_returns_value():
    """Test cache get returns deserialized value"""
    with patch('src.storage.cache.ConnectionPool'):
        with patch('src.storage.cache.redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_client.get.return_value = '{"key": "value"}'
            mock_redis.return_value = mock_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            result = cache.get("test_key")
            
            assert result == {"key": "value"}
            mock_client.get.assert_called_with("test_key")


def test_cache_set_serializes_value():
    """Test cache set serializes value to JSON"""
    with patch('src.storage.cache.ConnectionPool'):
        with patch('src.storage.cache.redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            result = cache.set("test_key", {"key": "value"}, ttl=300)
            
            assert result is True
            mock_client.setex.assert_called_once()


def test_cache_delete_removes_key():
    """Test cache delete removes key"""
    with patch('src.storage.cache.ConnectionPool'):
        with patch('src.storage.cache.redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_client.delete.return_value = 1
            mock_redis.return_value = mock_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            result = cache.delete("test_key")
            
            assert result is True
            mock_client.delete.assert_called_with("test_key")



def test_cache_hit_scenario():
    """Test cache hit returns cached value without fallback"""
    with patch('src.storage.cache.ConnectionPool'):
        with patch('src.storage.cache.redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_client.get.return_value = b'{"cached": true, "value": 123}'
            mock_redis.return_value = mock_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            
            fallback_called = False
            def fallback_func():
                nonlocal fallback_called
                fallback_called = True
                return {"fallback": True}
            
            result = cache.get("test_key", fallback=fallback_func)
            
            assert result == {"cached": True, "value": 123}
            assert not fallback_called, "Fallback should not be called on cache hit"
            mock_client.get.assert_called_with("test_key")


def test_cache_miss_scenario():
    """Test cache miss calls fallback and caches result"""
    with patch('src.storage.cache.ConnectionPool'):
        with patch('src.storage.cache.redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_client.get.return_value = None  # Cache miss
            mock_redis.return_value = mock_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            
            fallback_called = False
            fallback_value = {"fallback": True, "value": 456}
            
            def fallback_func():
                nonlocal fallback_called
                fallback_called = True
                return fallback_value
            
            result = cache.get("test_key", fallback=fallback_func)
            
            assert result == fallback_value
            assert fallback_called, "Fallback should be called on cache miss"
            mock_client.get.assert_called_with("test_key")
            # Verify fallback result was cached
            mock_client.set.assert_called_once()


def test_cache_miss_without_fallback():
    """Test cache miss without fallback returns None"""
    with patch('src.storage.cache.ConnectionPool'):
        with patch('src.storage.cache.redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_client.get.return_value = None  # Cache miss
            mock_redis.return_value = mock_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            result = cache.get("test_key")
            
            assert result is None
            mock_client.get.assert_called_with("test_key")


def test_ttl_expiration():
    """Test cache set with TTL calls setex"""
    with patch('src.storage.cache.ConnectionPool'):
        with patch('src.storage.cache.redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            cache.set("test_key", {"data": "value"}, ttl=300)
            
            # Verify setex was called with TTL
            mock_client.setex.assert_called_once()
            call_args = mock_client.setex.call_args
            assert call_args[0][0] == "test_key"
            assert call_args[0][1] == 300  # TTL


def test_ttl_no_expiration():
    """Test cache set without TTL calls set"""
    with patch('src.storage.cache.ConnectionPool'):
        with patch('src.storage.cache.redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            cache.set("test_key", {"data": "value"})
            
            # Verify set was called without TTL
            mock_client.set.assert_called_once()
            mock_client.setex.assert_not_called()


def test_fallback_on_redis_unavailable():
    """Test fallback is used when Redis is unavailable"""
    with patch('src.storage.cache.ConnectionPool'):
        with patch('src.storage.cache.redis.Redis') as mock_redis:
            mock_client = Mock()
            # Simulate Redis error
            from redis.exceptions import RedisError
            mock_client.get.side_effect = RedisError("Connection failed")
            mock_redis.return_value = mock_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            
            fallback_called = False
            fallback_value = {"fallback": True, "value": 789}
            
            def fallback_func():
                nonlocal fallback_called
                fallback_called = True
                return fallback_value
            
            result = cache.get("test_key", fallback=fallback_func)
            
            assert result == fallback_value
            assert fallback_called, "Fallback should be called when Redis is unavailable"


def test_fallback_on_redis_unavailable_without_fallback():
    """Test None is returned when Redis is unavailable and no fallback provided"""
    with patch('src.storage.cache.ConnectionPool'):
        with patch('src.storage.cache.redis.Redis') as mock_redis:
            mock_client = Mock()
            # Simulate Redis error
            from redis.exceptions import RedisError
            mock_client.get.side_effect = RedisError("Connection failed")
            mock_redis.return_value = mock_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            result = cache.get("test_key")
            
            assert result is None


def test_dataclass_serialization():
    """Test dataclass serialization and deserialization"""
    from dataclasses import dataclass
    from datetime import datetime
    
    @dataclass
    class TestData:
        name: str
        value: int
        timestamp: datetime
    
    with patch('src.storage.cache.ConnectionPool'):
        with patch('src.storage.cache.redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            
            test_obj = TestData(name="test", value=123, timestamp=datetime(2024, 1, 1))
            cache.set("test_key", test_obj, ttl=300)
            
            # Verify set was called
            mock_client.setex.assert_called_once()


def test_helper_methods_article_metrics():
    """Test article metrics helper methods"""
    with patch('src.storage.cache.ConnectionPool'):
        with patch('src.storage.cache.redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_client.get.return_value = b'{"views": 5000, "growth": 0.15}'
            mock_redis.return_value = mock_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            
            # Test set_article_metrics
            cache.set_article_metrics(123, {"views": 5000, "growth": 0.15})
            mock_client.setex.assert_called_once()
            
            # Test get_article_metrics
            result = cache.get_article_metrics(123)
            assert result == {"views": 5000, "growth": 0.15}


def test_helper_methods_dashboard_data():
    """Test dashboard data helper methods"""
    with patch('src.storage.cache.ConnectionPool'):
        with patch('src.storage.cache.redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_client.get.return_value = b'{"data": [1, 2, 3]}'
            mock_redis.return_value = mock_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            
            # Test set_dashboard_data
            cache.set_dashboard_data("demand_trends", "abc123", {"data": [1, 2, 3]})
            mock_client.setex.assert_called_once()
            
            # Test get_dashboard_data
            result = cache.get_dashboard_data("demand_trends", "abc123")
            assert result == {"data": [1, 2, 3]}


def test_cache_key_patterns():
    """Test cache key pattern constants are defined"""
    assert hasattr(RedisCache, 'METRICS_ARTICLE_REALTIME')
    assert hasattr(RedisCache, 'METRICS_CLUSTER_REALTIME')
    assert hasattr(RedisCache, 'DASHBOARD_DEMAND_TRENDS')
    assert hasattr(RedisCache, 'DASHBOARD_COMPETITOR_COMPARISON')
    assert hasattr(RedisCache, 'DASHBOARD_TOPIC_HEATMAP')
    assert hasattr(RedisCache, 'ALERTS_REPUTATION')
    assert hasattr(RedisCache, 'ALERTS_HYPE')
    
    # Verify pattern format
    assert "{article_id}" in RedisCache.METRICS_ARTICLE_REALTIME
    assert "{cluster_id}" in RedisCache.METRICS_CLUSTER_REALTIME
    assert "{hash}" in RedisCache.DASHBOARD_DEMAND_TRENDS
